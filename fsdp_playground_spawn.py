# %%
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import functools
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


# %% md
# 1.3 Distributed training setup. As we mentioned FSDP is a type of data parallelism which requires a distributed training environment, so here we use two helper functions to initialize the processes for distributed training and clean up.
# %%
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# %% md
# 2.1 Define our toy model for handwritten digit classification.
#
#
# %%
def prepare_dataset(rank, world_size, args, tokenizer, max_length=512):
    # Load dataset (example with wikitext, replace with your dataset)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset["train"].column_names
    )

    train_sampler = DistributedSampler(
        tokenized_dataset["train"], rank=rank, num_replicas=world_size, shuffle=True
    )
    test_sampler = DistributedSampler(
        tokenized_dataset["validation"], rank=rank, num_replicas=world_size
    )

    train_loader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=default_data_collator,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        collate_fn=default_data_collator,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, test_loader, train_sampler


# %% md
# 2.2 Define a train function
# %%
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for batch_idx, batch in pbar:
        batch = {k: v.to(rank) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch["input_ids"])
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


# %% md
# 2.3 Define a validation function
#
#
# %%
def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**batch)
            ddp_loss[0] += outputs.loss.item()
            ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[1]
        print(f"Test set: Average loss: {test_loss:.4f}\n")


# %% md
# 2.4 Define a distributed train function that wraps the model in FSDP
#
# **Note: to save the FSDP model, we need to call the state_dict on each rank then on Rank 0 save the overall states.**
# %%
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, test_loader, train_sampler = prepare_dataset(
        rank, world_size, args, tokenizer
    )
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=10_000_000
    )

    torch.cuda.set_device(rank)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # torch_dtype=torch.bfloat16,
    )
    model = model.to(rank)

    # # Wrap model with FSDP
    fsdp_config = {}
    # fsdp_config = dict(
    #     limit_all_gathers=True,
    #     mixed_precision=MixedPrecision(
    #         param_dtype=torch.bfloat16,
    #         reduce_dtype=torch.bfloat16,
    #         buffer_dtype=torch.bfloat16
    #     ),
    #     sharding_strategy=ShardingStrategy.FULL_SHARD,
    #     device_id=torch.cuda.current_device(),
    # )

    model = FSDP(
        model, auto_wrap_policy=my_auto_wrap_policy, **fsdp_config
    )  # , **fsdp_config

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=train_sampler,
        )
        test(model, rank, world_size, test_loader)

    init_end_event.record()

    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )

    if args.save_model:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "llama_finetuned.pt")

    cleanup()


# %%


if __name__ == "__main__":
    config = OmegaConf.load("configs/config_old.yaml")

    torch.manual_seed(config.training.seed)

    WORLD_SIZE = torch.cuda.device_count()
    N_PROC = WORLD_SIZE
    mp.spawn(fsdp_main, args=(N_PROC, config.training), nprocs=N_PROC, join=True)
# %%
