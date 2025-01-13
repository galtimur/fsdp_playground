### Script for distributed training using torchrun
# to run the script:
# torchrun --nnodes 1 --nproc_per_node 8 fsdp_playground_T5.py
#


# %%
import functools
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.optim as optim
import tqdm
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer, AutoModelForCausalLM, AutoTokenizer,
)
from transformers.models.t5.modeling_t5 import T5Block

from summarization_dataset import wikihow

# 1.4 Distributed training setup. Here we use two helper functions to initialize the processes for distributed training,
# and then to clean up after training completion. In this tutorial, we are going to use torch elastic, using torchrun ,
# which will set the worker RANK and WORLD_SIZE automatically.
# Otherwise you can setup env variables manually:
# %%
# def setup():
#     # initialize the process group
#     # os.environ["MASTER_ADDR"] = "localhost"
#     # os.environ["MASTER_PORT"] = "12355"
#     dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


# 2.1 Set up the HuggingFace T5 model and some helper functions
# %%
def setup_model(model_name: str):
    if model_name.startswith("t5"):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def format_metrics_to_gb(item):
    g_gigabyte = 1024 * 1024 * 1024
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


# %%


def save_model(rank, model, file_save_name, epoch, curr_val_loss, time_of_run):
    # save
    if rank == 0:
        print(f"--> entering save model state")

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        # print(f"saving process: rank {rank}  done w state_dict")

        print(f"--> saving model ...")
        currEpoch = "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
        print(f"--> attempting to save model prefix {currEpoch}")
        save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
        print(f"--> saving as model name {save_name}")

        torch.save(cpu_state, save_name)


def get_data(rank, world_size, args, tokenizer):
    train_dataset = wikihow(args.data.folder, tokenizer, "train", 150, 512, 150, False)
    val_dataset = wikihow(args.data.folder, tokenizer, "test", 100, 512, 150, False)

    sampler_train = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler_val = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": args.training.batch_size, "sampler": sampler_train}
    test_kwargs = {"batch_size": args.training.test_batch_size, "sampler": sampler_val}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    return train_loader, val_loader, sampler_train

# %%

fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float32,
)

# %%

def train(model, rank, train_loader, optimizer, epoch, sampler=None, task="causal"):
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_loss = torch.zeros(2).to(local_rank)

    # TODO Why to do it for separate sampler but not do train_loader.sampler ?
    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        if task == "causal":
            labels = batch["source_ids"]
        elif task == "cls":
            labels = batch["target_ids"]
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank == 0:
            pbar.update(1)

    # Gather all losses
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        pbar.close()
        print(f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}")
    return train_accuracy


# %%


def validation(model, rank, val_loader, task="causal"):
    model.eval()
    correct = 0
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            if task == "causal":
                labels = batch["source_ids"]
            elif task == "cls":
                labels = batch["target_ids"]
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=labels,
            )
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank == 0:
                pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


# %%


def fsdp_main(args):

    model, tokenizer = setup_model(args.training.model_name)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # setup()

    # Gets distributed data sampler that distribute data for GPUs (defined by rank variable)
    train_loader, val_loader, sampler_train = get_data(
        rank, world_size, args, tokenizer
    )

    # Pecularity of T5 - shared in and out embedding matrices, so, we prevent their splitting.
    if args.training.model_name.startswith("t5"):
        shared_layers = {T5Block}
        task = "cls"
    else:
        shared_layers = {}
        task = "causal"

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=shared_layers,
    )
    # sharding_strategy = (
    #     ShardingStrategy.SHARD_GRAD_OP
    # )  # for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)

    mp_policy = bf16_policy
    # mp_policy = None # defaults to fp32

    # Establishes GPU connection and processes
    dist.init_process_group("nccl")
    # model is on CPU before input to FSDP
    model = FSDP(
        model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        # sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        # backward_prefetch=BackwardPrefetch.BACKWARD_PRE
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.training.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.training.gamma)

    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "T5-model-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

        if args.features.track_memory:
            mem_alloc_tracker = []
            mem_reserved_tracker = []

    for epoch in range(1, args.training.epochs + 1):
        t0 = time.time()
        train_accuracy = train(
            model,
            rank,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler_train,
            task = task
        )
        if args.features.run_validation:
            curr_val_loss = validation(model, rank, val_loader, task = task)
        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.features.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.features.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"completed save and stats zone...")

        if args.features.save_model and curr_val_loss < best_val_loss:
            save_model(rank, model, file_save_name, epoch, curr_val_loss, time_of_run)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank == 0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    cleanup()


# %%

if __name__ == "__main__":
    config = OmegaConf.load("configs/config.yaml")
    print(config.training.model_name)
    torch.manual_seed(config.training.seed)
    fsdp_main(config)

# %%
