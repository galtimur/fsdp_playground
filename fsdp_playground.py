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
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim.lr_scheduler import StepLR
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer, AutoModelForCausalLM, AutoTokenizer,
)
from transformers.models.t5.modeling_t5 import T5Block

from train_val_scripts import train, validation
from data import get_data


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
    torch.cuda.set_device(local_rank)

    # Establishes GPU connection and processes
    dist.init_process_group("nccl")
    # model is on CPU before input to FSDP
    model = model.bfloat16()
    model = FSDP(
        model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        # backward_prefetch=BackwardPrefetch.BACKWARD_PRE
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.training.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.training.gamma)

    for epoch in range(1, args.training.epochs + 1):
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

    dist.barrier()
    cleanup()


# %%

if __name__ == "__main__":
    config = OmegaConf.load("configs/config.yaml")
    print(config.training.model_name)
    torch.manual_seed(config.training.seed)
    fsdp_main(config)

# %%
