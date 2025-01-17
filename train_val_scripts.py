#%%

import os
import torch
import torch.distributed as dist
from tqdm import tqdm


def train_(model, rank, train_loader, optimizer, epoch, sampler=None, task="causal"):
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_loss = torch.zeros(2).to(local_rank)

    # TODO Why to do it for separate sampler but not do train_loader.sampler ?
    if sampler:
        sampler.set_epoch(epoch)
    if rank == 0:
        pbar = tqdm(
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

def validation(model, rank, val_loader, task="causal"):
    model.eval()
    correct = 0
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_loss = torch.zeros(2).to(local_rank)
    if rank == 0:
        pbar = tqdm(
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


def train(model, rank, train_loader, optimizer, epoch, sampler=None, task="instruct"):
    model.train()
    local_rank = int(os.environ["LOCAL_RANK"])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)

    if rank == 0:
        pbar = tqdm(train_loader, colour="blue", desc="r0 Training Epoch")

    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)

        print(batch["tokens"].shape)
        optimizer.zero_grad()
        output = model(
            input_ids=batch["tokens"],
            attention_mask=batch["mask"],
            labels=batch["labels"],
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