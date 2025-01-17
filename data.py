import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torchtune.datasets import alpaca_dataset
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.models.llama2 import Llama2Tokenizer

from summarization_dataset import wikihow

#%%

# tokenizer = Llama3Tokenizer("/mnt/data/galimzyanov/temp/torchtune/llama2-13B/model/tokenizer.model")
# dataset = alpaca_dataset(tokenizer)
# # dataset_iter = iter(dataset)
# item = next(iter(dataset))
# #%%
# qq = [len(item["tokens"]) for item in dataset]
#%%

def collate_fn(batch):

    max_length = max(len(sample['tokens']) for sample in batch)
    max_length = min(max_length, 128)

    tokens = []
    masks = []
    labels = []

    for sample in batch:
        sample_tokens = sample['tokens']
        sample_mask = sample['mask']
        sample_labels = sample['labels']

        sample_tokens = sample_tokens[:max_length]
        sample_mask = sample_mask[:max_length]
        sample_labels = sample_labels[:max_length]

        padding = [0] * (max_length - len(sample_tokens))
        sample_tokens = sample_tokens + padding
        sample_mask = sample_mask + [True] * len(padding)
        sample_labels = sample_labels + padding

        tokens.append(torch.tensor(sample_tokens))
        masks.append(torch.tensor(sample_mask))
        labels.append(torch.tensor(sample_labels))

    tokens = torch.stack(tokens)
    masks = torch.stack(masks)
    labels = torch.stack(labels)

    return {
        'tokens': tokens,
        'mask': masks,
        'labels': labels
    }

#%%

def get_data_(rank, world_size, args, tokenizer):
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



def get_data(rank, world_size, args, tokenizer):

    # tokenizer_torchtune = Llama2Tokenizer("/mnt/data/galimzyanov/temp/torchtune/llama2-13B/model/tokenizer.model")
    tokenizer_torchtune = Llama3Tokenizer("/mnt/data/galimzyanov/temp/torchtune/llama3_2_3B_full/model/original/tokenizer.model")
    dataset = alpaca_dataset(tokenizer_torchtune)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.training.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, None, train_sampler


#%%

# tokenizer_torchtune = Llama2Tokenizer("/mnt/data/galimzyanov/temp/torchtune/llama2-13B/model/tokenizer.model")
# tokenizer_torchtune = Llama3Tokenizer("/mnt/data/galimzyanov/temp/torchtune/llama3_2_3B_full/model/original/tokenizer.model")
# dataset = alpaca_dataset(tokenizer_torchtune)
#
# train_sampler = DistributedSampler(dataset, num_replicas=8, rank=1, shuffle=True)
#
# train_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=7,
#     sampler=train_sampler,
#     num_workers=1,
#     pin_memory=True,
#     collate_fn=collate_fn
# )

#%%