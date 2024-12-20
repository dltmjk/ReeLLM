import re
import tiktoken
import importlib
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from SimpleTokenizer import SimpleTokenizer
import torch.nn as nn

# Extracting Raw Text
with open("ReeLRBWittgenstein.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def CreateDataloader(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
    def forward(self, x):
        keys = x @ self.W.key
        queries = x @ self.W.query
        values = x @ self.W.value

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax (attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vector = attention_weights @ values
        return context_vector


max_length = 4
dataloader = CreateDataloader(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#Create Embeddings
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings






