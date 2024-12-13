import re
import tiktoken
import importlib
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from SimpleTokenizer import SimpleTokenizer

# Extracting Raw Text
with open("ReeLRBWittgenstein.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        print(len(token_ids))

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

max_length = 4
dataloader = CreateDataloader(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\\n", inputs.shape)

torch.Size([8, 4, 256])



# # Preprocessing for Tokenization
# preprocessed = re.split(r'([,.:;?_!"()\'-]|\s)', raw_text)
# preprocessed = [
#  item.strip() for item in preprocessed if item.strip()
#  ]
# print(preprocessed)
#
# # Vocabulary Size
# all_words = sorted(list(set(preprocessed)))
# all_words.extend(["<|endoftext|>", "<|unk|>"])
# vocab_size = len(all_words)
# print(vocab_size)
#
# vocab = {token:integer for integer,token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break
