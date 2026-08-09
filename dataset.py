from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
encoding = tokenizer.encode("Hello there!")

import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, seq_len):
        self.seq_len = seq_len
        ids = tokenizer.encode(text).ids
        self.examples = []

        for i in range(len(ids) - seq_len):
            x = ids[i:i+seq_len]
            y = ids[i+1:i+seq_len+1]
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        
        return (
            torch.tensor(x),
            torch.tensor(y)
        )