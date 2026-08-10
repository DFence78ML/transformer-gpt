from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
encoding = tokenizer.encode("Hello there!")

import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, seq_len):
        self.seq_len = seq_len

        # Tokenize once and store the IDs as one tensor.
        ids = tokenizer.encode(text).ids
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.seq_len]
        y = self.ids[idx + 1:idx + self.seq_len + 1]

        return x, y
