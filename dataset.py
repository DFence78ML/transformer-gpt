import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


TOKENIZER_FILE = "tokenizer.json"
tokenizer = Tokenizer.from_file(
            TOKENIZER_FILE
            )

class GPTDataset(Dataset):
    def __init__(self,filename,seq_len):
        self.data = np.memmap(
            filename,
            dtype=np.uint16,
            mode="r"
        )
        self.seq_len = seq_len

    def __len__(self):
        return len(
            self.data
        ) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[
            idx:idx + self.seq_len + 1
        ]
        chunk = torch.from_numpy(
            chunk.astype(np.int64)
        )
        x = chunk[:-1]
        y = chunk[1:]

        return x, y