import numpy as np
import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(
        self,
        filename,
        seq_len
    ):
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