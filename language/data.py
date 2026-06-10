"""
Data loading for Shakespeare, TinyStories, and FineWeb-Edu.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def get_dataloaders(dataset, dir_data, seq_len=1024, batch_size=64, n_workers=4):
    dir_data = os.path.join(dir_data, dataset)
    return _load_bin_dataloaders(dir_data, seq_len, batch_size, n_workers)


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_samples = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def _load_bin_dataloaders(dir_data, seq_len, batch_size, n_workers):
    tr_tokens = np.memmap(
        os.path.join(dir_data, "train.bin"), dtype=np.uint16, mode="r"
    )
    vl_tokens = np.memmap(os.path.join(dir_data, "val.bin"), dtype=np.uint16, mode="r")
    tr_ds = TokenDataset(tr_tokens, seq_len)
    vl_ds = TokenDataset(vl_tokens, seq_len)
    tr_loader = DataLoader(tr_ds, batch_size, shuffle=True, num_workers=n_workers)
    vl_loader = DataLoader(vl_ds, batch_size, shuffle=False, num_workers=n_workers)
    steps_per_epoch = len(tr_ds) // batch_size
    return tr_loader, vl_loader, steps_per_epoch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    tr, vl, _ = get_dataloaders("shakespeare", dir_data=args.dir_data, seq_len=args.seq_len)  # fmt: skip
    print(f"Shakespeare train: {len(tr.dataset)}, val: {len(vl.dataset)}")
    tr, vl, _ = get_dataloaders("tinystories", dir_data=args.dir_data, seq_len=args.seq_len)  # fmt: skip
    print(f"TinyStories train: {len(tr.dataset)}, val: {len(vl.dataset)}")
    tr, vl, _ = get_dataloaders("fineweb", dir_data=args.dir_data, seq_len=args.seq_len)  # fmt: skip
    print(f"Fineweb train: {len(tr.dataset)}, val: {len(vl.dataset)}")
