"""
Data loading for hg38 character-level and k-mer tokenized genomics.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def get_dataloaders(dataset, k, stride, **kwargs):
    if dataset == "hg38_char":
        return get_hg38_char_dataloaders(**kwargs)
    elif dataset.startswith("hg38_kmer"):
        return get_hg38_kmer_dataloaders(k=k, stride=stride, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


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


def get_hg38_char_dataloaders(dir_data, seq_len=1024, batch_size=64, n_workers=4):
    return _load_bin_dataloaders(
        os.path.join(dir_data, "hg38_char"),
        seq_len,
        batch_size,
        n_workers,
        dtype=np.uint8,
    )


def get_hg38_kmer_dataloaders(dir_data, k, stride, seq_len=1024, batch_size=64, n_workers=4):  # fmt: skip
    return _load_bin_dataloaders(
        os.path.join(dir_data, f"hg38_kmer{k}_s{stride}"),
        seq_len,
        batch_size,
        n_workers,
        dtype=np.uint16,
    )


def _load_bin_dataloaders(dir_data, seq_len, batch_size, n_workers, dtype=np.uint8):
    tr_tokens = np.memmap(os.path.join(dir_data, "train.bin"), dtype=dtype, mode="r")
    vl_tokens = np.memmap(os.path.join(dir_data, "val.bin"), dtype=dtype, mode="r")
    tr_ds = TokenDataset(tr_tokens, seq_len)
    vl_ds = TokenDataset(vl_tokens, seq_len)
    tr_loader = DataLoader(tr_ds, batch_size, shuffle=False, num_workers=n_workers)
    vl_loader = DataLoader(vl_ds, batch_size, shuffle=False, num_workers=n_workers)
    steps_per_epoch = len(tr_ds) // batch_size
    return tr_loader, vl_loader, steps_per_epoch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    tr, vl, _ = get_dataloaders("hg38_char", dir_data=args.dir_data, seq_len=args.seq_len)  # fmt: skip
    print(f"hg38_char train: {len(tr.dataset)}, val: {len(vl.dataset)}")
    tr, vl, _ = get_dataloaders("hg38_kmer6_s6", dir_data=args.dir_data, seq_len=args.seq_len)  # fmt: skip
    print(f"hg38_kmer train: {len(tr.dataset)}, val: {len(vl.dataset)}")
