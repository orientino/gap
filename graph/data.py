"""
Data loading for graph datasets (ZINC 12k, ...).
"""

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader


def get_dataloaders(dataset, **kwargs):
    if dataset == "zinc12k":
        return get_zinc_dataloaders(subset=True, **kwargs)
    elif dataset == "zinc250k":
        return get_zinc_dataloaders(subset=False, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_zinc_dataloaders(dir_data, batch_size=128, n_workers=4, subset=True, pre_transform=None):
    root = dir_data + ("/zinc_rrwp" if pre_transform is not None else "/zinc")
    tr_ds = ZINC(root=root, subset=subset, split="train", pre_transform=pre_transform)
    vl_ds = ZINC(root=root, subset=subset, split="val", pre_transform=pre_transform)
    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    vl_loader = DataLoader(
        vl_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )
    steps_per_epoch = len(tr_ds) // batch_size
    return tr_loader, vl_loader, steps_per_epoch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data", type=str, required=True)
    args = parser.parse_args()

    for ds in ("zinc12k", "zinc250k"):
        tr, vl, _ = get_dataloaders(ds, dir_data=args.dir_data)
        print(f"{ds} train: {len(tr.dataset)}, val: {len(vl.dataset)}")
    batch = next(iter(tr))
    print(
        f"x: {batch.x.shape}, edge_index: {batch.edge_index.shape}, y: {batch.y.shape}"
    )
