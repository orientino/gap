"""
GCN and GAT models for ZINC 12k graph regression.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

# ZINC: 28 atom types (node features), 4 bond types (edge features)
N_ATOM_TYPES = 28


class GCN(nn.Module):
    def __init__(self, n_layers=4, d_embed=256):
        super().__init__()
        self.atom_embed = nn.Embedding(N_ATOM_TYPES, d_embed)
        self.convs = nn.ModuleList([GCNConv(d_embed, d_embed) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_embed) for _ in range(n_layers)])
        self.head = nn.Linear(d_embed, 1)

    def forward(self, data):
        x = self.atom_embed(data.x.squeeze(-1))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index)))
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, n_layers=4, d_embed=256, n_heads=8):
        super().__init__()
        self.atom_embed = nn.Embedding(N_ATOM_TYPES, d_embed)
        self.convs = nn.ModuleList(
            [
                GATConv(d_embed, d_embed // n_heads, heads=n_heads, concat=True)
                for _ in range(n_layers)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_embed) for _ in range(n_layers)])
        self.head = nn.Linear(d_embed, 1)

    def forward(self, data):
        x = self.atom_embed(data.x.squeeze(-1))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index)))
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze(-1)


if __name__ == "__main__":
    import torch
    from torch_geometric.data import Batch

    # Minimal synthetic graph batch
    x = torch.randint(0, 28, (10, 1))
    edge_index = torch.randint(0, 10, (2, 20))
    y = torch.randn(2)
    batch = torch.tensor([0] * 5 + [1] * 5)
    data = Batch(x=x, edge_index=edge_index, y=y, batch=batch)

    for model in [GCN, GAT]:
        m = model(n_layers=4, d_embed=256)
        out = m(data)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{model.__name__}: output {out.shape}, params {params:.2f}M")
