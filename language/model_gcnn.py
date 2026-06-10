"""
GPT with Gated Convolutional blocks (GLU) replacing Attention.
Dauphin et al. 2017 "Language Modeling with Gated Convolutional Networks".
Uses sinusoidal positional embeddings.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def posemb_sincos_1d(seq_len, width, temperature=10_000.0):
    pos = np.arange(seq_len)
    assert width % 2 == 0
    omega = np.arange(width // 2) / (width // 2 - 1)
    omega = 1.0 / (temperature**omega)
    pos = np.einsum("m,d->md", pos.astype(np.float32), omega)
    return np.concatenate([np.sin(pos), np.cos(pos)], axis=1)


class GatedConv(nn.Module):
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim * 2, kernel_size, padding=kernel_size - 1, bias=False
        )
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)  # (B, C, T) as conv expects
        x = self.conv(x)[..., :T]  # causal: left-pad only, (B, 2C, T)
        x = x.transpose(1, 2)  # (B, T, 2C)
        a, b = x.chunk(2, dim=-1)
        return self.proj(a * b.sigmoid())


class GatedConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.conv = GatedConv(dim, kernel_size=kernel_size)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()
        x = self.fc2(x)
        return x


class GCNN(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        seq_len=1024,
        n_layers=6,
        d_embed=384,
        kernel_size=4,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_embed)
        pos_embed = posemb_sincos_1d(seq_len, d_embed)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        self.norm_embed = nn.RMSNorm(d_embed, eps=1e-6)
        self.blocks = nn.ModuleList(
            [GatedConvBlock(d_embed, kernel_size, mlp_ratio) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(d_embed, eps=1e-6)
        self.head = nn.Linear(d_embed, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            nn.init.xavier_uniform_(block.conv.conv.weight)
            nn.init.xavier_uniform_(block.conv.proj.weight)
            block.conv.proj.weight.data *= scale
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            block.mlp.fc2.weight.data *= scale
        nn.init.zeros_(self.head.weight)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_embed(idx) + self.pos_embed[:, :T]
        x = self.norm_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        return x


def gcnn_small(seq_len=1024, n_layers=6, d_embed=384, kernel_size=4):
    return GCNN(
        vocab_size=50257,
        seq_len=seq_len,
        n_layers=n_layers,
        d_embed=d_embed,
        kernel_size=kernel_size,
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    model = gcnn_small(seq_len=256, n_layers=6)
    idx = torch.randint(0, 50257, (2, 256))
    logits = model(idx)
    print(f"Input shape: {idx.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
