"""
GatedDeltaNet variant of GPT for genomics (HG38).
Replaces the attention mixer with GatedDeltaNet from flash-linear-attention.
Requires: pip install flash-linear-attention
"""

import math

import torch
import torch.nn as nn
from fla.layers.gated_deltanet import GatedDeltaNet

from .model import MLP


class GDNBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, layer_idx=0):
        super().__init__()
        # GatedDeltaNet requires num_heads * head_dim = 0.75 * hidden_size
        assert (dim * 3 // 4) % n_heads == 0, f"0.75*{dim} must be divisible by n_heads"
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = GatedDeltaNet(
            hidden_size=dim,
            num_heads=n_heads,
            head_dim=(dim * 3 // 4) // n_heads,
            use_gate=True,
            norm_eps=1e-6,
            layer_idx=layer_idx,
        )
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        o, _, _ = self.attn(self.norm1(x))
        x = x + o
        x = x + self.mlp(self.norm2(x))
        return x


class GDN(nn.Module):
    def __init__(
        self,
        vocab_size=5,
        seq_len=1024,
        n_layers=6,
        n_heads=6,
        d_embed=384,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.d_embed = d_embed

        self.tok_embed = nn.Embedding(vocab_size, d_embed)
        self.norm_embed = nn.RMSNorm(d_embed, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                GDNBlock(d_embed, n_heads, mlp_ratio, layer_idx=i)
                for i in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_embed, eps=1e-6)
        self.head = nn.Linear(d_embed, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            gdn = block.attn
            for m in gdn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            gdn.o_proj.weight.data *= scale
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            block.mlp.fc2.weight.data *= scale
        nn.init.zeros_(self.head.weight)

    def forward(self, idx):
        x = self.tok_embed(idx)
        x = self.norm_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


def gdn_small(vocab_size=50257, seq_len=1024, n_layers=6, n_heads=6, d_embed=384):
    return GDN(
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        d_embed=d_embed,
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    model = gdn_small(seq_len=256, n_layers=6).cuda()
    idx = torch.randint(0, 5, (2, 256)).cuda()
    logits = model(idx)
    print(f"Input shape:  {idx.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
