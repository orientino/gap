"""
Vision Transformer Small (ViT-S/16) for ImageNet-1K.
https://arxiv.org/abs/2205.01580
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from einops.layers.torch import Rearrange


def init_t_xy(end_x, end_y):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim, end_x, end_y, theta=100):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    """from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_embed=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_chans, d_embed, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm_q = nn.RMSNorm(self.d_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.d_head, eps=1e-6)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.norm_q(q), self.norm_k(k)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        n_layers=12,
        n_heads=6,
        d_embed=384,
        n_classes=1000,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_embed = d_embed

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            d_embed=d_embed,
        )

        # patch_height = patch_width = patch_size
        # patch_dim = in_chans * patch_height * patch_width
        # self.patch_embed = nn.Sequential(
        #     Rearrange(
        #         "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        #         p1=patch_height,
        #         p2=patch_width,
        #     ),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, d_embed),
        #     nn.LayerNorm(d_embed),
        # )
        # grid_size = img_size // patch_size

        grid_size = self.patch_embed.grid_size
        freqs_cis = compute_axial_cis(d_embed // n_heads, grid_size, grid_size)
        self.register_buffer("freqs_cis", freqs_cis)
        self.norm_embed = nn.RMSNorm(d_embed, eps=1e-6)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_embed, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(d_embed, eps=1e-6)
        self.head = nn.Linear(d_embed, n_classes, bias=False)
        self._init_weights()

    def _init_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            block.attn.proj.weight.data *= scale
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            block.mlp.fc2.weight.data *= scale
        nn.init.zeros_(self.head.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.norm_embed(x)
        for block in self.blocks:
            x = block(x, self.freqs_cis)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def vit_small_patch16(n_layers=6, n_heads=6, d_embed=384, n_classes=1000):
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        n_layers=n_layers,
        n_heads=n_heads,
        d_embed=d_embed,
        n_classes=n_classes,
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    model = vit_small_patch16(n_layers=6, n_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"freqs_cis shape: {model.freqs_cis.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
