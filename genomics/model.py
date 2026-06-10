from language.model import GPT
from language.model_gdn import GDN


def gpt_small(seq_len=1024, n_layers=6, n_heads=6, d_embed=384, k=1):
    return GPT(
        vocab_size=4**k + 1,
        seq_len=seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        d_embed=d_embed,
        mlp_ratio=4.0,
    )


def gdn_small(seq_len=1024, n_layers=6, n_heads=6, d_embed=384, k=1):
    return GDN(
        vocab_size=4**k + 1,
        seq_len=seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        d_embed=d_embed,
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    import torch

    model = gpt_small(seq_len=256, n_layers=6)
    idx = torch.randint(0, 5, (2, 256))
    logits = model(idx)
    print(f"input: {idx.shape}, output: {logits.shape}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
