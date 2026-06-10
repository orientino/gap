"""
ConvNeXt Small for ImageNet-1K via timm.
https://arxiv.org/abs/2201.03545
"""

import timm
import torch


def convnext_tiny(n_classes=1000):
    return timm.create_model("convnext_tiny", pretrained=False, num_classes=n_classes)


if __name__ == "__main__":
    model = convnext_tiny(n_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
