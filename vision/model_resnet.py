"""
ResNet-50 for ImageNet-1K via timm.
https://arxiv.org/abs/1512.03385
"""

import timm
import torch


def resnet18(n_classes=1000):
    m = timm.create_model("resnet18.a1_in1k", pretrained=False, num_classes=n_classes)
    return m


def resnet50(n_classes=1000):
    m = timm.create_model("resnet50.a1_in1k", pretrained=False, num_classes=n_classes)
    return m


if __name__ == "__main__":
    model = resnet50(n_classes=19_167)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
