"""
ResNet-50 → ConvNeXt transition model.
Same channel dims and depths as ResNet-50: (256, 512, 1024, 2048), (3, 4, 6, 3).

Toggles (each independently swaps one design choice):
  depthwise    : 3×3 dense conv → 3×3 depthwise conv (groups=channels)
  patchify_stem: 7×7+MaxPool stem → 4×4 stride-4 conv, no activation after (same output: B×64×56×56)
  layernorm    : BatchNorm2d → LayerNorm2d, same positions and count
  gelu         : ReLU → GELU
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.norm = nn.LayerNorm(C)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _norm(C, layernorm):
    return LayerNorm2d(C) if layernorm else nn.BatchNorm2d(C)


def _act(gelu):
    return nn.GELU() if gelu else nn.ReLU(inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        depthwise=False,
        layernorm=False,
        gelu=False,
    ):
        super().__init__()
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = _norm(planes, layernorm)
        self.act1 = _act(gelu)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            3,
            stride=stride,
            padding=1,
            groups=planes if depthwise else 1,
            bias=False,
        )
        self.bn2 = _norm(planes, layernorm)
        self.act2 = _act(gelu)

        self.conv3 = nn.Conv2d(planes, outplanes, 1, bias=False)
        self.bn3 = _norm(outplanes, layernorm)

        self.act3 = _act(gelu)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.act3(x + shortcut)


def _make_stage(inplanes, planes, depth, stride, depthwise, layernorm, gelu):
    outplanes = planes * Bottleneck.expansion
    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride=stride, bias=False),
            _norm(outplanes, layernorm),
        )
    blocks = [
        Bottleneck(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            depthwise=depthwise,
            layernorm=layernorm,
            gelu=gelu,
        )
    ]
    for _ in range(1, depth):
        blocks.append(
            Bottleneck(
                outplanes, planes, depthwise=depthwise, layernorm=layernorm, gelu=gelu
            )
        )
    return nn.Sequential(*blocks)


class ResNetTransition(nn.Module):
    def __init__(
        self,
        n_classes=1000,
        depthwise=False,
        patchify_stem=False,
        layernorm=False,
        gelu=False,
    ):
        super().__init__()

        if patchify_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=4, bias=False),
                _norm(64, layernorm),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                _norm(64, layernorm),
                _act(gelu),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

        self.layer1 = _make_stage(
            64,
            64,
            depth=3,
            stride=1,
            depthwise=depthwise,
            layernorm=layernorm,
            gelu=gelu,
        )
        self.layer2 = _make_stage(
            256,
            128,
            depth=4,
            stride=2,
            depthwise=depthwise,
            layernorm=layernorm,
            gelu=gelu,
        )
        self.layer3 = _make_stage(
            512,
            256,
            depth=6,
            stride=2,
            depthwise=depthwise,
            layernorm=layernorm,
            gelu=gelu,
        )
        self.layer4 = _make_stage(
            1024,
            512,
            depth=3,
            stride=2,
            depthwise=depthwise,
            layernorm=layernorm,
            gelu=gelu,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2048, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x).flatten(1))


def resnet_transition(
    n_classes=1000, depthwise=False, patchify_stem=False, layernorm=False, gelu=False
):
    return ResNetTransition(
        n_classes,
        depthwise=depthwise,
        patchify_stem=patchify_stem,
        layernorm=layernorm,
        gelu=gelu,
    )


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    configs = [
        dict(),
        dict(depthwise=True),
        dict(patchify_stem=True),
        dict(layernorm=True),
        dict(gelu=True),
        dict(depthwise=True, patchify_stem=True, layernorm=True, gelu=True),
    ]
    for flags in configs:
        model = resnet_transition(n_classes=19000, **flags)
        y = model(x)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{str(flags) or 'baseline'}: output={y.shape}, params={n_params:.1f}M")
