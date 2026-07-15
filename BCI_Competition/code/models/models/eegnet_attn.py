"""EEGNet with Squeeze-and-Excitation channel attention.

Adds lightweight channel-wise attention between the two convolutional
blocks, letting the model adaptively re-weight spatial features before
temporal convolution.
"""

import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class EEGNetAttnClassifier(nn.Module):
    """EEGNet with channel attention, wider filters, and residual connection."""

    def __init__(
        self,
        num_classes: int,
        chans: int,
        samples: int,
        kernLenght: int = 64,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,
        dropoutRate: float = 0.5,
        norm_rate: float = 0.25,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kernLenght // 2 - 1, kernLenght - kernLenght // 2, 0, 0)),
            nn.Conv2d(1, F1, kernel_size=(1, kernLenght), stride=1, bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate),
        )

        self.atten1 = SEBlock(F1 * D)

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate),
        )

        self.atten2 = SEBlock(F2)

        self.classifier = nn.Linear(F2 * (samples // 32), num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.atten1(x)
        x = self.block2(x)
        x = self.atten2(x)

        features = x.reshape(x.size(0), -1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
