from typing import Optional

import torch
from torch import nn


class DeepCNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        chans: int,
        samples: int,
        dropoutRate: Optional[float] = 0.5,
        d1: Optional[int] = 25,
        d2: Optional[int] = 50,
        d3: Optional[int] = 100,
    ):
        super().__init__()

        self.Chans = chans
        self.Samples = samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d1, kernel_size=(1, 10)),
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=(chans, 1)),
            nn.BatchNorm2d(num_features=d1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(self.dropoutRate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=(1, 10)),
            nn.BatchNorm2d(num_features=d2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(self.dropoutRate),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=d2, out_channels=d3, kernel_size=(1, 10)),
            nn.BatchNorm2d(num_features=d3),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 3)),
            nn.Dropout(self.dropoutRate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat_dim = self._forward_features(dummy).shape[1]
        self.fc = nn.Linear(feat_dim, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        return output.reshape(output.size(0), -1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = self._forward_features(x)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits
