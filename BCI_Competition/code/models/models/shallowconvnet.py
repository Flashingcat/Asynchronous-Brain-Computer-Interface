import torch
from torch import nn


class ShallowConvNetClassifier(nn.Module):
    def __init__(self, num_classes: int, chans: int, samples: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 65), stride=1, padding="same"),
            nn.BatchNorm2d(32, momentum=0.1, affine=True, eps=1e-5),
            nn.Conv2d(32, 64, kernel_size=(chans, 1), padding="valid", groups=32),
            nn.BatchNorm2d(64, momentum=0.1, affine=True, eps=1e-5),
        )
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, chans, samples)
            feat_dim = self._forward_features(dummy).shape[1]
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(256, num_classes)

    def _forward_features(self, x):
        x = self.layer1(x.unsqueeze(1))
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 35), (1, 7))
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x.flatten(1)

    def forward(self, x, return_features: bool = False):
        features = self.projector(self._forward_features(x))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
