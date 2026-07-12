import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(self, channel: int, emb_size: int = 40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        return self.projection(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        res = x
        x = self.fn(x, **kwargs)
        return x + res


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        depth: int,
        emb_size: int,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size,
                    num_heads,
                    drop_p,
                    forward_expansion,
                    forward_drop_p,
                )
                for _ in range(depth)
            ]
        )


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
        )
        self.classifier = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor, return_features: bool = False):
        features = self.clshead(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class ConformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        chans: int,
        samples: int,
        emb_size: int = 40,
        depth: int = 6,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
    ):
        super().__init__()
        self.samples = samples
        self.patch_embedding = PatchEmbedding(chans, emb_size)
        self.encoder = TransformerEncoder(
            depth,
            emb_size,
            num_heads,
            drop_p,
            forward_expansion,
            forward_drop_p,
        )
        self.head = ClassificationHead(emb_size, num_classes)

    def forward(self, x: Tensor, return_features: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return self.head(x, return_features=return_features)
