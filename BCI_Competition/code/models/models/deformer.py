import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        in_chan: int,
        fine_grained_kernel: int = 11,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout),
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def cnn_block(self, in_chan: int, kernel_size: int, dp: float):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(
                in_channels=in_chan,
                out_channels=in_chan,
                kernel_size=kernel_size,
                padding=self.get_padding_1d(kernel=kernel_size),
            ),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)
        x = x.view(x.size(0), -1)
        return torch.cat((x, x_dense), dim=-1)

    def get_info(self, x):
        return torch.log(torch.clamp(torch.mean(x.pow(2), dim=-1), min=1e-6))

    def get_padding_1d(self, kernel: int):
        return int(0.5 * (kernel - 1))


class DeformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        chans: int,
        samples: int,
        temporal_kernel: int = 11,
        num_kernel: int = 64,
        depth: int = 4,
        heads: int = 16,
        mlp_dim: int = 16,
        dim_head: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.cnn_encoder = self.cnn_block(
            out_chan=num_kernel,
            kernel_size=(1, temporal_kernel),
            num_chan=chans,
        )

        dim = int(0.5 * samples)
        self.to_patch_embedding = Rearrange("b k c f -> b k (c f)")
        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            in_chan=num_kernel,
            fine_grained_kernel=temporal_kernel,
        )

        hidden_sizes = self.get_hidden_size(input_size=dim, num_layer=depth)
        out_size = int(num_kernel * hidden_sizes[-1]) + int(num_kernel * depth)
        self.mlp_head = nn.Linear(out_size, num_classes)

    def cnn_block(self, out_chan: int, kernel_size: tuple[int, int], num_chan: int):
        return nn.Sequential(
            nn.Conv2d(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1])),
            nn.Conv2d(out_chan, out_chan, (num_chan, 1), padding=0),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
        )

    def _forward_features(self, eeg):
        if eeg.dim() == 3:
            eeg = eeg.unsqueeze(1)
        x = self.cnn_encoder(eeg)
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding
        return self.transformer(x)

    def forward(self, eeg, return_features: bool = False):
        features = self._forward_features(eeg)
        logits = self.mlp_head(features)
        if return_features:
            return logits, features
        return logits

    def get_padding(self, kernel: int):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size: int, num_layer: int):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]
