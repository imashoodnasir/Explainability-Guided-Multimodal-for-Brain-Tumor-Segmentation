from __future__ import annotations
import torch
import torch.nn as nn
from models.encoder import ConvBlock3D


class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder3D(nn.Module):
    def __init__(self, base: int = 16, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.d4 = UpBlock3D(base * 16, base * 8, base * 8, dropout)
        self.d3 = UpBlock3D(base * 8, base * 4, base * 4, dropout)
        self.d2 = UpBlock3D(base * 4, base * 2, base * 2, dropout)
        self.d1 = UpBlock3D(base * 2, base, base, dropout)
        self.out = nn.Conv3d(base, num_classes, kernel_size=1)

    def forward(self, f1, f2, f3, f4, fb):
        x = self.d4(fb, f4)
        x = self.d3(x, f3)
        x = self.d2(x, f2)
        x = self.d1(x, f1)
        return self.out(x)
