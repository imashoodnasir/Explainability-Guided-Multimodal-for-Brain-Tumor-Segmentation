from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder3D(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 16, dropout: float = 0.2):
        super().__init__()
        self.e1 = ConvBlock3D(in_channels, base, dropout)
        self.p1 = nn.MaxPool3d(2)
        self.e2 = ConvBlock3D(base, base * 2, dropout)
        self.p2 = nn.MaxPool3d(2)
        self.e3 = ConvBlock3D(base * 2, base * 4, dropout)
        self.p3 = nn.MaxPool3d(2)
        self.e4 = ConvBlock3D(base * 4, base * 8, dropout)
        self.p4 = nn.MaxPool3d(2)
        self.b = ConvBlock3D(base * 8, base * 16, dropout)

    def forward(self, x: torch.Tensor):
        f1 = self.e1(x)
        f2 = self.e2(self.p1(f1))
        f3 = self.e3(self.p2(f2))
        f4 = self.e4(self.p3(f3))
        fb = self.b(self.p4(f4))
        return f1, f2, f3, f4, fb
