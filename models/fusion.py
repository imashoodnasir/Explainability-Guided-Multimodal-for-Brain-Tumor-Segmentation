from __future__ import annotations
import torch
import torch.nn as nn


class ModalityAwareFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.context(x)
        fused = x * w
        enhanced = fused + x
        return enhanced, w
