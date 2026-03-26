from __future__ import annotations
import torch
import torch.nn as nn
from models.encoder import Encoder3D
from models.fusion import ModalityAwareFusion
from models.decoder import Decoder3D


class ExplainableTumorNet(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 16, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = Encoder3D(in_channels, base, dropout)
        self.fusion = ModalityAwareFusion(base * 16)
        self.decoder = Decoder3D(base, num_classes, dropout)

    def forward(self, x: torch.Tensor):
        f1, f2, f3, f4, fb = self.encoder(x)
        fused, weights = self.fusion(fb)
        logits = self.decoder(f1, f2, f3, f4, fused)
        return {
            "logits": logits,
            "features": fused,
            "fusion_weights": weights,
            "skips": (f1, f2, f3, f4),
        }
