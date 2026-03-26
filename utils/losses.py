from __future__ import annotations
import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * onehot, dims)
    denom = torch.sum(probs, dims) + torch.sum(onehot, dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets.long())


def combined_segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    return alpha * dice_loss(logits, targets) + beta * cross_entropy_loss(logits, targets)
