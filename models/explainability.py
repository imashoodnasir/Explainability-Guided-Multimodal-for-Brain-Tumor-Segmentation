from __future__ import annotations
import torch
import torch.nn.functional as F


def gradient_attribution(features: torch.Tensor, logits: torch.Tensor, target_mask: torch.Tensor):
    # use foreground score to obtain attribution
    score = torch.softmax(logits, dim=1)[:, 1:, ...].sum()
    grads = torch.autograd.grad(score, features, retain_graph=True, create_graph=True)[0]
    attn = torch.relu((grads * features).sum(dim=1, keepdim=True))
    attn = attn / (attn.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)
    target = (target_mask > 0).float().unsqueeze(1)
    return attn, target


def attribution_alignment_loss(features: torch.Tensor, logits: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attn, target = gradient_attribution(features, logits, target_mask)
    loss = F.mse_loss(attn, target)
    return loss, attn


def consistency_regularization(attn1: torch.Tensor, attn2: torch.Tensor) -> torch.Tensor:
    attn1 = attn1 / (attn1.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)
    attn2 = attn2 / (attn2.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)
    return F.l1_loss(attn1, attn2)
