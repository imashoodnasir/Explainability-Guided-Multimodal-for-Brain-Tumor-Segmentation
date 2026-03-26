from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from medpy.metric.binary import hd95


def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num / (den + eps))


def dice_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return _safe_div(2 * inter, pred.sum() + target.sum())


def iou_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return _safe_div(inter, union)


def hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")
    return float(hd95(pred, target))


def brats_regions(mask: np.ndarray) -> Dict[str, np.ndarray]:
    wt = mask > 0
    tc = np.logical_or(mask == 1, mask == 3)
    et = mask == 3
    return {"WT": wt, "TC": tc, "ET": et}


def compute_case_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred_r = brats_regions(pred)
    target_r = brats_regions(target)
    out: Dict[str, float] = {}
    for region in ["WT", "TC", "ET"]:
        out[f"Dice_{region}"] = dice_binary(pred_r[region], target_r[region])
        out[f"IoU_{region}"] = iou_binary(pred_r[region], target_r[region])
        out[f"HD95_{region}"] = hd95_binary(pred_r[region], target_r[region])
    return out


def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(torch.softmax(logits, dim=1), dim=1)
