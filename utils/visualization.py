from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    1: np.array([1.0, 0.2, 0.2]),  # NCR/NET red
    2: np.array([0.2, 0.55, 1.0]), # edema blue
    3: np.array([0.2, 0.9, 0.2]),  # ET green
}


def pick_slice(mask: np.ndarray) -> int:
    sums = [(z, (mask[..., z] > 0).sum()) for z in range(mask.shape[-1])]
    sums.sort(key=lambda x: x[1], reverse=True)
    return int(sums[0][0]) if sums else mask.shape[-1] // 2


def overlay_mask(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    gray = gray.astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    out = rgb.copy()
    for cls, color in COLORS.items():
        region = mask == cls
        out[region] = (1 - alpha) * out[region] + alpha * color
    return np.clip(out, 0, 1)


def save_qualitative_figure(flair: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: str, title: Optional[str] = None) -> None:
    z = pick_slice(gt)
    flair2d = flair[..., z]
    gt2d = gt[..., z]
    pred2d = pred[..., z]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(flair2d, cmap="gray")
    axes[0].set_title("FLAIR")
    axes[1].imshow(overlay_mask(flair2d, gt2d))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(overlay_mask(flair2d, pred2d))
    axes[2].set_title("Prediction")
    axes[3].imshow(flair2d, cmap="gray")
    axes[3].contour(gt2d == 2, colors="deepskyblue", linewidths=1)
    axes[3].contour(gt2d == 3, colors="lime", linewidths=1)
    axes[3].contour(pred2d == 2, colors="royalblue", linewidths=1, linestyles="dashed")
    axes[3].contour(pred2d == 3, colors="green", linewidths=1, linestyles="dashed")
    axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_error_figure(flair: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: str, title: Optional[str] = None) -> None:
    z = pick_slice(gt)
    flair2d = flair[..., z]
    gt2d = gt[..., z]
    pred2d = pred[..., z]
    fp = np.logical_and(pred2d > 0, gt2d == 0)
    fn = np.logical_and(pred2d == 0, gt2d > 0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(overlay_mask(flair2d, gt2d))
    axes[0].set_title("FLAIR + GT")
    axes[1].imshow(overlay_mask(flair2d, pred2d))
    axes[1].set_title("FLAIR + Prediction")
    axes[2].imshow(flair2d, cmap="gray")
    axes[2].contour(gt2d > 0, colors="white", linewidths=1)
    axes[2].contour(pred2d > 0, colors="cyan", linewidths=1, linestyles="dashed")
    axes[2].set_title("Overlay + GT contours")
    axes[3].imshow(flair2d, cmap="gray")
    axes[3].imshow(np.where(fp, 1, np.nan), cmap="Reds", alpha=0.7)
    axes[3].imshow(np.where(fn, 1, np.nan), cmap="Blues", alpha=0.7)
    axes[3].set_title("Diff map (FP/FN)")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_uncertainty_figure(flair: np.ndarray, gt: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray, out_path: str) -> None:
    z = pick_slice(gt)
    flair2d = flair[..., z]
    gt2d = gt[..., z]
    pred2d = pred[..., z]
    unc2d = uncertainty[..., z]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(overlay_mask(flair2d, gt2d))
    axes[0].set_title("FLAIR + GT")
    axes[1].imshow(overlay_mask(flair2d, pred2d))
    axes[1].set_title("FLAIR + Prediction")
    im = axes[2].imshow(unc2d, cmap="inferno")
    axes[2].set_title("Uncertainty map")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(flair2d, cmap="gray")
    axes[3].imshow(unc2d, cmap="inferno", alpha=0.45)
    axes[3].set_title("Prediction + uncertainty")
    for ax in axes:
        ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
