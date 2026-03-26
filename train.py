from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config
from models.network import ExplainableTumorNet
from models.explainability import attribution_alignment_loss, consistency_regularization
from utils.losses import combined_segmentation_loss
from utils.dataset import make_loaders
from utils.metrics import logits_to_pred, compute_case_metrics
from utils.common import seed_everything, ensure_dir, get_device, save_json


@torch.no_grad()
def validate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_metrics: Dict[str, list] = {}
    for batch in tqdm(loader, desc="Validate", leave=False):
        x = batch["image"].to(device)
        y = batch["seg"].squeeze(1).to(device) if batch["seg"].ndim == 5 else batch["seg"].to(device)
        out = model(x)
        pred = logits_to_pred(out["logits"]).cpu().numpy()
        target = y.cpu().numpy()
        for b in range(pred.shape[0]):
            m = compute_case_metrics(pred[b], target[b])
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)
    return {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    seed_everything(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)
    cfg.save(out_dir / "config.json")

    device = get_device(cfg.device)
    train_loader, val_loader = make_loaders(
        cfg.train_split_json, cfg.val_split_json, cfg.batch_size, cfg.val_batch_size,
        cfg.num_workers, cfg.patch_size, cfg.spacing
    )

    model = ExplainableTumorNet(cfg.in_channels, cfg.base_channels, cfg.num_classes, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")

    best_score = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in loop:
            x = batch["image"].to(device)
            y = batch["seg"].squeeze(1).to(device) if batch["seg"].ndim == 5 else batch["seg"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp and device.type == "cuda"):
                out = model(x)
                logits = out["logits"]
                features = out["features"]
                seg_loss = combined_segmentation_loss(logits, y)
                expl_loss, attn = attribution_alignment_loss(features, logits, y)
                noisy_x = x + 0.01 * torch.randn_like(x)
                out2 = model(noisy_x)
                _, attn2 = attribution_alignment_loss(out2["features"], out2["logits"], y)
                cons_loss = consistency_regularization(attn, attn2)
                loss = seg_loss + cfg.explain_lambda * expl_loss + cfg.consistency_lambda * cons_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_metrics = validate(model, val_loader, device)
        score = (val_metrics["Dice_WT"] + val_metrics["Dice_TC"] + val_metrics["Dice_ET"]) / 3.0
        record = {"epoch": epoch, "train_loss": running_loss / max(1, len(train_loader)), **val_metrics}
        history.append(record)
        save_json(history, out_dir / "history.json")

        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, out_dir / "best_model.pt")
        if epoch % cfg.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, out_dir / f"checkpoint_epoch_{epoch}.pt")

        print({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()})

    print(f"Best mean Dice: {best_score:.4f}")


if __name__ == "__main__":
    main()
