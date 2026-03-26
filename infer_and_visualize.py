from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from monai.data import DataLoader

from config import Config
from models.network import ExplainableTumorNet
from utils.common import get_device, load_json, ensure_dir
from utils.dataset import make_dataset
from utils.metrics import logits_to_pred
from utils.visualization import save_qualitative_figure, save_error_figure, save_uncertainty_figure


def enable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


def mc_uncertainty(model: torch.nn.Module, x: torch.Tensor, passes: int) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    probs = []
    model.eval()
    enable_dropout(model)
    with torch.no_grad():
        for _ in range(passes):
            out = model(x)
            p = torch.softmax(out["logits"], dim=1)
            probs.append(p.cpu().numpy())
            preds.append(torch.argmax(p, dim=1).cpu().numpy())
    probs_np = np.stack(probs, axis=0)  # T,B,C,H,W,D
    uncertainty = probs_np.var(axis=0).mean(axis=1)[0]
    pred_mode = np.stack(preds, axis=0)
    pred = np.apply_along_axis(lambda a: np.bincount(a).argmax(), 0, pred_mode[:, 0])
    return pred, uncertainty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--split_json", type=str, default="data_splits/test_subjects.json")
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    parser.add_argument("--num_cases", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device is not None:
        cfg.device = args.device
    device = get_device(cfg.device)
    out_dir = ensure_dir(args.out_dir)

    subjects = load_json(args.split_json)[: args.num_cases]
    ds = make_dataset(subjects, train=False, patch_size=cfg.patch_size, spacing=cfg.spacing)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    model = ExplainableTumorNet(cfg.in_channels, cfg.base_channels, cfg.num_classes, cfg.dropout).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    for idx, batch in enumerate(loader):
        x = batch["image"].to(device)
        y = batch["seg"].squeeze(1).to(device) if batch["seg"].ndim == 5 else batch["seg"].to(device)
        sid = batch.get("id", [f"case_{idx}"])[0]
        with torch.no_grad():
            out = model(x)
            pred = logits_to_pred(out["logits"]).cpu().numpy()[0]
        image = batch["image"][0, 3].cpu().numpy()  # FLAIR
        gt = y.cpu().numpy()[0]
        save_qualitative_figure(image, gt, pred, str(Path(out_dir) / f"{sid}_qual.png"), sid)
        save_error_figure(image, gt, pred, str(Path(out_dir) / f"{sid}_error.png"), sid)
        pred_mc, unc = mc_uncertainty(model, x, cfg.mc_dropout_passes)
        save_uncertainty_figure(image, gt, pred_mc, unc, str(Path(out_dir) / f"{sid}_uncertainty.png"))
        print(f"Saved figures for {sid}")


if __name__ == "__main__":
    main()
