from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from config import Config
from models.network import ExplainableTumorNet
from utils.common import get_device, load_json, save_json
from utils.dataset import make_dataset
from utils.metrics import logits_to_pred, compute_case_metrics
from monai.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--split_json", type=str, default="data_splits/test_subjects.json")
    parser.add_argument("--out_json", type=str, default="outputs/test_metrics.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.device is not None:
        cfg.device = args.device
    device = get_device(cfg.device)

    subjects = load_json(args.split_json)
    ds = make_dataset(subjects, train=False, patch_size=cfg.patch_size, spacing=cfg.spacing)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=max(1, cfg.num_workers // 2))

    model = ExplainableTumorNet(cfg.in_channels, cfg.base_channels, cfg.num_classes, cfg.dropout).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_metrics: Dict[str, list] = {}
    per_case = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            x = batch["image"].to(device)
            y = batch["seg"].squeeze(1).to(device) if batch["seg"].ndim == 5 else batch["seg"].to(device)
            sid = batch.get("id", ["unknown"])[0]
            out = model(x)
            pred = logits_to_pred(out["logits"]).cpu().numpy()[0]
            target = y.cpu().numpy()[0]
            m = compute_case_metrics(pred, target)
            per_case[sid] = m
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)

    summary = {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}
    save_json({"summary": summary, "per_case": per_case}, args.out_json)
    print(summary)


if __name__ == "__main__":
    main()
