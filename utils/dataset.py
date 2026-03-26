from __future__ import annotations
from typing import Dict, List, Sequence

import torch
from monai.data import Dataset, DataLoader
from monai.transforms import MapTransform

from utils.transforms import train_transforms, val_transforms
from utils.common import load_json


class RemapBraTSLabels(MapTransform):
    """Remap BraTS labels from {0,1,2,4} to {0,1,2,3}."""
    def __call__(self, data):
        d = dict(data)
        seg = d["seg"]
        seg = torch.where(seg == 4, torch.tensor(3, device=seg.device), seg)
        d["seg"] = seg.long()
        return d


class StackModalities(MapTransform):
    def __call__(self, data):
        d = dict(data)
        x = torch.cat([d["t1"], d["t1ce"], d["t2"], d["flair"]], dim=0)
        d["image"] = x.float()
        return d


class DropUnusedKeys(MapTransform):
    def __call__(self, data):
        return {"image": data["image"], "seg": data["seg"], "id": data.get("id", "unknown")}


def _with_id(subjects: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return list(subjects)


def make_dataset(subjects: Sequence[Dict[str, str]], train: bool, patch_size=(128, 128, 128), spacing=(1.0, 1.0, 1.0)):
    keys = ["t1", "t1ce", "t2", "flair", "seg"]
    base = train_transforms(keys, patch_size, spacing) if train else val_transforms(keys, spacing)
    transforms = torch.nn.Sequential()  # placeholder not used
    # MONAI Compose supports call chaining better than Sequential
    from monai.transforms import Compose
    tfm = Compose([base, RemapBraTSLabels(keys=["seg"]), StackModalities(keys=keys), DropUnusedKeys(keys=["image", "seg"])])
    return Dataset(data=_with_id(subjects), transform=tfm)


def make_loaders(train_json: str, val_json: str, batch_size: int, val_batch_size: int, num_workers: int, patch_size, spacing):
    train_subjects = load_json(train_json)
    val_subjects = load_json(val_json)
    train_ds = make_dataset(train_subjects, train=True, patch_size=patch_size, spacing=spacing)
    val_ds = make_dataset(val_subjects, train=False, patch_size=patch_size, spacing=spacing)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=max(1, num_workers // 2), pin_memory=True)
    return train_loader, val_loader
