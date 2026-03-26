from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple
import json


@dataclass
class Config:
    # Paths
    brats2020_dir: str = "data/BraTS2020"
    brats2019_dir: str = "data/BraTS2019"
    output_dir: str = "outputs"
    train_split_json: str = "data_splits/train_subjects.json"
    val_split_json: str = "data_splits/val_subjects.json"
    test_split_json: str = "data_splits/test_subjects.json"

    # Data
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    roi_size: Tuple[int, int, int] = (128, 128, 128)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    num_workers: int = 4
    batch_size: int = 2
    val_batch_size: int = 1
    in_channels: int = 4
    num_classes: int = 4  # 0 bg, 1 NCR/NET, 2 ED, 3 ET after remap from 4->3

    # Training
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2
    base_channels: int = 16
    seed: int = 42
    amp: bool = True
    grad_clip: float = 12.0
    explain_lambda: float = 0.1
    consistency_lambda: float = 0.05
    mc_dropout_passes: int = 8

    # Misc
    device: str = "cuda"
    save_every: int = 10

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


cfg = Config()
