from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import argparse

from sklearn.model_selection import train_test_split

from utils.common import save_json, seed_everything


MODALITY_KEYS = ["t1", "t1ce", "t2", "flair", "seg"]


def discover_subjects(root: str) -> List[Dict[str, str]]:
    root_path = Path(root)
    subjects: List[Dict[str, str]] = []
    for subject_dir in sorted([p for p in root_path.rglob("*") if p.is_dir()]):
        files = {f.name.lower(): str(f) for f in subject_dir.glob("*.nii*")}
        subj = {}
        sid = subject_dir.name
        for key in MODALITY_KEYS:
            matched = [v for k, v in files.items() if key in k]
            if matched:
                subj[key] = matched[0]
        if all(k in subj for k in MODALITY_KEYS):
            subj["id"] = sid
            subjects.append(subj)
    if not subjects:
        raise FileNotFoundError(f"No complete subjects found in {root}")
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data_splits")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    subjects = discover_subjects(args.root)
    train_subj, test_subj = train_test_split(subjects, test_size=args.test_size, random_state=args.seed)
    rel_val = args.val_size / (1.0 - args.test_size)
    train_subj, val_subj = train_test_split(train_subj, test_size=rel_val, random_state=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(train_subj, out_dir / "train_subjects.json")
    save_json(val_subj, out_dir / "val_subjects.json")
    save_json(test_subj, out_dir / "test_subjects.json")
    print(f"Saved splits in {out_dir}")
    print(f"Train: {len(train_subj)}, Val: {len(val_subj)}, Test: {len(test_subj)}")


if __name__ == "__main__":
    main()
