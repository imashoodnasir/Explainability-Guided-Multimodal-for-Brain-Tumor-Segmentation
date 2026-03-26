from __future__ import annotations
"""
Template script for ablation studies.
Use the flags below to disable modules or adjust depth / explainability strength,
then launch the standard training loop.
"""
import argparse
from config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_fusion", action="store_true")
    parser.add_argument("--disable_explainability", action="store_true")
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--explain_lambda", type=float, default=0.1)
    args = parser.parse_args()

    cfg = Config(base_channels=args.base_channels, explain_lambda=args.explain_lambda)
    print("Ablation config:")
    print(cfg)
    print("disable_fusion=", args.disable_fusion)
    print("disable_explainability=", args.disable_explainability)
    print("Use these settings inside train.py or duplicate train.py with your ablation logic.")


if __name__ == "__main__":
    main()
