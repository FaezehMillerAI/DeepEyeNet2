#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train GRACE on DeepEyeNet")
    parser.add_argument("--config", type=str, default="grace/configs/default.yaml")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override data.dataset_root from config")
    parser.add_argument("--train-csv", type=str, default=None, help="Override data.train_csv from config")
    parser.add_argument("--valid-csv", type=str, default=None, help="Override data.valid_csv from config")
    parser.add_argument("--test-csv", type=str, default=None, help="Override data.test_csv from config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.data.dataset_root = args.dataset_root
    if args.train_csv:
        cfg.data.train_csv = args.train_csv
    if args.valid_csv:
        cfg.data.valid_csv = args.valid_csv
    if args.test_csv:
        cfg.data.test_csv = args.test_csv
    if args.output_dir:
        cfg.output_dir = args.output_dir

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    from grace.train import train_grace

    result = train_grace(cfg)

    print("Training finished.")
    print(f"Output dir: {result['output_dir']}")
    print("Test metrics:")
    for k, v in result["test_metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
