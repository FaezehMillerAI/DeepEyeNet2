#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def check_split(root: Path, csv_name: str) -> tuple[int, int]:
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["image_path", "Keywords", "clinical-description", "report_text"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

    missing_files = 0
    for p in df["image_path"].astype(str):
        if not (root / p).exists():
            missing_files += 1
    return len(df), missing_files


def main():
    parser = argparse.ArgumentParser(description="Validate DeepEyeNet folder layout and file paths")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--train-csv", default="train.csv", type=str)
    parser.add_argument("--valid-csv", default="valid.csv", type=str)
    parser.add_argument("--test-csv", default="test.csv", type=str)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    for split, csv_name in [("train", args.train_csv), ("valid", args.valid_csv), ("test", args.test_csv)]:
        rows, missing = check_split(root, csv_name)
        print(f"{split}: rows={rows}, missing_image_files={missing}")

    print("Dataset check complete.")


if __name__ == "__main__":
    main()
