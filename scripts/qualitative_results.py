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
    parser = argparse.ArgumentParser(description="Generate qualitative results from a saved GRACE checkpoint")
    parser.add_argument("--config", type=str, default="grace/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_grace.pt or last_checkpoint.pt")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override data.dataset_root from config")
    parser.add_argument("--test-csv", type=str, default=None, help="Override data.test_csv from config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test samples for qualitative results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.data.dataset_root = args.dataset_root
    if args.test_csv:
        cfg.data.test_csv = args.test_csv
    if args.output_dir:
        cfg.output_dir = args.output_dir

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    from grace.qualitative import run_qualitative

    result = run_qualitative(
        cfg,
        checkpoint_path=args.checkpoint,
        n_samples=args.num_samples,
        seed=args.seed,
    )

    print("Qualitative analysis finished.")
    print(f"Output dir: {result['output_dir']}")
    print(f"Samples: {result['num_samples']}")
    print(f"Report: {result['report']}")
    print(f"Table: {result['table']}")


if __name__ == "__main__":
    main()
