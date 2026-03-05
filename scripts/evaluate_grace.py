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
    parser = argparse.ArgumentParser(description="Evaluate GRACE from a saved checkpoint (no training)")
    parser.add_argument("--config", type=str, default="grace/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_grace.pt or last_checkpoint.pt")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override data.dataset_root from config")
    parser.add_argument("--test-csv", type=str, default=None, help="Override data.test_csv from config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.data.dataset_root = args.dataset_root
    if args.test_csv:
        cfg.data.test_csv = args.test_csv
    if args.output_dir:
        cfg.output_dir = args.output_dir

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    from grace.evaluate import evaluate_grace

    result = evaluate_grace(cfg, checkpoint_path=args.checkpoint)

    print("Evaluation finished.")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Output dir: {result['output_dir']}")
    print("Test metrics:")
    for k, v in result["test_metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
