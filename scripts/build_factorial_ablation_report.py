#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from sdg_distillation.reporting.factorial_ablation import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the corrected factorial ablation validation report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/factorial_ablation_validation_v1.yaml"),
        help="Path to factorial ablation report YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_run_df, summary_df, out_dir = build_report(args.config)
    print(f"Wrote report to {out_dir}")
    print(f"Per-run rows: {len(per_run_df)}")
    print(f"Condition rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
