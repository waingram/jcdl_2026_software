#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from sdg_distillation.reporting.confidence_bins import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the held-out confidence-binned comparison report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/heldout_confidence_bins_v1.yaml"),
        help="Path to the confidence-binned report YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_df, out_dir = build_report(args.config)
    print(f"Wrote report to {out_dir}")
    print(f"Rows: {len(rows_df)}")


if __name__ == "__main__":
    main()
