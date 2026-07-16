#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from sdg_distillation.reporting.etd_transfer_summary import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the three-condition ETD transfer summary report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/etd_transfer_three_way_v1.yaml"),
        help="Path to ETD transfer summary report YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, out_dir = build_report(args.config)
    print(f"Wrote report to {out_dir}")
    print(f"Condition rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
