#!/usr/bin/env python3
"""Build the Chapter 6 institutional profiling report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdg_distillation.reporting.etd_institutional_profile import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ETD institutional profiling report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/baselines/etd_institutional_profile_v1.yaml",
        help="Path to the institutional profiling report config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, out_dir = build_report(args.config)
    print(f"Wrote {len(summary_df)} department rows to {out_dir}")


if __name__ == "__main__":
    main()
