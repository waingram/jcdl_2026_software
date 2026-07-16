#!/usr/bin/env python3
"""Measure how many training pairs are suppressed by confidence interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
N_LABELS = 17
EPS = 1.0e-7
P1_COLS = [f"p1_sdg{i:02d}" for i in range(1, N_LABELS + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=ROOT / "data/processed/publication/student_ready_factorial_train.parquet",
        help="Publication factorial-training parquet with untempered p1_sdgXX columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis/intervention_coverage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probabilities = pd.read_parquet(args.train_path, columns=P1_COLS).to_numpy(
        dtype=np.float64
    )
    probabilities = np.clip(probabilities, EPS, 1.0 - EPS)
    logits = np.log(probabilities) - np.log1p(-probabilities)

    mask_excluded = (probabilities > 0.25) & (probabilities < 0.75)
    weighted_logit_limit = 2.0 * np.log(0.55 / 0.45)
    weighted_excluded = np.abs(logits) < weighted_logit_limit

    total = int(probabilities.size)
    rows = [
        {
            "intervention": "masked_t1_tau50",
            "criterion": "0.25 < p_t < 0.75",
            "excluded_pairs": int(mask_excluded.sum()),
            "total_pairs": total,
            "excluded_fraction": float(mask_excluded.mean()),
        },
        {
            "intervention": "weighted_t2_tau10",
            "criterion": f"|z_t| < {weighted_logit_limit:.12f}",
            "excluded_pairs": int(weighted_excluded.sum()),
            "total_pairs": total,
            "excluded_fraction": float(weighted_excluded.mean()),
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "intervention_coverage.csv"
    json_path = args.output_dir / "intervention_coverage.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['intervention']}: {row['excluded_pairs']:,}/{total:,} "
            f"({100.0 * row['excluded_fraction']:.2f}%)"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
