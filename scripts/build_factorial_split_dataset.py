#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from sdg_distillation.training.splits import build_teacher_confidence_strata, split_summary, split_train_val_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build one fixed, stratified train/validation split from a soft-target training artifact "
            "and project it onto both the soft-target and hard-label student-ready datasets."
        )
    )
    p.add_argument(
        "--soft-train-path",
        type=Path,
        default=ROOT / "data/processed/publication/student_ready_train.parquet",
    )
    p.add_argument(
        "--hard-train-path",
        type=Path,
        default=ROOT / "data/processed/publication/student_ready_hard_train.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/processed/publication",
    )
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260315)
    p.add_argument("--boundary-eps", type=float, default=0.05)
    p.add_argument(
        "--soft-stem",
        type=str,
        default="student_ready_factorial",
        help="Writes <stem>_train.parquet and <stem>_val.parquet",
    )
    p.add_argument(
        "--hard-stem",
        type=str,
        default="student_ready_hard_factorial",
        help="Writes <stem>_train.parquet and <stem>_val.parquet",
    )
    p.add_argument(
        "--summary-name",
        type=str,
        default="factorial_split_summary.json",
        help="Filename for the JSON split summary.",
    )
    p.add_argument(
        "--strata-name",
        type=str,
        default="factorial_split_strata.csv",
        help="Filename for the per-sample stratum assignment export.",
    )
    p.add_argument(
        "--split-column-name",
        type=str,
        default="factorial_split",
        help="Column name used in the exported strata CSV for the train/val assignment.",
    )
    p.add_argument("--write-csv", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    soft_df = pd.read_parquet(args.soft_train_path).reset_index(drop=True)
    hard_df = pd.read_parquet(args.hard_train_path).reset_index(drop=True)

    soft_ids = set(soft_df["sample_id"].astype(str))
    hard_ids = set(hard_df["sample_id"].astype(str))
    if soft_ids != hard_ids:
        raise ValueError("soft and hard train datasets do not share the same sample_id set")

    soft_train_df, soft_val_df = split_train_val_df(
        soft_df,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        boundary_eps=float(args.boundary_eps),
    )

    val_ids = set(soft_val_df["sample_id"].astype(str))
    hard_df = hard_df.copy()
    hard_df["sample_id"] = hard_df["sample_id"].astype(str)
    hard_val_df = hard_df[hard_df["sample_id"].isin(val_ids)].copy()
    hard_train_df = hard_df[~hard_df["sample_id"].isin(val_ids)].copy()

    soft_train_out = args.out_dir / f"{args.soft_stem}_train.parquet"
    soft_val_out = args.out_dir / f"{args.soft_stem}_val.parquet"
    hard_train_out = args.out_dir / f"{args.hard_stem}_train.parquet"
    hard_val_out = args.out_dir / f"{args.hard_stem}_val.parquet"
    summary_out = args.out_dir / str(args.summary_name)
    strata_out = args.out_dir / str(args.strata_name)

    soft_train_df.to_parquet(soft_train_out, index=False)
    soft_val_df.to_parquet(soft_val_out, index=False)
    hard_train_df.to_parquet(hard_train_out, index=False)
    hard_val_df.to_parquet(hard_val_out, index=False)

    if args.write_csv:
        soft_train_df.to_csv(soft_train_out.with_suffix('.csv'), index=False)
        soft_val_df.to_csv(soft_val_out.with_suffix('.csv'), index=False)
        hard_train_df.to_csv(hard_train_out.with_suffix('.csv'), index=False)
        hard_val_df.to_csv(hard_val_out.with_suffix('.csv'), index=False)

    strata = build_teacher_confidence_strata(soft_df, boundary_eps=float(args.boundary_eps))
    split_map = pd.DataFrame(
        {
            "sample_id": pd.concat([
                soft_train_df["sample_id"].astype(str),
                soft_val_df["sample_id"].astype(str),
            ], ignore_index=True),
            str(args.split_column_name): ["train"] * len(soft_train_df) + ["val"] * len(soft_val_df),
        }
    )
    strata = strata.merge(split_map, on="sample_id", how="left", validate="1:1")
    strata.to_csv(strata_out, index=False)

    summary = {
        "inputs": {
            "soft_train_path": str(args.soft_train_path),
            "hard_train_path": str(args.hard_train_path),
            "val_fraction": float(args.val_fraction),
            "seed": int(args.seed),
            "boundary_eps": float(args.boundary_eps),
        },
        "outputs": {
            "soft_train": str(soft_train_out),
            "soft_val": str(soft_val_out),
            "hard_train": str(hard_train_out),
            "hard_val": str(hard_val_out),
            "strata_csv": str(strata_out),
        },
        "full": split_summary(soft_df, boundary_eps=float(args.boundary_eps)),
        "train": split_summary(soft_train_df, boundary_eps=float(args.boundary_eps)),
        "val": split_summary(soft_val_df, boundary_eps=float(args.boundary_eps)),
        "actual_val_fraction": float(len(soft_val_df) / len(soft_df)),
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8")

    print(f"Wrote {soft_train_out} ({len(soft_train_df)} rows)")
    print(f"Wrote {soft_val_out} ({len(soft_val_df)} rows)")
    print(f"Wrote {hard_train_out} ({len(hard_train_df)} rows)")
    print(f"Wrote {hard_val_out} ({len(hard_val_df)} rows)")
    print(f"Wrote {strata_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
