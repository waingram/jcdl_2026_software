#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


TargetSet = Literal["p1", "teacher_logit", "hard_label", "both"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build student-ready ETD distillation datasets directly from the "
            "VTechWorks teacher_grid_wide_{train,test}.csv artifacts."
        )
    )
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=ROOT / "outputs/teacher/etd",
        help="Directory containing teacher_grid_wide_{train,test}.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/processed/etd",
        help="Output directory for student-ready ETD datasets.",
    )
    parser.add_argument(
        "--target-set",
        choices=["p1", "teacher_logit", "hard_label", "both"],
        default="both",
        help="Which target columns to keep.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Prefix for output files. Defaults to student_ready or student_ready_hard.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="abstract",
        help="Text column in the VTechWorks teacher-wide files.",
    )
    parser.add_argument(
        "--retrieval-sdg",
        type=int,
        default=0,
        help="Synthetic retrieval_sdg value to satisfy the training data contract.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV output next to Parquet.",
    )
    return parser.parse_args()


def _target_column_mapping(df: pd.DataFrame, target_set: TargetSet) -> dict[str, str]:
    p1_cols = [f"p1_sdg{i:02d}" for i in range(1, 18)]
    logit_cols = [f"teacher_logit_sdg{i:02d}" for i in range(1, 18)]
    hard_cols = [f"label_bit_sdg{i:02d}" for i in range(1, 18)]
    required = set(p1_cols + logit_cols + hard_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Teacher wide file missing target columns: {missing[:5]} ...")

    if target_set == "p1":
        return {c: c for c in p1_cols}
    if target_set == "teacher_logit":
        return {c: c for c in logit_cols}
    if target_set == "hard_label":
        return {f"label_bit_sdg{i:02d}": f"hard_label_sdg{i:02d}" for i in range(1, 18)}
    return {c: c for c in p1_cols + logit_cols}


def _build_one_split(*, teacher_dir: Path, split: str, text_col: str, target_set: TargetSet, retrieval_sdg: int) -> pd.DataFrame:
    teacher_path = teacher_dir / f"teacher_grid_wide_{split}.csv"
    if not teacher_path.exists():
        raise FileNotFoundError(teacher_path)

    df = pd.read_csv(teacher_path)
    required = {"sample_id", "split", "row_id", text_col, "is_complete_17d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{teacher_path.name} missing required columns: {sorted(missing)}")

    df = df[df["is_complete_17d"].astype(bool)].copy()
    df["retrieval_sdg"] = int(retrieval_sdg)
    df["text"] = df[text_col].fillna("").astype(str)

    target_map = _target_column_mapping(df, target_set=target_set)
    keep_metadata = [
        c
        for c in [
            "sample_id",
            "retrieval_sdg",
            "split",
            "row_id",
            "uri",
            "source_set",
            "oai_identifier",
            "department_normalized",
            "degree",
            "title",
            "text",
        ]
        if c in df.columns
    ]
    keep_cols = keep_metadata + list(target_map.keys())
    df = df.loc[:, keep_cols].rename(columns=target_map)
    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_stem = args.output_stem
    if output_stem is None:
        output_stem = "student_ready_hard" if args.target_set == "hard_label" else "student_ready"

    for split in ("train", "test"):
        df = _build_one_split(
            teacher_dir=args.teacher_dir,
            split=split,
            text_col=str(args.text_col),
            target_set=args.target_set,
            retrieval_sdg=int(args.retrieval_sdg),
        )
        parquet_path = args.out_dir / f"{output_stem}_{split}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path} ({len(df)} rows, {len(df.columns)} cols)")

        if args.write_csv:
            csv_path = args.out_dir / f"{output_stem}_{split}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Wrote {csv_path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
