#!/usr/bin/env python3

# Description: Join wide teacher-label targets with processed abstract text
# to build canonical student-ready train and test datasets for
# distillation experiments.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/build_student_ready_dataset.py [--teacher-dir DIR --processed-dir DIR --out-dir DIR --year YYYY --target-set {p1,teacher_logit,hard_label,both} --output-stem STEM --text-col NAME --doi-col NAME --write-csv --strict]

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd


TargetSet = Literal["p1", "teacher_logit", "hard_label", "both"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for student-ready dataset construction."""

    p = argparse.ArgumentParser(
        description=(
            "Join teacher-wide SDG targets with processed abstract text to build "
            "canonical student-ready datasets."
        )
    )
    p.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path("outputs/teacher/publication"),
        help="Directory containing teacher_grid_wide_{train,test}.csv",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing sdg{n}_2023_{train,test}.csv source files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/publication"),
        help="Output directory for student_ready_{train,test}.{csv,parquet}.",
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--target-set",
        choices=["p1", "teacher_logit", "hard_label", "both"],
        default="both",
        help="Which soft target columns to keep in student-ready outputs.",
    )
    p.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help=(
            "Prefix for output files. Defaults to 'student_ready' for soft targets "
            "and 'student_ready_hard' for hard-label targets."
        ),
    )
    p.add_argument(
        "--text-col",
        type=str,
        default="Abstract",
        help="Text column in processed source CSVs.",
    )
    p.add_argument(
        "--doi-col",
        type=str,
        default="DOI",
        help="DOI column in processed source CSVs.",
    )
    p.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV in addition to Parquet.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on DOI mismatches when both teacher/source DOI are present.",
    )
    return p.parse_args()


def _target_column_mapping(df: pd.DataFrame, target_set: TargetSet) -> dict[str, str]:
    p1_cols = [f"p1_sdg{i:02d}" for i in range(1, 18)]
    t_cols = [f"teacher_logit_sdg{i:02d}" for i in range(1, 18)]
    label_cols = [f"label_bit_sdg{i:02d}" for i in range(1, 18)]
    required = set(p1_cols + t_cols + label_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Teacher wide file missing target columns: {missing[:5]} ...")

    if target_set == "p1":
        return {c: c for c in p1_cols}
    if target_set == "teacher_logit":
        return {c: c for c in t_cols}
    if target_set == "hard_label":
        return {f"label_bit_sdg{i:02d}": f"hard_label_sdg{i:02d}" for i in range(1, 18)}
    return {c: c for c in p1_cols + t_cols}


def _load_processed(
    path: Path,
    text_col: str,
    doi_col: str,
    retrieval_sdg: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    src = pd.read_csv(path)
    for c in (text_col, doi_col):
        if c not in src.columns:
            raise ValueError(f"{path.name} missing required column {c!r}")

    src = src[[doi_col, text_col]].copy()
    src["row_id"] = range(len(src))
    src["retrieval_sdg"] = retrieval_sdg
    src["source_doi"] = src[doi_col].fillna("").astype(str).str.strip()
    src["text"] = src[text_col].fillna("").astype(str)
    return src[["retrieval_sdg", "row_id", "source_doi", "text"]]


def _join_one_split(
    *,
    split: str,
    teacher_dir: Path,
    processed_dir: Path,
    year: int,
    text_col: str,
    doi_col: str,
    target_set: TargetSet,
    strict: bool,
) -> pd.DataFrame:
    teacher_path = teacher_dir / f"teacher_grid_wide_{split}.csv"
    if not teacher_path.exists():
        raise FileNotFoundError(teacher_path)

    teacher = pd.read_csv(teacher_path)
    required = {"retrieval_sdg", "split", "row_id", "DOI", "is_complete_17d"}
    missing = required - set(teacher.columns)
    if missing:
        raise ValueError(f"{teacher_path.name} missing required columns: {sorted(missing)}")

    # Keep only complete 17D rows to avoid partial targets in training.
    teacher = teacher[teacher["is_complete_17d"].astype(bool)].copy()
    teacher["retrieval_sdg"] = pd.to_numeric(teacher["retrieval_sdg"], errors="raise").astype(int)
    teacher["row_id"] = pd.to_numeric(teacher["row_id"], errors="raise").astype(int)
    teacher["teacher_doi"] = teacher["DOI"].fillna("").astype(str).str.strip()

    target_map = _target_column_mapping(teacher, target_set=target_set)
    teacher_keep = ["retrieval_sdg", "row_id", "split", "teacher_doi"] + list(target_map.keys())
    teacher = teacher[teacher_keep]
    teacher = teacher.rename(columns=target_map)
    out_target_cols = list(target_map.values())

    frames: list[pd.DataFrame] = []
    for retrieval_sdg in sorted(teacher["retrieval_sdg"].unique().tolist()):
        src_path = processed_dir / f"sdg{retrieval_sdg}_{year}_{split}.csv"
        src = _load_processed(
            src_path,
            text_col=text_col,
            doi_col=doi_col,
            retrieval_sdg=retrieval_sdg,
        )

        t = teacher[teacher["retrieval_sdg"] == retrieval_sdg].copy()
        merged = t.merge(
            src,
            on=["retrieval_sdg", "row_id"],
            how="left",
            validate="one_to_one",
        )

        if merged["text"].isna().any():
            n_missing = int(merged["text"].isna().sum())
            raise ValueError(
                f"{split}/sdg{retrieval_sdg}: {n_missing} rows missing text after join. "
                f"Check row alignment and source file."
            )

        # DOI consistency diagnostics.
        both = (merged["teacher_doi"] != "") & (merged["source_doi"] != "")
        mism = both & (merged["teacher_doi"] != merged["source_doi"])
        n_both = int(both.sum())
        n_mism = int(mism.sum())
        if n_mism > 0:
            msg = (
                f"{split}/sdg{retrieval_sdg}: DOI mismatches where both present: "
                f"{n_mism}/{n_both}"
            )
            if strict:
                raise ValueError(msg)
            print(f"WARNING: {msg}")

        merged["sample_id"] = (
            merged["retrieval_sdg"].astype(str) + "_" + merged["split"] + "_" + merged["row_id"].astype(str)
        )

        # Canonical column order.
        front = ["sample_id", "retrieval_sdg", "split", "row_id", "source_doi", "teacher_doi", "text"]
        merged = merged[front + out_target_cols]
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    return out


def main() -> None:
    """Build student-ready train and test datasets from teacher and source files."""

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem
    if output_stem is None:
        output_stem = "student_ready_hard" if args.target_set == "hard_label" else "student_ready"

    for split in ("train", "test"):
        df = _join_one_split(
            split=split,
            teacher_dir=args.teacher_dir,
            processed_dir=args.processed_dir,
            year=args.year,
            text_col=args.text_col,
            doi_col=args.doi_col,
            target_set=args.target_set,
            strict=args.strict,
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
