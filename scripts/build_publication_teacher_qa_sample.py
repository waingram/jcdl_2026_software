#!/usr/bin/env python3
"""Build a simple expert-QA sample from the publication teacher corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INPUTS = [
    ROOT / "data/processed/publication/student_ready_train.parquet",
    ROOT / "data/processed/publication/student_ready_test.parquet",
]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/expert_qa"

LABELS = [f"sdg{i:02d}" for i in range(1, 18)]
P1_COLS = [f"p1_{label}" for label in LABELS]
HARD_COLS = [f"hard_label_{label}" for label in LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-paths",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more publication student_ready parquet files to sample from.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of rows to sample for expert QA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the sampled CSV and manifest.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Optional custom output stem. Defaults to 'publication_teacher_qa_sample_n{N}_seed{seed}'.",
    )
    parser.add_argument(
        "--minimal-columns",
        action="store_true",
        help="Write only sample_id, split, retrieval_sdg, doi, hard_label_vector, and abstract_text.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type for {path}")


def _coalesce_doi(df: pd.DataFrame) -> pd.Series:
    source = df.get("source_doi", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    teacher = df.get("teacher_doi", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    return source.where(source.ne(""), teacher)


def _positive_sdgs(hard: np.ndarray) -> list[str]:
    out: list[str] = []
    for row in hard:
        positives = [str(i + 1) for i, value in enumerate(row.tolist()) if int(value) == 1]
        out.append("|".join(positives))
    return out


def main() -> None:
    args = parse_args()
    frames = [_read_table(path) for path in args.input_paths]
    df = pd.concat(frames, ignore_index=True, sort=False)

    missing = [col for col in ["sample_id", "split", "retrieval_sdg", "text", *P1_COLS] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_rows = len(df)
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.sample_size > n_rows:
        raise ValueError(f"--sample-size={args.sample_size} exceeds corpus size {n_rows}")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_rows, size=args.sample_size, replace=False)
    sample = df.iloc[np.sort(sample_idx)].reset_index(drop=True).copy()

    p1 = sample[P1_COLS].to_numpy(dtype=np.float64)
    hard = (p1 >= 0.5).astype(np.int64)
    for col_idx, col in enumerate(HARD_COLS):
        sample[col] = hard[:, col_idx]

    sample["doi"] = _coalesce_doi(sample)
    sample["abstract_text"] = sample["text"].fillna("").astype(str)
    sample["positive_sdgs"] = _positive_sdgs(hard)
    sample["hard_label_vector"] = [json.dumps(row.tolist()) for row in hard]
    sample["num_positive_sdgs"] = hard.sum(axis=1).astype(int)

    if args.minimal_columns:
        out_cols = [
            "sample_id",
            "split",
            "retrieval_sdg",
            "doi",
            "hard_label_vector",
            "abstract_text",
        ]
    else:
        out_cols = [
            "sample_id",
            "split",
            "retrieval_sdg",
            "doi",
            "source_doi",
            "teacher_doi",
            "num_positive_sdgs",
            "positive_sdgs",
            "hard_label_vector",
            "abstract_text",
            *HARD_COLS,
        ]
    out_cols = [col for col in out_cols if col in sample.columns]
    out = sample[out_cols].copy()

    stem = args.stem or f"publication_teacher_qa_sample_n{args.sample_size}_seed{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{stem}.csv"
    manifest_path = args.output_dir / f"{stem}.manifest.json"

    out.to_csv(csv_path, index=False)

    manifest = {
        "input_paths": [str(path) for path in args.input_paths],
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "minimal_columns": bool(args.minimal_columns),
        "output_csv": str(csv_path),
        "n_source_rows": int(n_rows),
        "n_sample_rows": int(len(out)),
        "mean_positive_sdgs_in_sample": float(sample["num_positive_sdgs"].mean()),
        "rows_with_any_positive": int((sample["num_positive_sdgs"] > 0).sum()),
        "rows_with_empty_doi": int((sample["doi"].fillna("").astype(str).str.strip() == "").sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {manifest_path}")
    print(
        "Sample summary: "
        f"n={len(out)} any_positive={(sample['num_positive_sdgs'] > 0).sum()} "
        f"mean_positive_sdgs={sample['num_positive_sdgs'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
