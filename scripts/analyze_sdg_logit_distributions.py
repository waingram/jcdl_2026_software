#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


LABELS = [f"sdg{i:02d}" for i in range(1, 18)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-SDG logit distributions for one or more teacher-labeled artifacts "
            "and one or more student-prediction artifacts."
        )
    )
    parser.add_argument(
        "--teacher-paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more CSV/parquet files with teacher_logit_sdgXX columns.",
    )
    parser.add_argument(
        "--student-paths",
        type=Path,
        nargs="*",
        default=[],
        help="Optional CSV/parquet files with student_logit_sdgXX columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis",
        help="Parent output directory for the report.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        required=True,
        help="Output report directory name.",
    )
    parser.add_argument(
        "--sdgs-path",
        type=Path,
        default=ROOT / "configs/sdgs.yaml",
        help="SDG metadata config for titles.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type for {path}")


def _load_many(paths: list[Path]) -> pd.DataFrame:
    frames = [_read_table(path) for path in paths]
    if not frames:
        raise ValueError("At least one path is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_sdg_titles(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    sdgs = payload.get("sdgs", {})
    titles: dict[str, str] = {}
    for i in range(1, 18):
        key = str(i)
        label = f"sdg{i:02d}"
        titles[label] = str(sdgs.get(key, {}).get("title", label))
    return titles


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_arr = np.abs(arr)
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "mean_abs": float(np.mean(abs_arr)),
        "median_abs": float(np.median(abs_arr)),
        "p75_abs": float(np.quantile(abs_arr, 0.75)),
        "p95_abs": float(np.quantile(abs_arr, 0.95)),
        "positive_rate": float(np.mean(arr > 0.0)),
        "negative_rate": float(np.mean(arr < 0.0)),
        "near_boundary_abs_lt_0_5": float(np.mean(abs_arr < 0.5)),
        "near_boundary_abs_lt_1_0": float(np.mean(abs_arr < 1.0)),
        "near_boundary_abs_lt_2_0": float(np.mean(abs_arr < 2.0)),
    }


def _summarize_frame(df: pd.DataFrame, *, logit_prefix: str, titles: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label_idx, label in enumerate(LABELS):
        col = f"{logit_prefix}{label}"
        if col not in df.columns:
            raise ValueError(f"Missing required column {col!r}")
        stats = _distribution_stats(df[col].to_numpy(dtype=np.float64))
        rows.append(
            {
                "label_index": label_idx + 1,
                "label": label,
                "title": titles.get(label, label),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _overall_summary(summary_df: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_of_label_means": float(summary_df["mean"].mean()),
        "mean_of_label_std": float(summary_df["std"].mean()),
        "mean_of_label_median_abs": float(summary_df["median_abs"].mean()),
        "mean_positive_rate": float(summary_df["positive_rate"].mean()),
        "mean_near_boundary_abs_lt_1_0": float(summary_df["near_boundary_abs_lt_1_0"].mean()),
    }


def _merge_summaries(teacher_summary: pd.DataFrame, student_summary: pd.DataFrame) -> pd.DataFrame:
    merged = teacher_summary.merge(
        student_summary,
        on=["label_index", "label", "title"],
        suffixes=("_teacher", "_student"),
        how="inner",
        validate="one_to_one",
    )
    merged["mean_shift_student_minus_teacher"] = merged["mean_student"] - merged["mean_teacher"]
    merged["median_shift_student_minus_teacher"] = merged["median_student"] - merged["median_teacher"]
    merged["median_abs_shift_student_minus_teacher"] = merged["median_abs_student"] - merged["median_abs_teacher"]
    merged["positive_rate_shift_student_minus_teacher"] = merged["positive_rate_student"] - merged["positive_rate_teacher"]
    merged["near_boundary_abs_lt_1_0_shift_student_minus_teacher"] = (
        merged["near_boundary_abs_lt_1_0_student"] - merged["near_boundary_abs_lt_1_0_teacher"]
    )
    return merged


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / args.report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    titles = _load_sdg_titles(args.sdgs_path)
    teacher_df = _load_many(list(args.teacher_paths))
    teacher_summary = _summarize_frame(teacher_df, logit_prefix="teacher_logit_", titles=titles)
    teacher_out = report_dir / "teacher_logit_distribution_by_sdg.csv"
    teacher_summary.to_csv(teacher_out, index=False)

    summary_payload: dict[str, Any] = {
        "teacher_paths": [str(path) for path in args.teacher_paths],
        "student_paths": [str(path) for path in args.student_paths],
        "report_dir": str(report_dir),
        "teacher_rows": int(len(teacher_df)),
        "teacher_overall": _overall_summary(teacher_summary),
    }
    print(f"Wrote {teacher_out}")

    if args.student_paths:
        student_df = _load_many(list(args.student_paths))
        student_summary = _summarize_frame(student_df, logit_prefix="student_logit_", titles=titles)
        student_out = report_dir / "student_logit_distribution_by_sdg.csv"
        student_summary.to_csv(student_out, index=False)
        print(f"Wrote {student_out}")

        merged = _merge_summaries(teacher_summary, student_summary)
        merged_out = report_dir / "teacher_student_distribution_comparison_by_sdg.csv"
        merged.to_csv(merged_out, index=False)
        print(f"Wrote {merged_out}")

        summary_payload["student_rows"] = int(len(student_df))
        summary_payload["student_overall"] = _overall_summary(student_summary)
        summary_payload["teacher_student_distribution_comparison"] = {
            "mean_of_mean_shift": float(merged["mean_shift_student_minus_teacher"].mean()),
            "mean_of_median_abs_shift": float(merged["median_abs_shift_student_minus_teacher"].mean()),
            "mean_of_positive_rate_shift": float(merged["positive_rate_shift_student_minus_teacher"].mean()),
            "largest_positive_rate_shift_label": str(
                merged.loc[merged["positive_rate_shift_student_minus_teacher"].idxmax(), "label"]
            ),
            "largest_positive_rate_shift_value": float(
                merged["positive_rate_shift_student_minus_teacher"].max()
            ),
        }

    summary_out = report_dir / "summary.json"
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
