#!/usr/bin/env python3
"""Summarize duplicate-teacher logit-margin variation by SDG."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


N_SDGS = 17
LOGIT_COLS = [f"teacher_logit_sdg{i:02d}" for i in range(1, N_SDGS + 1)]
LABEL_COLS = [f"label_bit_sdg{i:02d}" for i in range(1, N_SDGS + 1)]

SDG_SHORT = {
    1: "Poverty",
    2: "Hunger",
    3: "Health",
    4: "Education",
    5: "Gender",
    6: "Water",
    7: "Energy",
    8: "Work",
    9: "Industry",
    10: "Inequality",
    11: "Cities",
    12: "Consumption",
    13: "Climate",
    14: "Oceans",
    15: "Land",
    16: "Institutions",
    17: "Partnerships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--occurrences",
        type=Path,
        default=ROOT / "outputs/analysis/duplicate_teacher_repeatability_v1/duplicate_teacher_occurrences_test.csv",
        help="Duplicate-teacher occurrence CSV or parquet produced by analyze_duplicate_teacher_repeatability.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis/duplicate_teacher_repeatability_v1",
        help="Directory for CSV and LaTeX outputs.",
    )
    parser.add_argument(
        "--manuscript-table",
        type=Path,
        default=None,
        help="Optional manuscript-table path to mirror the LaTeX output.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split label used in output filenames.",
    )
    return parser.parse_args()


def _format_sdg(sdg: int) -> str:
    return f"SDG {sdg:02d} {SDG_SHORT[sdg]}"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_pairwise_sdg_deltas(occurrences: pd.DataFrame) -> pd.DataFrame:
    missing = {"group_id", *LOGIT_COLS, *LABEL_COLS} - set(occurrences.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    ordered = occurrences.sort_values(["group_id", "retrieval_sdg", "row_id"], kind="stable")
    for group_id, group in ordered.groupby("group_id", sort=True):
        if len(group) < 2:
            continue
        logits = group[LOGIT_COLS].to_numpy(dtype=float)
        labels = group[LABEL_COLS].to_numpy(dtype=int)
        for i, j in combinations(range(len(group)), 2):
            abs_delta = np.abs(logits[i] - logits[j])
            hard_flip = labels[i] != labels[j]
            for sdg_idx, value in enumerate(abs_delta, start=1):
                rows.append(
                    {
                        "group_id": group_id,
                        "sdg": sdg_idx,
                        "abs_delta_logit": float(value),
                        "hard_flip": bool(hard_flip[sdg_idx - 1]),
                    }
                )

    return pd.DataFrame(rows)


def summarize_by_sdg(pairwise: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sdg in range(1, N_SDGS + 1):
        sub = pairwise[pairwise["sdg"] == sdg]
        values = sub["abs_delta_logit"].to_numpy(dtype=float)
        rows.append(
            {
                "split": split,
                "sdg": sdg,
                "sdg_label": _format_sdg(sdg),
                "pairwise_observations": int(len(values)),
                "mean_abs_delta_logit": float(np.mean(values)),
                "sd_abs_delta_logit": float(np.std(values, ddof=1)),
                "median_abs_delta_logit": float(np.median(values)),
                "p95_abs_delta_logit": float(np.quantile(values, 0.95)),
                "max_abs_delta_logit": float(np.max(values)),
                "hard_flip_event_count": int(sub["hard_flip"].sum()),
                "hard_flip_rate": float(sub["hard_flip"].mean()),
            }
        )
    return pd.DataFrame(rows)


def render_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption[Per-SDG duplicate-teacher logit-margin variation]{Per-SDG variation in duplicate-teacher logit margins for duplicate DOI groups in the held-out publication test split. Each row summarizes all pairwise repeated-row comparisons for that \ac{SDG}; the reported change is the absolute teacher-margin difference $|\Delta z_t|$.}",
        r"  \label{tab:duplicate-teacher-logit-delta-sdg}",
        r"  \footnotesize",
        r"  \begin{tabular}{@{} l r r r r @{}}",
        r"    \toprule",
        r"    \textbf{SDG} & \textbf{Pairs} & \textbf{Mean $|\Delta z_t|$} & \textbf{SD $|\Delta z_t|$} & \textbf{Hard flips} \\",
        r"    \midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "    "
            f"{row.sdg_label} & "
            f"{row.pairwise_observations:,} & "
            f"{row.mean_abs_delta_logit:.3f} & "
            f"{row.sd_abs_delta_logit:.3f} & "
            f"{row.hard_flip_event_count}"
            r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    occurrences = _read_table(args.occurrences)
    pairwise = build_pairwise_sdg_deltas(occurrences)
    summary = summarize_by_sdg(pairwise, split=args.split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"duplicate_teacher_logit_delta_sdg_summary_{args.split}.csv"
    tex_path = args.output_dir / f"duplicate_teacher_logit_delta_sdg_summary_{args.split}.tex"
    summary.to_csv(csv_path, index=False)
    latex = render_latex(summary)
    tex_path.write_text(latex, encoding="utf-8")

    if args.manuscript_table:
        args.manuscript_table.parent.mkdir(parents=True, exist_ok=True)
        args.manuscript_table.write_text(latex, encoding="utf-8")

    print(summary[["sdg_label", "pairwise_observations", "mean_abs_delta_logit", "sd_abs_delta_logit", "hard_flip_event_count"]].to_string(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {tex_path}")
    if args.manuscript_table:
        print(f"Wrote {args.manuscript_table}")


if __name__ == "__main__":
    main()
