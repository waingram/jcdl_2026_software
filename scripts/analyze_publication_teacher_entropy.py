#!/usr/bin/env python3
"""Compute publication-test teacher entropy overall and by confidence bin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


N_LABELS = 17
EPS = 1.0e-7
TEACHER_PROB_COLS = [f"teacher_prob_sdg{i:02d}" for i in range(1, N_LABELS + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs/evaluation/heldout_plain_bce_t1_best_v1/predictions.parquet",
        help="Publication held-out predictions parquet containing teacher_prob_sdgXX columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis/publication_teacher_entropy_v1",
        help="Directory for entropy summary outputs.",
    )
    parser.add_argument(
        "--bin-edges",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
        help="Teacher-confidence bin edges.",
    )
    return parser.parse_args()


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float64), EPS, 1.0 - EPS)


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = _clip(probs)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def _bin_labels(edges: list[float]) -> list[str]:
    labels: list[str] = []
    for idx, (lo, hi) in enumerate(zip(edges, edges[1:])):
        right = "]" if idx == len(edges) - 2 else ")"
        labels.append(f"[{lo:.2f}, {hi:.2f}{right}")
    return labels


def render_latex(summary: pd.DataFrame, overall: dict[str, object]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption[Teacher entropy by confidence bin]{Mean teacher entropy over held-out publication document--\ac{SDG} pairs, computed from the untempered teacher probability $p_t$ using natural logs. The overall value is averaged over all documents and all seventeen goals; bin rows use the same teacher-confidence bins as the confidence-binned evaluation table.}",
        r"  \label{tab:publication-teacher-entropy}",
        r"  \small",
        r"  \begin{tabular}{@{} l r r @{}}",
        r"    \toprule",
        r"    \textbf{Region} & \textbf{Instances} & \textbf{Mean entropy (nats)} \\",
        r"    \midrule",
        f"    Overall & {int(overall['n_instances']):,} & {float(overall['mean_teacher_entropy_nats']):.4f}" + r" \\",
        r"    \midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"    {row.confidence_bin} & {row.n_instances:,} & {row.mean_teacher_entropy_nats:.4f}" + r" \\"
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
    df = pd.read_parquet(args.predictions, columns=TEACHER_PROB_COLS)
    teacher_probs = df[TEACHER_PROB_COLS].to_numpy(dtype=np.float64)
    entropy = _entropy(teacher_probs)
    confidence = 2.0 * np.abs(teacher_probs - 0.5)

    flat_entropy = entropy.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    edges = [float(x) for x in args.bin_edges]
    cut_edges = list(edges)
    cut_edges[-1] = float(np.nextafter(cut_edges[-1], np.inf))
    labels = _bin_labels(edges)
    bins = pd.cut(flat_confidence, bins=cut_edges, labels=labels, include_lowest=True, right=False)

    rows: list[dict[str, object]] = []
    for idx, label in enumerate(labels):
        mask = np.asarray(bins == label)
        values = flat_entropy[mask]
        rows.append(
            {
                "confidence_bin": label,
                "bin_index": idx,
                "bin_lower": edges[idx],
                "bin_upper": edges[idx + 1],
                "n_instances": int(values.size),
                "mean_teacher_entropy_nats": float(values.mean()),
                "sd_teacher_entropy_nats": float(values.std(ddof=1)),
                "median_teacher_entropy_nats": float(np.median(values)),
            }
        )

    summary = pd.DataFrame(rows)
    overall = {
        "n_documents": int(df.shape[0]),
        "n_labels": N_LABELS,
        "n_instances": int(flat_entropy.size),
        "epsilon": EPS,
        "mean_teacher_entropy_nats": float(flat_entropy.mean()),
        "sd_teacher_entropy_nats": float(flat_entropy.std(ddof=1)),
        "median_teacher_entropy_nats": float(np.median(flat_entropy)),
        "averaging_convention": "elementwise mean over all document-goal pairs; equivalent to document mean over 17 goals then corpus mean",
        "predictions_path": str(args.predictions),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "teacher_entropy_by_confidence_bin.csv"
    overall_path = args.output_dir / "teacher_entropy_summary.json"
    tex_path = args.output_dir / "teacher_entropy_by_confidence_bin.tex"
    summary.to_csv(summary_path, index=False)
    overall_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    tex_path.write_text(render_latex(summary, overall), encoding="utf-8")

    print(json.dumps(overall, indent=2))
    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
