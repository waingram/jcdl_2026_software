#!/usr/bin/env python3
"""Paired document-level bootstrap comparison for the three ETD transfer regimes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdg_distillation.training.data_contract import sdg_target_columns
from sdg_distillation.training.evaluation import (
    _binary_auroc,
    _brier_score,
    _ece,
    _micro_auroc,
    _precision_recall_f1,
)


PREDICTION_THRESHOLD = 0.5
HARD_LABEL_THRESHOLD = 0.5
CALIBRATION_BINS = 10


@dataclass(frozen=True)
class RegimeSpec:
    key: str
    display_name: str
    path: Path


REGIME_SPECS = [
    RegimeSpec(
        key="zero_shot",
        display_name="Zero-shot",
        path=ROOT / "outputs/inference/etd/zero_shot/student_predictions_merged.parquet",
    ),
    RegimeSpec(
        key="adapted",
        display_name="Adapted",
        path=ROOT / "outputs/inference/etd/adapted/student_predictions_merged.parquet",
    ),
    RegimeSpec(
        key="scratch",
        display_name="Scratch",
        path=ROOT / "outputs/inference/etd/scratch/student_predictions_merged.parquet",
    ),
]

COMPARISONS = [
    ("adapted", "zero_shot"),
    ("adapted", "scratch"),
    ("zero_shot", "scratch"),
]

METRIC_SPECS = [
    ("macro_auroc", "Macro AUROC", True),
    ("micro_auroc", "Micro AUROC", True),
    ("macro_f1", "Macro F1", True),
    ("micro_f1", "Micro F1", True),
    ("soft_mae_p1", "Soft MAE", False),
    ("soft_rmse_p1", "Soft RMSE", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis/etd_bootstrap_transfer_v1",
        help="Directory for CSV/JSON/LaTeX outputs.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of paired bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def _student_prob_columns() -> list[str]:
    return [f"student_prob_sdg{i:02d}" for i in range(1, 18)]


def _load_regime_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["sample_id", *sdg_target_columns("p1"), *_student_prob_columns()]
    df = pd.read_parquet(path, columns=columns).copy()
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        n_dup = int(df["sample_id"].duplicated().sum())
        raise ValueError(f"Duplicate sample_id rows in {path}: {n_dup}")
    return df


def _align_regime_frames(specs: list[RegimeSpec]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    teacher_cols = sdg_target_columns("p1")
    prob_cols = _student_prob_columns()

    base_df = _load_regime_frame(specs[0].path)
    sample_ids = base_df["sample_id"].tolist()
    teacher_probs = base_df[teacher_cols].to_numpy(dtype=np.float32)
    regime_probs: dict[str, np.ndarray] = {
        specs[0].key: base_df[prob_cols].to_numpy(dtype=np.float32)
    }

    for spec in specs[1:]:
        df = _load_regime_frame(spec.path).set_index("sample_id", drop=False)
        aligned = df.reindex(sample_ids)
        if aligned.isnull().any().any():
            missing = aligned.index[aligned[teacher_cols[0]].isnull()].tolist()[:5]
            raise ValueError(f"Missing sample_ids when aligning {spec.key}: {missing}")
        other_teacher = aligned[teacher_cols].to_numpy(dtype=np.float32)
        if not np.allclose(other_teacher, teacher_probs, atol=1e-7):
            raise ValueError(f"Teacher probability columns differ for regime {spec.key}")
        regime_probs[spec.key] = aligned[prob_cols].to_numpy(dtype=np.float32)

    return sample_ids, teacher_probs, regime_probs


def _compute_summary_only(pred_probs: np.ndarray, teacher_probs: np.ndarray, hard_targets: np.ndarray) -> dict[str, float]:
    if pred_probs.shape != teacher_probs.shape or pred_probs.shape != hard_targets.shape:
        raise ValueError("All arrays must have the same shape")

    pred_hard = (pred_probs >= PREDICTION_THRESHOLD).astype(np.int64)
    aucs: list[float] = []
    f1s: list[float] = []

    for label_idx in range(pred_probs.shape[1]):
        y_true = hard_targets[:, label_idx]
        y_prob = pred_probs[:, label_idx]
        y_pred = pred_hard[:, label_idx]
        aucs.append(_binary_auroc(y_true, y_prob))
        _, _, f1 = _precision_recall_f1(y_true, y_pred)
        f1s.append(f1)

    _, _, micro_f1 = _precision_recall_f1(hard_targets.reshape(-1), pred_hard.reshape(-1))
    macro_brier = float(
        np.mean([
            _brier_score(hard_targets[:, i], pred_probs[:, i])
            for i in range(pred_probs.shape[1])
        ])
    )
    macro_ece = float(
        np.nanmean([
            _ece(hard_targets[:, i], pred_probs[:, i], bins=CALIBRATION_BINS)
            for i in range(pred_probs.shape[1])
        ])
    )
    return {
        "macro_auroc": float(np.nanmean(np.asarray(aucs, dtype=np.float64))),
        "micro_auroc": _micro_auroc(hard_targets, pred_probs),
        "macro_f1": float(np.mean(np.asarray(f1s, dtype=np.float64))),
        "micro_f1": micro_f1,
        "soft_mae_p1": float(np.mean(np.abs(pred_probs - teacher_probs))),
        "soft_rmse_p1": float(np.sqrt(np.mean((pred_probs - teacher_probs) ** 2))),
        "macro_brier": macro_brier,
        "micro_brier": _brier_score(hard_targets.reshape(-1), pred_probs.reshape(-1)),
        "macro_ece": macro_ece,
        "micro_ece": _ece(hard_targets.reshape(-1), pred_probs.reshape(-1), bins=CALIBRATION_BINS),
    }


def _format_delta(value: float) -> str:
    return f"{value:+.4f}"


def _format_ci(lo: float, hi: float) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def _render_latex_table(summary_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[p]",
        r"  \centering",
        r"  \caption[Paired bootstrap ETD transfer deltas for the three training regimes]{Paired document-level bootstrap 95\% confidence intervals for \ac{ETD} transfer deltas across the three training regimes. Positive deltas favor the left-hand regime for \ac{AUROC} and F1, whereas negative deltas favor the left-hand regime for soft \ac{MAE} and soft \ac{RMSE}.}",
        r"  \label{tab:ETD_bootstrap_transfer}",
        r"  \small",
        r"  \begin{tabular}{@{} l l r l r @{}}",
        r"    \toprule",
        r"    \textbf{Comparison} & \textbf{Metric} & \textbf{Point $\Delta$} & \textbf{95\% CI} & \textbf{Pr(left better)} \\",
        r"    \midrule",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"    {row.comparison_display} & {row.metric_display} & {row.point_delta_fmt} & {row.ci_fmt} & {row.prob_left_better:.3f}" + r" \\"
        )
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids, teacher_probs, regime_probs = _align_regime_frames(REGIME_SPECS)
    hard_targets = (teacher_probs >= HARD_LABEL_THRESHOLD).astype(np.int64)
    n_samples = len(sample_ids)
    print(f"Loaded {n_samples} ETD test documents across {len(REGIME_SPECS)} regimes")

    point_rows: list[dict[str, object]] = []
    point_metrics: dict[str, dict[str, float]] = {}
    for spec in REGIME_SPECS:
        summary = _compute_summary_only(regime_probs[spec.key], teacher_probs, hard_targets)
        point_metrics[spec.key] = summary
        for metric_key, metric_display, _ in METRIC_SPECS:
            point_rows.append(
                {
                    "regime": spec.key,
                    "regime_display": spec.display_name,
                    "metric": metric_key,
                    "metric_display": metric_display,
                    "value": float(summary[metric_key]),
                }
            )

    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(output_dir / "point_estimates.csv", index=False)

    delta_samples: dict[tuple[str, str, str], np.ndarray] = {}
    for left_key, right_key in COMPARISONS:
        for metric_key, _, _ in METRIC_SPECS:
            delta_samples[(left_key, right_key, metric_key)] = np.empty(args.n_bootstrap, dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    for b in range(args.n_bootstrap):
        if (b + 1) % 100 == 0 or b == 0:
            print(f"Bootstrap replicate {b + 1}/{args.n_bootstrap}")
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        teacher_sub = teacher_probs[sample_idx]
        hard_sub = hard_targets[sample_idx]

        boot_metrics: dict[str, dict[str, float]] = {}
        for spec in REGIME_SPECS:
            boot_metrics[spec.key] = _compute_summary_only(
                regime_probs[spec.key][sample_idx],
                teacher_sub,
                hard_sub,
            )

        for left_key, right_key in COMPARISONS:
            for metric_key, _, _ in METRIC_SPECS:
                delta_samples[(left_key, right_key, metric_key)][b] = (
                    boot_metrics[left_key][metric_key] - boot_metrics[right_key][metric_key]
                )

    rows: list[dict[str, object]] = []
    display_lookup = {spec.key: spec.display_name for spec in REGIME_SPECS}
    for left_key, right_key in COMPARISONS:
        comparison_display = f"{display_lookup[left_key]} - {display_lookup[right_key]}"
        for metric_key, metric_display, higher_is_better in METRIC_SPECS:
            deltas = delta_samples[(left_key, right_key, metric_key)]
            point_delta = point_metrics[left_key][metric_key] - point_metrics[right_key][metric_key]
            ci_low = float(np.quantile(deltas, 0.025))
            ci_high = float(np.quantile(deltas, 0.975))
            prob_left_better = float(np.mean(deltas > 0.0)) if higher_is_better else float(np.mean(deltas < 0.0))
            ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
            rows.append(
                {
                    "left_regime": left_key,
                    "right_regime": right_key,
                    "comparison_display": comparison_display,
                    "metric": metric_key,
                    "metric_display": metric_display,
                    "higher_is_better": higher_is_better,
                    "left_value": float(point_metrics[left_key][metric_key]),
                    "right_value": float(point_metrics[right_key][metric_key]),
                    "point_delta": float(point_delta),
                    "bootstrap_mean_delta": float(np.mean(deltas)),
                    "bootstrap_median_delta": float(np.median(deltas)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "prob_left_better": prob_left_better,
                    "ci_excludes_zero": ci_excludes_zero,
                    "point_delta_fmt": _format_delta(point_delta),
                    "ci_fmt": _format_ci(ci_low, ci_high),
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "pairwise_bootstrap_deltas.csv", index=False)
    (output_dir / "pairwise_bootstrap_deltas.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pairwise_bootstrap_deltas.tex").write_text(
        _render_latex_table(summary_df),
        encoding="utf-8",
    )
    (output_dir / "bootstrap_config.json").write_text(
        json.dumps(
            {
                "n_samples": n_samples,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "prediction_threshold": PREDICTION_THRESHOLD,
                "hard_label_threshold": HARD_LABEL_THRESHOLD,
                "calibration_bins": CALIBRATION_BINS,
                "regimes": [
                    {
                        "key": spec.key,
                        "display_name": spec.display_name,
                        "path": str(spec.path),
                    }
                    for spec in REGIME_SPECS
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    highlights = summary_df[[
        "comparison_display",
        "metric_display",
        "point_delta_fmt",
        "ci_fmt",
        "prob_left_better",
    ]]
    print("\nBootstrap summary:")
    print(highlights.to_string(index=False))
    print(f"\nWrote {output_dir / 'point_estimates.csv'}")
    print(f"Wrote {output_dir / 'pairwise_bootstrap_deltas.csv'}")
    print(f"Wrote {output_dir / 'pairwise_bootstrap_deltas.tex'}")


if __name__ == "__main__":
    main()
