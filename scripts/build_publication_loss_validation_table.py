#!/usr/bin/env python3
"""Build the compact publication-loss validation table used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = ROOT / "outputs/analysis/factorial_ablation_validation_v1/condition_summary.csv"
DEFAULT_EXTENSION = ROOT / "outputs/analysis/factorial_ablation_extension_v1/condition_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/publication_loss_validation_summary"

ROW_SPECS = [
    ("plain_bce_t1_v1", "Plain BCE"),
    ("weighted_bce_t20_v1", r"Weighted BCE ($T=2.0$)"),
    ("masked_bce_t1_tau50_v1", r"Masked BCE ($\tau_c=0.50$)"),
    ("hard_label_v1", "Hard-label BCE"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-summary", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--extension-summary", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_summary(core_path: Path, extension_path: Path) -> pd.DataFrame:
    combined = pd.concat(
        [pd.read_csv(core_path), pd.read_csv(extension_path)],
        ignore_index=True,
    )
    if combined["condition_key"].duplicated().any():
        duplicates = combined.loc[
            combined["condition_key"].duplicated(keep=False), "condition_key"
        ].tolist()
        raise ValueError(f"Duplicate condition summaries: {duplicates}")

    indexed = combined.set_index("condition_key")
    missing = [key for key, _ in ROW_SPECS if key not in indexed.index]
    if missing:
        raise ValueError(f"Missing conditions: {missing}")

    plain_mean = float(indexed.loc["plain_bce_t1_v1", "val_soft_bce_p1_mean"])
    rows = []
    for key, label in ROW_SPECS:
        source = indexed.loc[key]
        mean = float(source["val_soft_bce_p1_mean"])
        rows.append(
            {
                "condition_key": key,
                "regime": label,
                "n_seeds": int(source["n_runs"]),
                "mean_validation_soft_bce": mean,
                "sd_validation_soft_bce": float(source["val_soft_bce_p1_sd"]),
                "delta_vs_plain": mean - plain_mean,
            }
        )
    return pd.DataFrame(rows)


def render_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption[Validation soft-BCE by publication-domain loss variant]{Validation soft-\ac{BCE} against the teacher probability target $p_t$ by publication-domain loss variant. Values are means $\pm$ standard deviations across three training seeds. Positive $\Delta$ indicates lower teacher fidelity relative to plain \ac{BCE}.}",
        r"  \label{tab:publication_loss_validation_summary}",
        r"  \footnotesize",
        r"  \begin{tabular}{@{} l c r @{}}",
        r"    \toprule",
        r"    \textbf{Regime} & \textbf{Validation soft-\ac{BCE}} & \textbf{$\Delta$ vs. plain} \\",
        r"    \midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"    {row.regime} & "
            f"{row.mean_validation_soft_bce:.4f} $\\pm$ {row.sd_validation_soft_bce:.4f} & "
            f"{row.delta_vs_plain:+.4f} \\\\"
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
    summary = build_summary(args.core_summary, args.extension_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "validation_loss_summary_paper.csv"
    tex_path = args.output_dir / "validation_loss_summary_paper.tex"
    summary.to_csv(csv_path, index=False)
    tex_path.write_text(render_latex(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
