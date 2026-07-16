#!/usr/bin/env python3
"""Build teacher-logit distribution figures and a supplemental table for Scopus teacher outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sdg_shared.plot_style import (
        GRID_COLOR,
        NEGATIVE_DELTA_COLOR,
        PLAIN_COLOR,
        TEACHER_COLOR,
        TEXT_COLOR,
        apply_manuscript_theme,
        finish_axis,
        save_figure,
    )
except Exception:
    GRID_COLOR = "#D5DEE5"
    TEXT_COLOR = "#1F2933"
    TEACHER_COLOR = "#0B7A75"
    PLAIN_COLOR = "#8A94A3"
    NEGATIVE_DELTA_COLOR = "#B24633"

    def apply_manuscript_theme(*, context: str = "paper") -> None:
        _ = context
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "mathtext.fontset": "stix",
                "text.color": TEXT_COLOR,
                "axes.labelcolor": TEXT_COLOR,
                "axes.edgecolor": "#2F3A44",
                "axes.facecolor": "white",
                "axes.linewidth": 0.9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titlesize": 14,
                "axes.titleweight": "semibold",
                "axes.labelsize": 11.5,
                "xtick.labelsize": 9.5,
                "ytick.labelsize": 9.5,
                "grid.color": GRID_COLOR,
                "grid.alpha": 0.55,
                "grid.linewidth": 0.7,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 300,
            }
        )

    def finish_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
        ax.set_axisbelow(True)
        ax.grid(False)
        if grid_axis is not None:
            ax.grid(True, axis=grid_axis, color=GRID_COLOR, alpha=0.55, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            fig.savefig(output_dir / f"{stem}{suffix}", bbox_inches="tight")
        plt.close(fig)


DEFAULT_TEACHER_PATHS = [
    ROOT / "outputs/teacher/publication/teacher_grid_wide_train.csv",
    ROOT / "outputs/teacher/publication/teacher_grid_wide_test.csv",
]
DEFAULT_SUMMARY = ROOT / "outputs/analysis/teacher_signal/teacher_logit_distribution_by_sdg.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/figures/teacher_signal"
DEFAULT_MANUSCRIPT_TABLE_DIR = ROOT / "outputs/tables/teacher_signal"
STEM = "teacher_logit_distribution_figure"
TABLE_STEM = "teacher_logit_distribution_summary"
LATEX_SHORT_CAPTION = "Per-SDG summary of teacher logit distributions, publication domain"
LATEX_CAPTION = (
    "Supplemental per-SDG summary for the publication-domain teacher-logit distributions. "
    "Rows follow the same ordering as Figure~\\ref{fig:teacher-logit-distribution}: descending positive-margin "
    "rate, with ties broken by median absolute margin."
)
LATEX_LABEL = "tab:teacher-logit-distribution-summary"

SDG_SHORT = {
    "sdg01": "Poverty",
    "sdg02": "Hunger",
    "sdg03": "Health",
    "sdg04": "Education",
    "sdg05": "Gender",
    "sdg06": "Water",
    "sdg07": "Energy",
    "sdg08": "Work",
    "sdg09": "Industry",
    "sdg10": "Inequality",
    "sdg11": "Cities",
    "sdg12": "Consumption",
    "sdg13": "Climate",
    "sdg14": "Oceans",
    "sdg15": "Land",
    "sdg16": "Institutions",
    "sdg17": "Partnerships",
}

FIGURE_PROFILES = {
    "dissertation": {
        "stem": STEM,
        "figsize": (12.8, 7.8),
        "width_ratios": [6.3, 1.8],
        "wspace": 0.06,
        "violin_width": 0.82,
        "bar_height": 0.62,
        "show_mean": True,
        "show_explanatory_note": True,
        "show_boundary_band": False,
        "boundary_band": (-1.0, 1.0),
        "show_boundary_note": False,
        "xlim": (-20.0, 18.0),
        "xticks": [-20, -15, -10, -5, 0, 5, 10, 15],
        "label_mode": "long",
        "ylabel_size": 9.5,
        "xlabel_size": 11.5,
        "bar_label_size": 9.0,
        "median_size": 46,
        "mean_size": 20,
        "iqr_linewidth": 4.0,
        "boundary_alpha": 0.05,
        "show_boundary_annotation": False,
        "boundary_annotation_size": 9.0,
    },
    "acm": {
        "stem": f"{STEM}_acm",
        "figsize": (7.1, 5.0),
        "width_ratios": [5.4, 1.9],
        "wspace": 0.07,
        "violin_width": 0.76,
        "bar_height": 0.56,
        "show_mean": False,
        "show_explanatory_note": False,
        "show_boundary_band": True,
        "boundary_band": (-1.0, 1.0),
        "show_boundary_note": False,
        "xlim": (-18.0, 16.0),
        "xticks": [-15, -10, -5, 0, 5, 10, 15],
        "label_mode": "compact",
        "show_boundary_annotation": True,
        "ylabel_size": 8.2,
        "xlabel_size": 10.0,
        "bar_label_size": 7.8,
        "median_size": 28,
        "mean_size": 0,
        "iqr_linewidth": 3.0,
        "boundary_alpha": 0.08,
        "boundary_annotation_size": 7.6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-paths",
        type=Path,
        nargs="+",
        default=DEFAULT_TEACHER_PATHS,
        help="One or more teacher-wide CSV/parquet files with teacher_logit_sdgXX columns.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Per-SDG teacher summary CSV produced by analyze_sdg_logit_distributions.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output figure.",
    )
    parser.add_argument(
        "--max-points-per-sdg",
        type=int,
        default=40000,
        help="Maximum number of values to draw per SDG in the violins.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for violin downsampling.",
    )
    parser.add_argument(
        "--manuscript-table-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_TABLE_DIR,
        help="Directory for the generated manuscript table CSV/TEX.",
    )
    parser.add_argument(
        "--figure-profile",
        choices=sorted(FIGURE_PROFILES.keys()),
        default="dissertation",
        help="Visual profile for the output figure.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Optional output stem override.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type for {path}")


def _load_teacher(paths: list[Path]) -> pd.DataFrame:
    frames = [_read_table(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def _label_text(label: str) -> str:
    return f"{label.replace('sdg', 'SDG ')}  {SDG_SHORT[label]}"


def _compact_label_text(label: str) -> str:
    return f"SDG {int(label.replace('sdg', '')):02d} {SDG_SHORT[label]}"


def _sample(values: np.ndarray, *, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= max_points:
        return values
    idx = rng.choice(values.size, size=max_points, replace=False)
    return values[idx]


def build_table_df(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.loc[
        :,
        [
            "label",
            "median_abs",
            "median",
            "positive_rate",
            "near_boundary_abs_lt_0_5",
            "near_boundary_abs_lt_1_0",
            "near_boundary_abs_lt_2_0",
        ],
    ].copy()
    table["sdg"] = table["label"].map(_label_text)
    for col in (
        "positive_rate",
        "near_boundary_abs_lt_0_5",
        "near_boundary_abs_lt_1_0",
        "near_boundary_abs_lt_2_0",
    ):
        table[col] = 100.0 * table[col].astype(float)
    table = table.rename(
        columns={
            "median_abs": "median_abs_z_t",
            "median": "median_z_t",
            "positive_rate": "positive_margin_pct",
            "near_boundary_abs_lt_0_5": "near_boundary_lt_0_5_pct",
            "near_boundary_abs_lt_1_0": "near_boundary_lt_1_0_pct",
            "near_boundary_abs_lt_2_0": "near_boundary_lt_2_0_pct",
        }
    )
    return table[
        [
            "sdg",
            "median_abs_z_t",
            "positive_margin_pct",
            "median_z_t",
            "near_boundary_lt_0_5_pct",
            "near_boundary_lt_1_0_pct",
            "near_boundary_lt_2_0_pct",
        ]
    ]


def render_latex_table(table_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[p]",
        r"  \centering",
        f"  \\caption[{LATEX_SHORT_CAPTION}]{{{LATEX_CAPTION}}}",
        f"  \\label{{{LATEX_LABEL}}}",
        r"  \scriptsize",
        r"  \begin{tabular}{@{} l r r r r r r @{}}",
        r"    \toprule",
        r"    \textbf{SDG} & \textbf{Median $|z_t|$} & \textbf{$z_t > 0$ (\%)} & \textbf{Median $z_t$} & \textbf{$|z_t| < 0.5$ (\%)} & \textbf{$|z_t| < 1.0$ (\%)} & \textbf{$|z_t| < 2.0$ (\%)} \\",
        r"    \midrule",
    ]
    for row in table_df.itertuples(index=False):
        lines.append(
            "    "
            + " & ".join(
                [
                    str(row.sdg),
                    f"{float(row.median_abs_z_t):.2f}",
                    f"{float(row.positive_margin_pct):.1f}",
                    f"{float(row.median_z_t):.2f}",
                    f"{float(row.near_boundary_lt_0_5_pct):.1f}",
                    f"{float(row.near_boundary_lt_1_0_pct):.1f}",
                    f"{float(row.near_boundary_lt_2_0_pct):.1f}",
                ]
            )
            + r" \\",
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_table_outputs(table_df: pd.DataFrame, *, output_dir: Path, manuscript_table_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{TABLE_STEM}.csv"
    tex_path = output_dir / f"{TABLE_STEM}.tex"
    table_df.to_csv(csv_path, index=False)
    tex_path.write_text(render_latex_table(table_df), encoding="utf-8")

    manuscript_table_dir.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(manuscript_table_dir / f"{TABLE_STEM}.csv", index=False)
    (manuscript_table_dir / f"{TABLE_STEM}.tex").write_text(render_latex_table(table_df), encoding="utf-8")


def _apply_profile_rc(profile_name: str) -> None:
    apply_manuscript_theme(context="paper")
    if profile_name == "acm":
        plt.rcParams.update(
            {
                "axes.labelsize": 10.0,
                "xtick.labelsize": 8.2,
                "ytick.labelsize": 8.0,
                "grid.linewidth": 0.65,
                "axes.linewidth": 0.85,
            }
        )


def _labels_for_order(order: list[str], label_mode: str) -> list[str]:
    if label_mode == "compact":
        return [_compact_label_text(label) for label in order]
    return [_label_text(label) for label in order]


def build_figure(
    *,
    table: pd.DataFrame,
    values: list[np.ndarray],
    output_dir: Path,
    stem: str,
    profile_name: str,
) -> None:
    profile = FIGURE_PROFILES[profile_name]
    _apply_profile_rc(profile_name)

    fig, (ax, ax_rate) = plt.subplots(
        ncols=2,
        figsize=profile["figsize"],
        gridspec_kw={"width_ratios": profile["width_ratios"], "wspace": profile["wspace"]},
        constrained_layout=False,
        sharey=True,
    )

    y = np.arange(len(table))[::-1]

    if profile["show_boundary_band"]:
        lo, hi = profile["boundary_band"]
        ax.axvspan(lo, hi, color=NEGATIVE_DELTA_COLOR, alpha=profile["boundary_alpha"], zorder=0)

    parts = ax.violinplot(
        values,
        positions=y,
        vert=False,
        widths=profile["violin_width"],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=200,
    )
    for body in parts["bodies"]:
        body.set_facecolor(TEACHER_COLOR)
        body.set_edgecolor("none")
        body.set_alpha(0.34)

    ax.axvline(0.0, color=NEGATIVE_DELTA_COLOR, linewidth=1.1, linestyle=(0, (3, 2)), zorder=1)
    if profile.get("show_boundary_annotation", False):
        ax.annotate(
            r"decision boundary ($z_t = 0$)",
            xy=(0.0, 0.985),
            xycoords=ax.get_xaxis_transform(),
            xytext=(6, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=profile["boundary_annotation_size"],
            color=NEGATIVE_DELTA_COLOR,
        )

    for yi, row in zip(y, table.itertuples(index=False)):
        ax.plot(
            [row.p25, row.p75],
            [yi, yi],
            color=TEXT_COLOR,
            linewidth=profile["iqr_linewidth"],
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            [row.median],
            [yi],
            s=profile["median_size"],
            color=TEACHER_COLOR,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        if profile["show_mean"]:
            ax.scatter(
                [row.mean],
                [yi],
                s=profile["mean_size"],
                color=PLAIN_COLOR,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )

    labels = _labels_for_order(table["label"].tolist(), profile["label_mode"])
    ax.set_yticks(y, labels=labels)
    ax.tick_params(axis="y", labelsize=profile["ylabel_size"])
    ax.set_xlabel(r"Teacher logit margin $z_t$", fontsize=profile["xlabel_size"])
    ax.set_ylabel(None)
    ax.set_xlim(*profile["xlim"])
    ax.set_xticks(profile["xticks"])
    finish_axis(ax, grid_axis="x")

    if profile["show_explanatory_note"]:
        ax.text(
            0.01,
            1.02,
            "Violin = sampled distribution; thick line = IQR; dot = median; dashed line = decision boundary",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.3,
            color=PLAIN_COLOR,
        )
    if profile["show_boundary_note"]:
        lo, hi = profile["boundary_band"]
        ax.text(
            0.99,
            1.01,
            rf"shaded band = $|z_t| < {abs(hi):.0f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.8,
            color=PLAIN_COLOR,
        )

    rates = (100.0 * table["positive_rate"]).to_numpy(dtype=float)
    ax_rate.barh(y, rates, color=TEACHER_COLOR, alpha=0.84, height=profile["bar_height"])
    ax_rate.set_xlim(0.0, max(42.0, float(rates.max()) + 4.0))
    ax_rate.set_xlabel(r"$z_t > 0$ (%)", fontsize=profile["xlabel_size"])
    ax_rate.tick_params(axis="y", which="both", left=False, labelleft=False)
    finish_axis(ax_rate, grid_axis="x")

    for yi, rate in zip(y, rates):
        ax_rate.text(
            rate + 0.6,
            yi,
            f"{rate:.1f}",
            ha="left",
            va="center",
            fontsize=profile["bar_label_size"],
            color=TEXT_COLOR,
        )

    save_figure(fig, output_dir, stem)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_path).sort_values(
        ["positive_rate", "median_abs"], ascending=[False, False], kind="stable"
    )
    teacher = _load_teacher(list(args.teacher_paths))

    rng = np.random.default_rng(args.seed)
    order = summary["label"].tolist()
    values = []
    for label in order:
        col = f"teacher_logit_{label}"
        arr = teacher[col].to_numpy(dtype=np.float64)
        values.append(_sample(arr, max_points=int(args.max_points_per_sdg), rng=rng))

    table = summary.set_index("label").loc[order].reset_index()
    supplement_table = build_table_df(table)
    stem = args.stem or str(FIGURE_PROFILES[args.figure_profile]["stem"])

    build_figure(
        table=table,
        values=values,
        output_dir=args.output_dir,
        stem=stem,
        profile_name=args.figure_profile,
    )
    write_table_outputs(
        supplement_table,
        output_dir=args.output_dir,
        manuscript_table_dir=args.manuscript_table_dir,
    )


if __name__ == "__main__":
    main()
