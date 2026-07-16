#!/usr/bin/env python3
"""Build a manuscript figure for prompt-variant flip categories and margins."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sdg_shared.plot_style import (
        GRID_COLOR,
        NEGATIVE_DELTA_COLOR,
        P1_COLOR,
        PLAIN_COLOR,
        TEACHER_COLOR,
        apply_manuscript_theme,
        finish_axis,
    )
except Exception:
    GRID_COLOR = "#D5DEE5"
    P1_COLOR = "#2E5C8A"
    TEACHER_COLOR = "#0B7A75"
    PLAIN_COLOR = "#8A94A3"
    NEGATIVE_DELTA_COLOR = "#B24633"

    def apply_manuscript_theme(*, context: str = "paper") -> None:
        _ = context
        mpl.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "mathtext.fontset": "stix",
                "text.color": "#1F2933",
                "axes.labelcolor": "#1F2933",
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


DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "analysis"
    / "prompt_format"
    / "flip_overlap__merged_ABC_with_logits.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "figures" / "flip_margin_v1"
DEFAULT_MANUSCRIPT_DIR = ROOT / "outputs" / "figures" / "flip_margin_v1"
STEM = "flip_category_abs_margin_sdg1"


CATEGORY_SPECS = [
    {
        "key": "no_flip",
        "name": "No flip",
        "math": r"$A=B=C$",
        "mask": lambda df: (~df["flip_AB"]) & (~df["flip_BC"]),
        "color": PLAIN_COLOR,
    },
    {
        "key": "ab_only",
        "name": "AB only",
        "math": r"$A\ne B,\ B=C$",
        "mask": lambda df: df["flip_AB"] & (~df["flip_BC"]),
        "color": P1_COLOR,
    },
    {
        "key": "bc_only",
        "name": "BC only",
        "math": r"$A=B,\ B\ne C$",
        "mask": lambda df: (~df["flip_AB"]) & df["flip_BC"],
        "color": NEGATIVE_DELTA_COLOR,
    },
    {
        "key": "double_flip",
        "name": "Double flip",
        "math": r"$A\ne B,\ B\ne C$",
        "mask": lambda df: df["flip_AB"] & df["flip_BC"],
        "color": TEACHER_COLOR,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manuscript-dir", type=Path, default=DEFAULT_MANUSCRIPT_DIR)
    return parser.parse_args()


def _bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False})


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "flip_AB", "flip_BC", "abs_margin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    flip_ab = _bool_series(df["flip_AB"])
    flip_bc = _bool_series(df["flip_BC"])
    if flip_ab.isna().any() or flip_bc.isna().any():
        raise ValueError("flip_AB and flip_BC must contain boolean values")

    return df.assign(
        flip_AB=flip_ab,
        flip_BC=flip_bc,
        abs_margin=pd.to_numeric(df["abs_margin"], errors="raise"),
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_category_table(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for spec in CATEGORY_SPECS:
        mask = spec["mask"](df)
        margins = df.loc[mask, "abs_margin"].astype(float)
        rows.append(
            {
                "key": spec["key"],
                "name": spec["name"],
                "math": spec["math"],
                "color": spec["color"],
                "n": int(mask.sum()),
                "pct_total": float(mask.sum() / total),
                "median": float(margins.median()),
                "q25": float(margins.quantile(0.25)),
                "q75": float(margins.quantile(0.75)),
                "values": margins.to_numpy(),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, *dirs: Path) -> None:
    for out_dir in dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{STEM}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{STEM}.png", bbox_inches="tight", dpi=300)


def write_tex_wrapper(out_dir: Path) -> None:
    tex = r"""\begin{figure}[t]
  \centering
  \includegraphics[width=0.94\linewidth]{figures/generated/main/flip_category_abs_margin_sdg1.pdf}
  \caption[Teacher margin distributions by prompt-variant flip category]{Absolute teacher logit margin $|z_t|$ by prompt-variant flip category for the SDG~1 consistency analysis. Categories are defined by whether the hard label changes from the justification variant ($A$) to the binary-label variant ($B$), and from the binary-label variant ($B$) to the binary-bit variant ($C$). The margin is taken from the concurrent \texttt{binary\_bit\_with\_probs} diagnostic run for the same abstract. Violin shapes show the within-category distribution of $|z_t|$; dots mark medians; thick horizontal intervals show the interquartile range. Category labels report the number and percentage of abstracts in each category.}
  \label{fig:flip-category-abs-margin-sdg1}
\end{figure}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{STEM}.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = prepare_data(_read_table(args.input))
    table = build_category_table(df)

    apply_manuscript_theme(context="paper")
    fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=False)

    y_positions = np.arange(len(table))[::-1]
    values = table["values"].tolist()
    colors = table["color"].tolist()

    parts = ax.violinplot(
        values,
        positions=y_positions,
        vert=False,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=200,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.34)

    for y, row in zip(y_positions, table.itertuples(index=False)):
        ax.plot(
            [row.q25, row.q75],
            [y, y],
            color="#1F2933",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            [row.median],
            [y],
            s=58,
            color=row.color,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            row.median + 0.28,
            y,
            f"median {row.median:.2f}",
            ha="left",
            va="center",
            fontsize=9.4,
            fontweight="semibold",
            color="#1F2933",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            zorder=5,
        )

    labels = [
        f"{row.name}\n{row.math}\nn={int(row.n):,}; {float(row.pct_total) * 100.0:.1f}%"
        for row in table.itertuples(index=False)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Absolute teacher margin $|z_t|$")
    ax.set_xlim(0, 17.2)
    ax.set_ylim(-0.65, len(table) - 0.35)
    ax.text(
        0.01,
        0.98,
        "Dot = median; thick line = IQR; violin = within-category distribution",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#4B5563",
    )
    finish_axis(ax, grid_axis="x")
    fig.tight_layout()

    save_figure(fig, args.output_dir, args.manuscript_dir)
    plt.close(fig)
    write_tex_wrapper(args.manuscript_dir)

    summary = table[["name", "n", "pct_total", "median", "q25", "q75"]].copy()
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_dir / (STEM + '.pdf')}")
    print(f"Wrote {args.manuscript_dir / (STEM + '.tex')}")


if __name__ == "__main__":
    main()
