"""Shared plotting style helpers for manuscript-quality figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


TEXT_COLOR = "#1F2933"
SPINE_COLOR = "#2F3A44"
GRID_COLOR = "#D5DEE5"

P1_COLOR = "#2E5C8A"
TEACHER_COLOR = "#0B7A75"
WEIGHTED_COLOR = "#0B7A75"
MASKED_COLOR = "#C58B39"
PLAIN_COLOR = "#8A94A3"
BEST_CHECKPOINT_COLOR = "#0B7A75"
OPERATIONAL_COLOR = "#B95C27"
NEGATIVE_DELTA_COLOR = "#B24633"
POSITIVE_DELTA_COLOR = "#2E7D5B"
VAL_COLOR = "#2E5C8A"
TEST_COLOR = "#0B7A75"
RUNTIME_COLOR = "#C58B39"

BASE_PALETTE = [
    TEACHER_COLOR,
    P1_COLOR,
    RUNTIME_COLOR,
    "#785D9A",
    "#7A8B49",
    NEGATIVE_DELTA_COLOR,
]

LOSS_MODE_PALETTE = {
    "plain_bce": PLAIN_COLOR,
    "masked_bce": MASKED_COLOR,
    "weighted_bce": WEIGHTED_COLOR,
}

TARGET_MODE_PALETTE = {
    "p1": P1_COLOR,
    "teacher_logit": TEACHER_COLOR,
}

CHECKPOINT_PALETTE = {
    "best_checkpoint": BEST_CHECKPOINT_COLOR,
    "operational_recipe": OPERATIONAL_COLOR,
}

TEMPERATURE_PALETTE = {
    1.0: PLAIN_COLOR,
    1.25: TEACHER_COLOR,
    1.5: RUNTIME_COLOR,
}

THRESHOLD_PALETTE = {
    0.05: P1_COLOR,
    0.1: TEACHER_COLOR,
}

SPLIT_METRIC_PALETTE = {
    "val_loss": VAL_COLOR,
    "test_loss": TEST_COLOR,
}


def apply_manuscript_theme(*, context: str = "paper") -> None:
    """Apply a consistent manuscript-first theme.

    We intentionally re-apply rc settings after `sns.set_theme`, because seaborn
    overwrites several rcParams loaded from `matplotlibrc`.
    """

    sns.set_theme(context=context, style="ticks", palette=BASE_PALETTE)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": SPINE_COLOR,
            "axes.facecolor": "white",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.55,
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 9.5,
            "lines.linewidth": 1.8,
            "scatter.marker": "o",
            "figure.figsize": (8.6, 5.4),
            "figure.dpi": 200,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    """Apply common axis cleanup."""
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, alpha=0.55, linewidth=0.7)


def finish_legend(ax: plt.Axes, *, title: str | None = None) -> None:
    """Normalize legend styling when a legend exists."""
    legend = ax.get_legend()
    if legend is None:
        return
    legend.set_title(title)
    legend.get_frame().set_linewidth(0.0)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save one figure as PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"{stem}{suffix}", bbox_inches="tight")
    plt.close(fig)
