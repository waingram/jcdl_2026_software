from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from .data_contract import ensure_columns_present, sdg_target_columns


def _quantile_labels(n_bins: int) -> list[str]:
    return [f"q{i}" for i in range(1, n_bins + 1)]


def _mean_confidence_bins(mean_confidence: np.ndarray, *, max_bins: int = 4) -> pd.Series:
    series = pd.Series(mean_confidence, copy=False)
    n_unique = int(series.nunique(dropna=True))
    n_bins = max(1, min(max_bins, n_unique))
    if n_bins == 1:
        return pd.Series(["q1"] * len(series), index=series.index, dtype="object")

    codes, bins = pd.qcut(series, q=n_bins, labels=False, retbins=True, duplicates="drop")
    actual_bins = max(1, len(bins) - 1)
    labels = _quantile_labels(actual_bins)
    if actual_bins == 1:
        return pd.Series([labels[0]] * len(series), index=series.index, dtype="object")

    codes_series = pd.Series(codes, index=series.index)
    return codes_series.map(lambda x: labels[int(x)]).astype("object")


def build_teacher_confidence_strata(
    df: pd.DataFrame,
    *,
    boundary_eps: float = 0.05,
) -> pd.DataFrame:
    if boundary_eps <= 0 or boundary_eps >= 0.5:
        raise ValueError("boundary_eps must lie in (0, 0.5).")

    p1_cols = sdg_target_columns("p1")
    ensure_columns_present(df, ["sample_id", "retrieval_sdg", *p1_cols])

    df = df.reset_index(drop=True).copy()
    probs = df[p1_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    confidence = 2.0 * np.abs(probs - 0.5)
    positive_cardinality = (probs >= 0.5).sum(axis=1)
    boundary_count = (np.abs(probs - 0.5) < float(boundary_eps)).sum(axis=1)
    mean_confidence = confidence.mean(axis=1)

    pos_bin = pd.cut(
        positive_cardinality,
        bins=[-1, 0, 1, 2, 4, len(p1_cols)],
        labels=["0", "1", "2", "3-4", "5+"],
    ).astype(str)
    boundary_bin = pd.cut(
        boundary_count,
        bins=[-1, 0, 1, len(p1_cols)],
        labels=["0", "1", "2+"],
    ).astype(str)
    mean_conf_bin = _mean_confidence_bins(mean_confidence)

    retrieval = df["retrieval_sdg"].astype(str)
    stratum = retrieval + "__" + pos_bin + "__" + boundary_bin + "__" + mean_conf_bin

    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "retrieval_sdg": retrieval.to_numpy(),
            "positive_cardinality": positive_cardinality,
            "positive_cardinality_bin": pos_bin.to_numpy(),
            "boundary_count": boundary_count,
            "boundary_count_bin": boundary_bin.to_numpy(),
            "mean_confidence": mean_confidence,
            "mean_confidence_bin": mean_conf_bin.to_numpy(),
            "stratum": stratum.to_numpy(),
        }
    )


def split_train_val_df(
    df: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
    boundary_eps: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_fraction <= 0:
        return df, df.iloc[0:0].copy()
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in [0,1).")
    if len(df) < 2:
        return df, df.iloc[0:0].copy()

    strata = build_teacher_confidence_strata(df, boundary_eps=boundary_eps)
    grouped = strata.groupby("stratum", sort=False).indices

    rng = random.Random(seed)
    val_idx: set[int] = set()

    for indices in grouped.values():
        group_idx = list(indices)
        n_group = len(group_idx)
        if n_group < 2:
            continue
        rng.shuffle(group_idx)
        n_val = min(n_group - 1, max(1, int(round(n_group * val_fraction))))
        val_idx.update(group_idx[:n_val])

    is_val = df.index.isin(sorted(val_idx))
    train_df = df.loc[~is_val].copy()
    val_df = df.loc[is_val].copy()
    return train_df, val_df


def split_summary(df: pd.DataFrame, *, boundary_eps: float = 0.05) -> dict[str, Any]:
    strata = build_teacher_confidence_strata(df, boundary_eps=boundary_eps)
    return {
        "rows": int(len(df)),
        "retrieval_sdg_counts": {
            str(k): int(v) for k, v in strata["retrieval_sdg"].value_counts().sort_index().items()
        },
        "positive_cardinality_mean": float(strata["positive_cardinality"].mean()),
        "boundary_rate_gt0": float((strata["boundary_count"] > 0).mean()),
        "boundary_rate_ge2": float((strata["boundary_count"] >= 2).mean()),
        "mean_confidence_mean": float(strata["mean_confidence"].mean()),
        "strata_count": int(strata["stratum"].nunique()),
    }
