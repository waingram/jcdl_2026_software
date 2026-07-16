from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

TargetMode = Literal["p1", "teacher_logit", "hard_label", "both"]

ID_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "retrieval_sdg",
    "split",
    "row_id",
)

TEXT_COLUMN = "text"


def sdg_target_columns(prefix: Literal["p1", "teacher_logit", "hard_label"]) -> list[str]:
    return [f"{prefix}_sdg{i:02d}" for i in range(1, 18)]


def target_columns_for_mode(target_mode: TargetMode) -> list[str]:
    if target_mode == "p1":
        return sdg_target_columns("p1")
    if target_mode == "teacher_logit":
        return sdg_target_columns("teacher_logit")
    if target_mode == "hard_label":
        return sdg_target_columns("hard_label")
    return sdg_target_columns("p1") + sdg_target_columns("teacher_logit")


def required_columns(target_mode: TargetMode) -> list[str]:
    return [*ID_COLUMNS, TEXT_COLUMN, *target_columns_for_mode(target_mode)]


def validate_student_ready_df(
    df: pd.DataFrame,
    target_mode: TargetMode = "p1",
    *,
    require_unique_sample_id: bool = True,
    require_finite_targets: bool = True,
    expected_split: str | None = None,
) -> None:
    missing = [c for c in required_columns(target_mode) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if require_unique_sample_id and df["sample_id"].duplicated().any():
        n_dup = int(df["sample_id"].duplicated().sum())
        raise ValueError(f"sample_id must be unique; found duplicated rows: {n_dup}")

    if expected_split is not None:
        bad_split = ~df["split"].astype(str).eq(expected_split)
        if bad_split.any():
            vals = sorted(df.loc[bad_split, "split"].astype(str).unique().tolist())
            raise ValueError(
                f"Expected split={expected_split!r}, found additional split values: {vals}"
            )

    # Enforce text as non-null string-like content.
    if df[TEXT_COLUMN].isna().any():
        n_na = int(df[TEXT_COLUMN].isna().sum())
        raise ValueError(f"{TEXT_COLUMN!r} contains nulls: {n_na}")

    # Target checks.
    target_cols = target_columns_for_mode(target_mode)
    if require_finite_targets:
        bad_any = ~df[target_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        if bad_any.any():
            n_bad = int(bad_any.sum())
            raise ValueError(f"Found rows with non-finite targets: {n_bad}")

    if target_mode in {"p1", "both"}:
        p1_cols = sdg_target_columns("p1")
        p1 = df[p1_cols].apply(pd.to_numeric, errors="coerce")
        oob = (p1 < 0.0) | (p1 > 1.0)
        if oob.any().any():
            n_oob = int(oob.any(axis=1).sum())
            raise ValueError(f"p1 target values out of [0,1] in rows: {n_oob}")

    if target_mode == "hard_label":
        hard_cols = sdg_target_columns("hard_label")
        hard = df[hard_cols].apply(pd.to_numeric, errors="coerce")
        valid = (hard == 0.0) | (hard == 1.0)
        if (~valid).any().any():
            n_bad = int((~valid).any(axis=1).sum())
            raise ValueError(f"hard_label target values must be binary in rows: {n_bad}")


def ensure_columns_present(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
