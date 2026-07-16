from __future__ import annotations

import pandas as pd

from sdg_distillation.training.splits import build_teacher_confidence_strata, split_train_val_df


def _make_df(n: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        row: dict[str, object] = {
            "sample_id": f"s{i}",
            "retrieval_sdg": (i % 3) + 1,
        }
        for j in range(1, 18):
            if i % 10 == 0:
                value = 0.51 if j % 2 == 0 else 0.49
            elif i % 3 == 0:
                value = 0.9 if j % 2 == 0 else 0.1
            elif i % 3 == 1:
                value = 0.7 if j % 2 == 0 else 0.3
            else:
                value = 0.97 if j % 2 == 0 else 0.03
            row[f"p1_sdg{j:02d}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_teacher_confidence_strata_contains_expected_columns() -> None:
    df = _make_df(24)
    strata = build_teacher_confidence_strata(df)
    assert len(strata) == len(df)
    assert {"sample_id", "stratum", "positive_cardinality", "boundary_count", "mean_confidence"}.issubset(strata.columns)
    assert (strata["boundary_count"] > 0).any()


def test_split_train_val_df_is_deterministic_and_disjoint() -> None:
    df = _make_df(90)
    train_a, val_a = split_train_val_df(df, val_fraction=0.1, seed=17)
    train_b, val_b = split_train_val_df(df, val_fraction=0.1, seed=17)

    assert train_a["sample_id"].tolist() == train_b["sample_id"].tolist()
    assert val_a["sample_id"].tolist() == val_b["sample_id"].tolist()
    assert set(train_a["sample_id"]).isdisjoint(set(val_a["sample_id"]))
    assert len(train_a) + len(val_a) == len(df)
    assert len(val_a) > 0
