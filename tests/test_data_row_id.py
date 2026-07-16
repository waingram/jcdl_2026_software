# tests/test_data_row_id.py
from __future__ import annotations

import pandas as pd

from sdg_relevance_labeling.data import iter_batches


def test_iter_batches_row_id_is_stable_and_contiguous():
    df = pd.DataFrame(
        {
            "DOI": [f"10.x/{i}" for i in range(23)],
            "Abstract": [f"abs {i}" for i in range(23)],
        }
    )

    seen = []
    for batch in iter_batches(df, batch_size=8, doi_col="DOI", abstract_col="Abstract"):
        for r in batch:
            seen.append((r.row_id, r.doi, r.abstract))

    # row_id should be 0..n-1 in order, matching df row order.
    assert [x[0] for x in seen] == list(range(len(df)))
    assert [x[1] for x in seen] == df["DOI"].tolist()
    assert [x[2] for x in seen] == df["Abstract"].tolist()


def test_iter_batches_allows_missing_doi():
    df = pd.DataFrame(
        {
            "DOI": ["10.x/0", None, "", "10.x/3"],
            "Abstract": ["a0", "a1", "a2", None],
        }
    )

    rows = []
    for batch in iter_batches(df, batch_size=2, doi_col="DOI", abstract_col="Abstract"):
        rows.extend(batch)

    assert len(rows) == 4
    assert rows[1].doi == ""  # None -> "" at read stage or str(None).strip(), depends on upstream cleaning.
    assert rows[3].abstract == ""  # None abstract -> ""