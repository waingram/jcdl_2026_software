# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import pandas as pd


@dataclass(frozen=True)
class InputRow:
    row_id: int
    doi: str
    abstract: str


def load_scopus_csv(
    path: Path,
    doi_col: str = "DOI",
    abstract_col: str = "Abstract",
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if doi_col not in df.columns:
        raise ValueError(f"Missing DOI column {doi_col!r}")
    if abstract_col not in df.columns:
        raise ValueError(f"Missing abstract column {abstract_col!r}")

    # DOI: normalize to a clean string; missing becomes "".
    doi = df[doi_col].fillna("").astype(str).str.strip()

    # Guard against pandas/stringification artifacts.
    # If a prior run wrote literal "nan" strings, treat them as missing.
    doi = doi.mask(doi.str.lower().eq("nan"), "")

    df[doi_col] = doi

    # Abstract: keep as-is except NaN -> "" and force to str.
    # Do NOT strip; you want exact comparability across runs.
    abs_series = df[abstract_col]
    abs_series = abs_series.where(~abs_series.isna(), "")
    df[abstract_col] = abs_series.astype(str)

    return df


def iter_batches(
    df: pd.DataFrame,
    batch_size: int,
    doi_col: str = "DOI",
    abstract_col: str = "Abstract",
) -> Iterator[List[InputRow]]:
    n = len(df)

    # Stable row_id: 0..n-1 for the df you actually run.
    for start in range(0, n, batch_size):
        chunk = df.iloc[start : start + batch_size]

        batch: List[InputRow] = []
        # Use itertuples for speed and stable column access.
        # We rely on column order here, so pull the two columns explicitly.
        sub = chunk[[doi_col, abstract_col]]

        for local_i, row in enumerate(sub.itertuples(index=False, name=None)):
            row_id = start + local_i
            doi, abstract = row

            # DOI can be empty; row_id is the join key.
            doi_s = str(doi).strip() if doi is not None else ""
            if doi_s.lower() == "nan":
                doi_s = ""

            abstract_s = str(abstract) if abstract is not None else ""
            if abstract_s.lower() == "nan":
                # Only possible if someone serialized NaN as a string.
                abstract_s = ""

            batch.append(InputRow(row_id=row_id, doi=doi_s, abstract=abstract_s))

        yield batch