#!/usr/bin/env python3

# Description: Combine per-retrieval/per-eval teacher CSV outputs into
# long and wide teacher-grid datasets, while checking SDG coverage and
# row alignment across evaluation runs.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/combine_teacher_grid.py [--data-dir DIR --run-tag NAME --year YYYY --out-dir DIR --splits SPLIT [SPLIT ...] --retrieval-sdgs N [N ...] --strict]

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLS = {
    "row_id",
    "DOI",
    "SDG",
    "generated_text",
    "parsed_label",
    "logit_0",
    "logit_1",
    "teacher_logit",
    "p1",
}

LABEL_TO_BIT = {
    "Relevant": 1,
    "Non-Relevant": 0,
}


@dataclass(frozen=True)
class FileSpec:
    """Filename-derived metadata for one teacher output CSV."""

    path: Path
    retrieval_sdg: int
    eval_sdg: int
    split: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for teacher-grid combination and validation."""

    p = argparse.ArgumentParser(
        description=(
            "Combine SDG teacher CSV grid files into long and wide outputs. "
            "Expected filename pattern: sdg{r}xsdg{e}_{year}_{split}__{run_tag}.csv"
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("outputs/teacher/publication"))
    p.add_argument(
        "--run-tag",
        type=str,
        default="scopus_sdg1_qwen_binary_bit_with_probs_v1",
        help="Suffix in filenames after the '__' separator (without .csv).",
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/teacher/publication"))
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to include (default: train test).",
    )
    p.add_argument(
        "--retrieval-sdgs",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of retrieval SDGs to include (e.g., --retrieval-sdgs 1 2 3).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if any (retrieval_sdg, split) group does not contain exactly eval SDGs 1..17 "
            "or has inconsistent row_id coverage across eval SDGs."
        ),
    )
    return p.parse_args()


def discover_files(
    data_dir: Path,
    run_tag: str,
    year: int,
    splits: Iterable[str],
    retrieval_filter: set[int] | None,
) -> list[FileSpec]:
    """Discover teacher CSVs matching the expected filename schema."""

    splits_set = {s.strip().lower() for s in splits}
    if not splits_set:
        raise ValueError("No splits were provided.")

    rx = re.compile(
        rf"^sdg(\d+)xsdg(\d+)_{year}_(\w+)__{re.escape(run_tag)}\.csv$"
    )

    out: list[FileSpec] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        m = rx.match(p.name)
        if not m:
            continue

        r = int(m.group(1))
        e = int(m.group(2))
        split = m.group(3).lower()
        if split not in splits_set:
            continue
        if retrieval_filter is not None and r not in retrieval_filter:
            continue

        out.append(FileSpec(path=p, retrieval_sdg=r, eval_sdg=e, split=split))

    if not out:
        raise FileNotFoundError(
            f"No files matched pattern under {data_dir} for run_tag={run_tag!r}, year={year}."
        )
    return out


def _require_schema(path: Path, df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")


def load_one(spec: FileSpec) -> pd.DataFrame:
    """Load one teacher CSV and attach filename-derived provenance columns."""

    df = pd.read_csv(spec.path)
    _require_schema(spec.path, df)

    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["DOI"] = out["DOI"].fillna("").astype(str).str.strip()
    out["parsed_label"] = out["parsed_label"].astype(str).str.strip()
    out["teacher_logit"] = pd.to_numeric(out["teacher_logit"], errors="coerce")
    out["p1"] = pd.to_numeric(out["p1"], errors="coerce")
    out["logit_0"] = pd.to_numeric(out["logit_0"], errors="coerce")
    out["logit_1"] = pd.to_numeric(out["logit_1"], errors="coerce")

    bad = sorted(set(out["parsed_label"].unique()) - set(LABEL_TO_BIT))
    if bad:
        raise ValueError(f"{spec.path.name}: unexpected parsed_label values: {bad}")

    out["label_bit"] = out["parsed_label"].map(LABEL_TO_BIT).astype(int)
    out["retrieval_sdg"] = spec.retrieval_sdg
    out["eval_sdg"] = spec.eval_sdg
    out["split"] = spec.split
    out["source_file"] = spec.path.name
    return out


def _check_eval_coverage(df: pd.DataFrame, strict: bool) -> list[str]:
    expected = set(range(1, 18))
    issues: list[str] = []
    for (r, split), g in df.groupby(["retrieval_sdg", "split"], sort=True):
        have = set(g["eval_sdg"].unique().tolist())
        if have != expected:
            missing = sorted(expected - have)
            extra = sorted(have - expected)
            issues.append(
                f"retrieval_sdg={r}, split={split}: missing_eval={missing}, extra_eval={extra}"
            )

    if issues and strict:
        raise ValueError("Eval-SDG coverage check failed:\n" + "\n".join(issues))
    return issues


def _check_rowid_alignment(df: pd.DataFrame, strict: bool) -> list[str]:
    issues: list[str] = []
    for (r, split), g in df.groupby(["retrieval_sdg", "split"], sort=True):
        by_eval = {
            int(e): set(sub["row_id"].tolist())
            for e, sub in g.groupby("eval_sdg", sort=True)
        }
        if not by_eval:
            continue

        ref_eval = sorted(by_eval.keys())[0]
        ref_rows = by_eval[ref_eval]
        for e, rows in by_eval.items():
            if rows != ref_rows:
                missing = len(ref_rows - rows)
                extra = len(rows - ref_rows)
                issues.append(
                    f"retrieval_sdg={r}, split={split}, eval_sdg={e}: "
                    f"row_id mismatch vs eval_sdg={ref_eval} (missing={missing}, extra={extra})"
                )

    if issues and strict:
        raise ValueError("row_id alignment check failed:\n" + "\n".join(issues))
    return issues


def build_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long teacher outputs into one row per retrieval_sdg/split/row_id."""

    # Resolve DOI per (retrieval_sdg, split, row_id): first non-empty DOI.
    keys = ["retrieval_sdg", "split", "row_id"]
    doi_resolved = (
        long_df.assign(_doi=long_df["DOI"].fillna("").astype(str).str.strip())
        .sort_values(keys + ["eval_sdg"])
        .groupby(keys, as_index=False)["_doi"]
        .agg(lambda s: next((x for x in s if x), ""))
        .rename(columns={"_doi": "DOI"})
    )

    # Warn-worthy consistency metric: how often DOI disagrees across eval SDGs.
    doi_card = (
        long_df.assign(_doi=long_df["DOI"].fillna("").astype(str).str.strip())
        .groupby(keys)["_doi"]
        .nunique(dropna=False)
        .rename("doi_unique_count")
        .reset_index()
    )
    wide = doi_resolved.merge(doi_card, on=keys, how="left", validate="one_to_one")

    value_cols = ["teacher_logit", "p1", "label_bit", "parsed_label"]
    for col in value_cols:
        p = long_df.pivot_table(
            index=keys,
            columns="eval_sdg",
            values=col,
            aggfunc="first",
        )
        p.columns = [f"{col}_sdg{int(c):02d}" for c in p.columns]
        p = p.reset_index()
        wide = wide.merge(p, on=keys, how="left", validate="one_to_one")

    sdg_cols = [f"teacher_logit_sdg{i:02d}" for i in range(1, 18)]
    present = [c for c in sdg_cols if c in wide.columns]
    wide["n_eval_present"] = wide[present].notna().sum(axis=1)
    wide["is_complete_17d"] = wide["n_eval_present"].eq(17)
    return wide


def main() -> None:
    """Combine discovered teacher files and write long and wide outputs."""

    args = parse_args()
    retrieval_filter = set(args.retrieval_sdgs) if args.retrieval_sdgs is not None else None

    files = discover_files(
        data_dir=args.data_dir,
        run_tag=args.run_tag,
        year=args.year,
        splits=args.splits,
        retrieval_filter=retrieval_filter,
    )
    print(f"Discovered {len(files)} files.")

    frames = [load_one(spec) for spec in files]
    long_df = pd.concat(frames, ignore_index=True)

    coverage_issues = _check_eval_coverage(long_df, strict=args.strict)
    align_issues = _check_rowid_alignment(long_df, strict=args.strict)
    if coverage_issues:
        print(f"WARNING: eval coverage issues={len(coverage_issues)}")
    if align_issues:
        print(f"WARNING: row_id alignment issues={len(align_issues)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_out_all = args.out_dir / "teacher_grid_long_all.csv"
    long_df.to_csv(long_out_all, index=False)
    print(f"Wrote {long_out_all} ({len(long_df)} rows)")

    for split, sub in long_df.groupby("split", sort=True):
        long_out = args.out_dir / f"teacher_grid_long_{split}.csv"
        sub.to_csv(long_out, index=False)
        print(f"Wrote {long_out} ({len(sub)} rows)")

        wide = build_wide(sub)
        wide_out = args.out_dir / f"teacher_grid_wide_{split}.csv"
        wide.to_csv(wide_out, index=False)
        n_complete = int(wide["is_complete_17d"].sum())
        print(
            f"Wrote {wide_out} ({len(wide)} rows, complete_17d={n_complete}/{len(wide)})"
        )


if __name__ == "__main__":
    main()
