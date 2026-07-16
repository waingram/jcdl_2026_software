#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

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

SOURCE_COLS = [
    "source_set",
    "oai_identifier",
    "date_issued",
    "author",
    "department",
    "department_normalized",
    "committeechair",
    "committeecochair",
    "abstract",
    "title",
    "uri",
    "degree",
]


@dataclass(frozen=True)
class FileSpec:
    path: Path
    split: str
    eval_sdg: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine VTechWorks ETD teacher CSVs into long and wide teacher-reference outputs. "
            "Expected filename pattern: vt_etds_{split}__EVAL_sdg{e}__{run_tag}.csv"
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/teacher/etd"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/etd"),
        help="Directory containing vt_etds_{train,test}.csv source files.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="vtechworks_qwen_binary_bit_with_probs_v1",
        help="Suffix in filenames after the final '__' separator (without .csv).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to include (default: train test).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/teacher/etd"),
        help="Output directory for combined teacher-reference files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on eval coverage, row alignment, or URI mismatch issues.",
    )
    return parser.parse_args()


def discover_files(data_dir: Path, run_tag: str, splits: set[str]) -> list[FileSpec]:
    rx = re.compile(rf"^vt_etds_(train|test)__EVAL_sdg(\d+)__{re.escape(run_tag)}\.csv$")
    out: list[FileSpec] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        match = rx.match(path.name)
        if not match:
            continue
        split = match.group(1).lower()
        if split not in splits:
            continue
        out.append(FileSpec(path=path, split=split, eval_sdg=int(match.group(2))))
    if not out:
        raise FileNotFoundError(
            f"No VTechWorks teacher files found in {data_dir} for run_tag={run_tag!r}"
        )
    return out


def _require_schema(path: Path, df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")


def load_one(spec: FileSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.path)
    _require_schema(spec.path, df)

    out = df.copy()
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["uri"] = out["DOI"].fillna("").astype(str).str.strip()
    out["parsed_label"] = out["parsed_label"].astype(str).str.strip()
    out["teacher_logit"] = pd.to_numeric(out["teacher_logit"], errors="coerce")
    out["p1"] = pd.to_numeric(out["p1"], errors="coerce")
    out["logit_0"] = pd.to_numeric(out["logit_0"], errors="coerce")
    out["logit_1"] = pd.to_numeric(out["logit_1"], errors="coerce")

    bad = sorted(set(out["parsed_label"].unique()) - set(LABEL_TO_BIT))
    if bad:
        raise ValueError(f"{spec.path.name}: unexpected parsed_label values: {bad}")

    out["label_bit"] = out["parsed_label"].map(LABEL_TO_BIT).astype(int)
    out["split"] = spec.split
    out["eval_sdg"] = spec.eval_sdg
    out["source_file"] = spec.path.name
    return out[
        [
            "split",
            "row_id",
            "uri",
            "eval_sdg",
            "generated_text",
            "parsed_label",
            "label_bit",
            "logit_0",
            "logit_1",
            "teacher_logit",
            "p1",
            "source_file",
        ]
    ]


def _load_source_split(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    src = pd.read_csv(path)
    missing = [c for c in SOURCE_COLS if c not in src.columns]
    if missing:
        raise ValueError(f"{path.name}: missing source metadata columns: {missing}")
    src = src[SOURCE_COLS].copy()
    src["row_id"] = range(len(src))
    src["split"] = split
    src["uri"] = src["uri"].fillna("").astype(str).str.strip()
    return src[["split", "row_id", *SOURCE_COLS]]


def _check_eval_coverage(df: pd.DataFrame, strict: bool) -> list[str]:
    expected = set(range(1, 18))
    issues: list[str] = []
    for split, group in df.groupby("split", sort=True):
        have = set(group["eval_sdg"].unique().tolist())
        if have != expected:
            missing = sorted(expected - have)
            extra = sorted(have - expected)
            issues.append(f"split={split}: missing_eval={missing}, extra_eval={extra}")
    if issues and strict:
        raise ValueError("Eval-SDG coverage check failed:\n" + "\n".join(issues))
    return issues


def _check_rowid_alignment(df: pd.DataFrame, strict: bool) -> list[str]:
    issues: list[str] = []
    for split, group in df.groupby("split", sort=True):
        by_eval = {
            int(eval_sdg): set(sub["row_id"].tolist())
            for eval_sdg, sub in group.groupby("eval_sdg", sort=True)
        }
        if not by_eval:
            continue
        ref_eval = sorted(by_eval.keys())[0]
        ref_rows = by_eval[ref_eval]
        for eval_sdg, rows in by_eval.items():
            if rows != ref_rows:
                issues.append(
                    f"split={split}, eval_sdg={eval_sdg}: "
                    f"row_id mismatch vs eval_sdg={ref_eval} "
                    f"(missing={len(ref_rows - rows)}, extra={len(rows - ref_rows)})"
                )
    if issues and strict:
        raise ValueError("row_id alignment check failed:\n" + "\n".join(issues))
    return issues


def build_wide(long_df: pd.DataFrame, source_df: pd.DataFrame, *, strict: bool) -> pd.DataFrame:
    keys = ["split", "row_id"]

    uri_resolved = (
        long_df.sort_values(keys + ["eval_sdg"])
        .groupby(keys, as_index=False)["uri"]
        .agg(lambda s: next((x for x in s if x), ""))
        .rename(columns={"uri": "teacher_uri"})
    )

    uri_card = (
        long_df.groupby(keys)["uri"]
        .nunique(dropna=False)
        .rename("teacher_uri_unique_count")
        .reset_index()
    )

    wide = uri_resolved.merge(uri_card, on=keys, how="left", validate="one_to_one")
    wide = wide.merge(
        source_df,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator="_source_merge",
    )
    if not wide["_source_merge"].eq("both").all():
        n_missing = int((wide["_source_merge"] != "both").sum())
        raise ValueError(f"Missing source metadata rows after join: {n_missing}")
    wide = wide.drop(columns="_source_merge")

    uri_mismatch = (
        (wide["teacher_uri"] != "")
        & (wide["uri"] != "")
        & wide["teacher_uri"].ne(wide["uri"])
    )
    if uri_mismatch.any():
        n_bad = int(uri_mismatch.sum())
        msg = f"Teacher/source URI mismatch rows: {n_bad}"
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")

    for col in ["teacher_logit", "p1", "label_bit", "parsed_label"]:
        pivoted = long_df.pivot_table(
            index=keys,
            columns="eval_sdg",
            values=col,
            aggfunc="first",
        )
        pivoted.columns = [f"{col}_sdg{int(c):02d}" for c in pivoted.columns]
        wide = wide.merge(pivoted.reset_index(), on=keys, how="left", validate="one_to_one")

    present = [f"teacher_logit_sdg{i:02d}" for i in range(1, 18) if f"teacher_logit_sdg{i:02d}" in wide.columns]
    wide["n_eval_present"] = wide[present].notna().sum(axis=1)
    wide["is_complete_17d"] = wide["n_eval_present"].eq(17)
    wide["sample_id"] = wide["split"].astype(str) + "_" + wide["row_id"].astype(str)
    front = [
        "sample_id",
        "split",
        "row_id",
        "uri",
        "teacher_uri",
        "teacher_uri_unique_count",
        "source_set",
        "oai_identifier",
        "date_issued",
        "author",
        "department",
        "department_normalized",
        "committeechair",
        "committeecochair",
        "degree",
        "title",
        "abstract",
    ]
    remainder = [c for c in wide.columns if c not in front]
    return wide[front + remainder]


def main() -> None:
    args = parse_args()
    splits = {s.strip().lower() for s in args.splits}
    files = discover_files(args.data_dir, args.run_tag, splits)
    print(f"Discovered {len(files)} files.")

    frames = [load_one(spec) for spec in files]
    long_df = pd.concat(frames, ignore_index=True)

    coverage_issues = _check_eval_coverage(long_df, strict=args.strict)
    alignment_issues = _check_rowid_alignment(long_df, strict=args.strict)
    if coverage_issues:
        print(f"WARNING: eval coverage issues={len(coverage_issues)}")
    if alignment_issues:
        print(f"WARNING: row_id alignment issues={len(alignment_issues)}")

    source_frames = [
        _load_source_split(args.processed_dir / f"vt_etds_{split}.csv", split)
        for split in sorted(splits)
    ]
    source_df = pd.concat(source_frames, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_all_path = args.out_dir / "teacher_grid_long_all.csv"
    long_df.to_csv(long_all_path, index=False)
    print(f"Wrote {long_all_path} ({len(long_df)} rows)")

    wide_all = build_wide(long_df, source_df, strict=bool(args.strict))
    wide_all_path = args.out_dir / "teacher_grid_wide_all.csv"
    wide_all.to_csv(wide_all_path, index=False)
    print(f"Wrote {wide_all_path} ({len(wide_all)} rows)")

    for split, split_long in long_df.groupby("split", sort=True):
        split_long_path = args.out_dir / f"teacher_grid_long_{split}.csv"
        split_long.to_csv(split_long_path, index=False)
        print(f"Wrote {split_long_path} ({len(split_long)} rows)")

        split_source = source_df[source_df["split"] == split].reset_index(drop=True)
        split_wide = build_wide(split_long.reset_index(drop=True), split_source, strict=bool(args.strict))
        split_wide_path = args.out_dir / f"teacher_grid_wide_{split}.csv"
        split_wide.to_csv(split_wide_path, index=False)
        n_complete = int(split_wide["is_complete_17d"].sum())
        print(f"Wrote {split_wide_path} ({len(split_wide)} rows, complete_17d={n_complete}/{len(split_wide)})")


if __name__ == "__main__":
    main()
