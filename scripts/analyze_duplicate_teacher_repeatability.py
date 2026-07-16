#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


N_SDGS = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze repeated teacher inference for duplicate-DOI publication records. "
            "Writes occurrence-, group-, pairwise-, and SDG-level summaries."
        )
    )
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path("outputs/teacher/publication"),
        help="Directory containing teacher_grid_wide_{split}.csv files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing sdg{n}_2023_{split}.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/duplicate_teacher_repeatability_v1"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to analyze (default: test).",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Exclude full abstract text from the occurrences CSV.",
    )
    return parser.parse_args()


def sdg_cols(prefix: str) -> list[str]:
    return [f"{prefix}_sdg{i:02d}" for i in range(1, N_SDGS + 1)]


LABEL_COLS = sdg_cols("label_bit")
P1_COLS = sdg_cols("p1")
LOGIT_COLS = sdg_cols("teacher_logit")


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def load_teacher_split(teacher_dir: Path, split: str) -> pd.DataFrame:
    path = teacher_dir / f"teacher_grid_wide_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"DOI", "retrieval_sdg", "row_id"} | set(LABEL_COLS) | set(P1_COLS) | set(LOGIT_COLS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing[:10]}")
    df["DOI"] = df["DOI"].fillna("").astype(str).str.strip()
    df["retrieval_sdg"] = pd.to_numeric(df["retrieval_sdg"], errors="raise").astype(int)
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    return df


def attach_source_metadata(
    duplicate_rows: pd.DataFrame,
    processed_dir: Path,
    year: int,
    split: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for retrieval_sdg in sorted(duplicate_rows["retrieval_sdg"].unique().tolist()):
        src_path = processed_dir / f"sdg{retrieval_sdg}_{year}_{split}.csv"
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        src = pd.read_csv(src_path, usecols=lambda c: c in {"DOI", "Abstract", "SDG"})
        src["row_id"] = np.arange(len(src), dtype=int)
        src["retrieval_sdg"] = retrieval_sdg
        src["source_doi"] = src["DOI"].fillna("").astype(str).str.strip()
        src["abstract_text"] = src["Abstract"].fillna("").astype(str)
        src["abstract_hash"] = src["abstract_text"].map(_sha1_text)
        src["abstract_char_count"] = src["abstract_text"].str.len().astype(int)
        frames.append(src[["retrieval_sdg", "row_id", "source_doi", "abstract_text", "abstract_hash", "abstract_char_count"]])
    source = pd.concat(frames, ignore_index=True)
    out = duplicate_rows.merge(source, on=["retrieval_sdg", "row_id"], how="left", validate="one_to_one")
    if out["abstract_text"].isna().any():
        missing = int(out["abstract_text"].isna().sum())
        raise ValueError(f"{split}: missing source abstract text for {missing} duplicate rows")
    return out


def build_occurrences(df: pd.DataFrame, split: str, include_text: bool) -> pd.DataFrame:
    dup_counts = df.groupby("DOI").size().rename("group_size")
    duplicate_dois = dup_counts[dup_counts > 1].index
    occurrences = df[df["DOI"].isin(duplicate_dois)].copy()
    occurrences["group_size"] = occurrences["DOI"].map(dup_counts).astype(int)
    occurrences["group_id"] = split + "|" + occurrences["DOI"]
    occurrences["teacher_vector_hash"] = hash_pandas_object(
        occurrences[LABEL_COLS + P1_COLS + LOGIT_COLS],
        index=False,
    ).astype(str)
    occurrences["num_positive_hard_labels"] = occurrences[LABEL_COLS].sum(axis=1).astype(int)
    occurrences["mean_p1"] = occurrences[P1_COLS].mean(axis=1)
    occurrences["mean_abs_teacher_logit"] = occurrences[LOGIT_COLS].abs().mean(axis=1)
    occurrences["same_source_doi"] = occurrences["source_doi"].eq(occurrences["DOI"])

    front = [
        "group_id",
        "split",
        "DOI",
        "group_size",
        "retrieval_sdg",
        "row_id",
        "source_doi",
        "same_source_doi",
        "abstract_hash",
        "abstract_char_count",
        "teacher_vector_hash",
        "num_positive_hard_labels",
        "mean_p1",
        "mean_abs_teacher_logit",
    ]
    if include_text:
        front.append("abstract_text")
    rest = [c for c in occurrences.columns if c not in front]
    return occurrences[front + rest].sort_values(["DOI", "retrieval_sdg", "row_id"]).reset_index(drop=True)


def analyze_groups(occurrences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pairwise_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    sdg_group_rows: list[dict[str, Any]] = []
    sdg_pair_rows: list[dict[str, Any]] = []

    split = str(occurrences["split"].iloc[0]) if not occurrences.empty else ""

    differing_combo_counter: Counter[str] = Counter()

    for doi, group in occurrences.groupby("DOI", sort=True):
        group = group.reset_index(drop=True)
        group_size = int(group.shape[0])
        retrieval_seq = [int(x) for x in group["retrieval_sdg"].tolist()]
        retrieval_seq_str = "|".join(str(x) for x in retrieval_seq)
        retrieval_set = sorted(set(retrieval_seq))
        retrieval_set_str = "|".join(str(x) for x in retrieval_set)
        abstract_hashes = group["abstract_hash"].astype(str)
        vector_hashes = group["teacher_vector_hash"].astype(str)
        same_text_all = abstract_hashes.nunique() == 1
        unique_abstract_hash_count = int(abstract_hashes.nunique())
        unique_teacher_vector_count = int(vector_hashes.nunique())
        has_teacher_vector_difference = unique_teacher_vector_count > 1
        if has_teacher_vector_difference:
            differing_combo_counter[retrieval_seq_str] += 1

        num_pairs = group_size * (group_size - 1) // 2
        pairwise_mean_abs_delta_p1: list[float] = []
        pairwise_median_abs_delta_p1: list[float] = []
        pairwise_max_abs_delta_p1: list[float] = []
        pairwise_mean_abs_delta_logit: list[float] = []
        pairwise_median_abs_delta_logit: list[float] = []
        pairwise_max_abs_delta_logit: list[float] = []
        pairwise_flip_event_total = 0
        pairwise_flip_sdgs: set[int] = set()

        for sdg in range(1, N_SDGS + 1):
            label_col = f"label_bit_sdg{sdg:02d}"
            p1_col = f"p1_sdg{sdg:02d}"
            logit_col = f"teacher_logit_sdg{sdg:02d}"
            labels = group[label_col].to_numpy(dtype=int)
            p1_vals = group[p1_col].to_numpy(dtype=float)
            logit_vals = group[logit_col].to_numpy(dtype=float)
            sdg_group_rows.append(
                {
                    "split": split,
                    "DOI": doi,
                    "group_id": f"{split}|{doi}",
                    "group_size": group_size,
                    "sdg": sdg,
                    "retrieval_sdgs": retrieval_seq_str,
                    "same_text_all": same_text_all,
                    "teacher_vector_differs": has_teacher_vector_difference,
                    "p1_mean": float(np.mean(p1_vals)),
                    "p1_std": float(np.std(p1_vals, ddof=0)),
                    "p1_range": float(np.max(p1_vals) - np.min(p1_vals)),
                    "logit_mean": float(np.mean(logit_vals)),
                    "logit_std": float(np.std(logit_vals, ddof=0)),
                    "logit_range": float(np.max(logit_vals) - np.min(logit_vals)),
                    "hard_positive_rate": float(np.mean(labels)),
                    "hard_flip_any": bool(np.unique(labels).size > 1),
                }
            )

        records = group.to_dict("records")
        for a, b in combinations(records, 2):
            p1_a = np.array([a[col] for col in P1_COLS], dtype=float)
            p1_b = np.array([b[col] for col in P1_COLS], dtype=float)
            z_a = np.array([a[col] for col in LOGIT_COLS], dtype=float)
            z_b = np.array([b[col] for col in LOGIT_COLS], dtype=float)
            h_a = np.array([a[col] for col in LABEL_COLS], dtype=int)
            h_b = np.array([b[col] for col in LABEL_COLS], dtype=int)

            p1_delta = np.abs(p1_a - p1_b)
            logit_delta = np.abs(z_a - z_b)
            hard_flip_mask = h_a != h_b
            hard_flip_sdgs = [idx + 1 for idx, flag in enumerate(hard_flip_mask) if flag]
            hard_flip_count = int(hard_flip_mask.sum())
            pairwise_flip_event_total += hard_flip_count
            pairwise_flip_sdgs.update(hard_flip_sdgs)

            if hard_flip_count > 0:
                flip_threshold_distances = [
                    float(min(abs(p1_a[idx] - 0.5), abs(p1_b[idx] - 0.5)))
                    for idx in np.where(hard_flip_mask)[0]
                ]
                min_flip_threshold_distance = float(np.min(flip_threshold_distances))
                mean_flip_threshold_distance = float(np.mean(flip_threshold_distances))
                max_flip_threshold_distance = float(np.max(flip_threshold_distances))
            else:
                min_flip_threshold_distance = np.nan
                mean_flip_threshold_distance = np.nan
                max_flip_threshold_distance = np.nan

            pairwise_mean_abs_delta_p1.append(float(np.mean(p1_delta)))
            pairwise_median_abs_delta_p1.append(float(np.median(p1_delta)))
            pairwise_max_abs_delta_p1.append(float(np.max(p1_delta)))
            pairwise_mean_abs_delta_logit.append(float(np.mean(logit_delta)))
            pairwise_median_abs_delta_logit.append(float(np.median(logit_delta)))
            pairwise_max_abs_delta_logit.append(float(np.max(logit_delta)))

            pairwise_rows.append(
                {
                    "split": split,
                    "DOI": doi,
                    "group_id": f"{split}|{doi}",
                    "group_size": group_size,
                    "retrieval_sdgs": retrieval_seq_str,
                    "retrieval_sdg_a": int(a["retrieval_sdg"]),
                    "row_id_a": int(a["row_id"]),
                    "retrieval_sdg_b": int(b["retrieval_sdg"]),
                    "row_id_b": int(b["row_id"]),
                    "same_text": bool(a["abstract_hash"] == b["abstract_hash"]),
                    "teacher_vector_equal": bool(a["teacher_vector_hash"] == b["teacher_vector_hash"]),
                    "mean_abs_delta_p1": float(np.mean(p1_delta)),
                    "median_abs_delta_p1": float(np.median(p1_delta)),
                    "max_abs_delta_p1": float(np.max(p1_delta)),
                    "mean_abs_delta_logit": float(np.mean(logit_delta)),
                    "median_abs_delta_logit": float(np.median(logit_delta)),
                    "max_abs_delta_logit": float(np.max(logit_delta)),
                    "hard_flip_event_count": hard_flip_count,
                    "hard_flip_sdg_count": len(hard_flip_sdgs),
                    "hard_flip_sdgs": "|".join(str(x) for x in hard_flip_sdgs),
                    "min_flip_threshold_distance": min_flip_threshold_distance,
                    "mean_flip_threshold_distance": mean_flip_threshold_distance,
                    "max_flip_threshold_distance": max_flip_threshold_distance,
                }
            )

            for idx in range(N_SDGS):
                sdg_pair_rows.append(
                    {
                        "split": split,
                        "DOI": doi,
                        "group_id": f"{split}|{doi}",
                        "sdg": idx + 1,
                        "same_text": bool(a["abstract_hash"] == b["abstract_hash"]),
                        "teacher_vector_equal": bool(a["teacher_vector_hash"] == b["teacher_vector_hash"]),
                        "abs_delta_p1": float(p1_delta[idx]),
                        "abs_delta_logit": float(logit_delta[idx]),
                        "hard_flip": bool(hard_flip_mask[idx]),
                        "flip_threshold_distance": (
                            float(min(abs(p1_a[idx] - 0.5), abs(p1_b[idx] - 0.5))) if hard_flip_mask[idx] else np.nan
                        ),
                    }
                )

        p1_ranges = [row["p1_range"] for row in sdg_group_rows[-N_SDGS:]]
        logit_ranges = [row["logit_range"] for row in sdg_group_rows[-N_SDGS:]]

        group_rows.append(
            {
                "split": split,
                "DOI": doi,
                "group_id": f"{split}|{doi}",
                "group_size": group_size,
                "retrieval_sdgs": retrieval_seq_str,
                "retrieval_sdg_set": retrieval_set_str,
                "unique_retrieval_sdgs": len(retrieval_set),
                "unique_abstract_hash_count": unique_abstract_hash_count,
                "same_text_all": same_text_all,
                "unique_teacher_vector_count": unique_teacher_vector_count,
                "has_teacher_vector_difference": has_teacher_vector_difference,
                "num_pairwise_comparisons": num_pairs,
                "pairwise_hard_flip_event_count": pairwise_flip_event_total,
                "pairwise_hard_flip_sdg_count": len(pairwise_flip_sdgs),
                "has_any_hard_flip": len(pairwise_flip_sdgs) > 0,
                "hard_flip_sdgs": "|".join(str(x) for x in sorted(pairwise_flip_sdgs)),
                "avg_pairwise_mean_abs_delta_p1": float(np.mean(pairwise_mean_abs_delta_p1)) if pairwise_mean_abs_delta_p1 else 0.0,
                "median_pairwise_mean_abs_delta_p1": float(np.median(pairwise_mean_abs_delta_p1)) if pairwise_mean_abs_delta_p1 else 0.0,
                "max_pairwise_max_abs_delta_p1": float(np.max(pairwise_max_abs_delta_p1)) if pairwise_max_abs_delta_p1 else 0.0,
                "avg_pairwise_mean_abs_delta_logit": float(np.mean(pairwise_mean_abs_delta_logit)) if pairwise_mean_abs_delta_logit else 0.0,
                "median_pairwise_mean_abs_delta_logit": float(np.median(pairwise_mean_abs_delta_logit)) if pairwise_mean_abs_delta_logit else 0.0,
                "max_pairwise_max_abs_delta_logit": float(np.max(pairwise_max_abs_delta_logit)) if pairwise_max_abs_delta_logit else 0.0,
                "mean_group_p1_range": float(np.mean(p1_ranges)) if p1_ranges else 0.0,
                "median_group_p1_range": float(np.median(p1_ranges)) if p1_ranges else 0.0,
                "max_group_p1_range": float(np.max(p1_ranges)) if p1_ranges else 0.0,
                "mean_group_logit_range": float(np.mean(logit_ranges)) if logit_ranges else 0.0,
                "median_group_logit_range": float(np.median(logit_ranges)) if logit_ranges else 0.0,
                "max_group_logit_range": float(np.max(logit_ranges)) if logit_ranges else 0.0,
            }
        )

    groups_df = pd.DataFrame(group_rows).sort_values(["has_teacher_vector_difference", "group_size", "DOI"], ascending=[False, False, True]).reset_index(drop=True)
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["DOI", "retrieval_sdg_a", "row_id_a", "retrieval_sdg_b", "row_id_b"]).reset_index(drop=True)
    sdg_group_df = pd.DataFrame(sdg_group_rows)
    sdg_pair_df = pd.DataFrame(sdg_pair_rows)

    sdg_summary_rows: list[dict[str, Any]] = []
    for sdg in range(1, N_SDGS + 1):
        gsub = sdg_group_df[sdg_group_df["sdg"] == sdg]
        psub = sdg_pair_df[sdg_pair_df["sdg"] == sdg]
        flip_distances = psub.loc[psub["hard_flip"], "flip_threshold_distance"].dropna().to_numpy(dtype=float)
        sdg_summary_rows.append(
            {
                "split": split,
                "sdg": sdg,
                "groups_observed": int(len(gsub)),
                "groups_with_teacher_vector_difference": int(gsub["teacher_vector_differs"].sum()),
                "groups_with_hard_flip": int(gsub["hard_flip_any"].sum()),
                "mean_group_p1": float(gsub["p1_mean"].mean()),
                "mean_group_p1_range": float(gsub["p1_range"].mean()),
                "median_group_p1_range": float(gsub["p1_range"].median()),
                "max_group_p1_range": float(gsub["p1_range"].max()),
                "mean_group_logit_range": float(gsub["logit_range"].mean()),
                "median_group_logit_range": float(gsub["logit_range"].median()),
                "max_group_logit_range": float(gsub["logit_range"].max()),
                "pairwise_observations": int(len(psub)),
                "mean_pairwise_abs_delta_p1": float(psub["abs_delta_p1"].mean()),
                "median_pairwise_abs_delta_p1": float(psub["abs_delta_p1"].median()),
                "max_pairwise_abs_delta_p1": float(psub["abs_delta_p1"].max()),
                "mean_pairwise_abs_delta_logit": float(psub["abs_delta_logit"].mean()),
                "median_pairwise_abs_delta_logit": float(psub["abs_delta_logit"].median()),
                "max_pairwise_abs_delta_logit": float(psub["abs_delta_logit"].max()),
                "hard_flip_event_count": int(psub["hard_flip"].sum()),
                "hard_flip_rate": float(psub["hard_flip"].mean()),
                "mean_flip_threshold_distance": float(np.mean(flip_distances)) if flip_distances.size else np.nan,
                "median_flip_threshold_distance": float(np.median(flip_distances)) if flip_distances.size else np.nan,
                "max_flip_threshold_distance": float(np.max(flip_distances)) if flip_distances.size else np.nan,
            }
        )
    sdg_summary_df = pd.DataFrame(sdg_summary_rows).sort_values("sdg").reset_index(drop=True)

    summary = {
        "split": split,
        "duplicate_group_count": int(len(groups_df)),
        "groups_with_teacher_vector_difference": int(groups_df["has_teacher_vector_difference"].sum()),
        "groups_with_identical_teacher_vectors": int((~groups_df["has_teacher_vector_difference"]).sum()),
        "groups_with_same_text_all": int(groups_df["same_text_all"].sum()),
        "differing_groups_with_same_text_all": int(groups_df.loc[groups_df["has_teacher_vector_difference"], "same_text_all"].sum()),
        "groups_with_any_hard_flip": int(groups_df["has_any_hard_flip"].sum()),
        "pairwise_hard_flip_event_count": int(groups_df["pairwise_hard_flip_event_count"].sum()),
        "median_avg_pairwise_mean_abs_delta_p1": float(groups_df["avg_pairwise_mean_abs_delta_p1"].median()),
        "median_max_pairwise_max_abs_delta_p1": float(groups_df["max_pairwise_max_abs_delta_p1"].median()),
        "p95_max_pairwise_max_abs_delta_p1": float(groups_df["max_pairwise_max_abs_delta_p1"].quantile(0.95)),
        "median_avg_pairwise_mean_abs_delta_logit": float(groups_df["avg_pairwise_mean_abs_delta_logit"].median()),
        "median_max_pairwise_max_abs_delta_logit": float(groups_df["max_pairwise_max_abs_delta_logit"].median()),
        "top_differing_retrieval_combinations": [
            {"retrieval_sdgs": combo, "count": count}
            for combo, count in differing_combo_counter.most_common(10)
        ],
        "top_sdgs_by_hard_flip_events": [
            {"sdg": int(row.sdg), "hard_flip_event_count": int(row.hard_flip_event_count)}
            for row in sdg_summary_df.sort_values("hard_flip_event_count", ascending=False).head(10).itertuples(index=False)
        ],
    }

    return groups_df, pairwise_df, sdg_summary_df, summary


def run_split(
    *,
    teacher_dir: Path,
    processed_dir: Path,
    out_dir: Path,
    year: int,
    split: str,
    include_text: bool,
) -> dict[str, Any]:
    teacher = load_teacher_split(teacher_dir, split)
    empty_doi_rows = int((teacher["DOI"] == "").sum())
    teacher_nonempty = teacher[teacher["DOI"] != ""].copy()
    counts = teacher_nonempty.groupby("DOI").size()
    duplicate_dois = counts[counts > 1].index
    duplicate_rows = teacher_nonempty[teacher_nonempty["DOI"].isin(duplicate_dois)].copy()
    duplicate_rows = attach_source_metadata(duplicate_rows, processed_dir, year, split)
    duplicate_rows["split"] = split
    occurrences_df = build_occurrences(duplicate_rows, split, include_text=include_text)

    groups_df, pairwise_df, sdg_summary_df, summary = analyze_groups(occurrences_df)
    summary.update(
        {
            "teacher_rows_total": int(len(teacher)),
            "teacher_rows_with_empty_doi_excluded": empty_doi_rows,
            "teacher_rows_nonempty_doi": int(len(teacher_nonempty)),
            "duplicate_occurrence_rows": int(len(occurrences_df)),
            "duplicate_rows_beyond_first": int((counts[counts > 1] - 1).sum()),
            "duplicate_group_size_distribution": {str(k): int(v) for k, v in Counter(groups_df["group_size"].tolist()).items()},
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    occ_path = out_dir / f"duplicate_teacher_occurrences_{split}.csv"
    grp_path = out_dir / f"duplicate_teacher_groups_{split}.csv"
    pair_path = out_dir / f"duplicate_teacher_pairwise_{split}.csv"
    sdg_path = out_dir / f"duplicate_teacher_sdg_summary_{split}.csv"
    summary_path = out_dir / f"duplicate_teacher_summary_{split}.json"

    occurrences_df.to_csv(occ_path, index=False)
    groups_df.to_csv(grp_path, index=False)
    pairwise_df.to_csv(pair_path, index=False)
    sdg_summary_df.to_csv(sdg_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({k: _to_python(v) for k, v in summary.items()}, f, indent=2)

    print(f"[{split}] Wrote {occ_path} ({len(occurrences_df)} rows)")
    print(f"[{split}] Wrote {grp_path} ({len(groups_df)} rows)")
    print(f"[{split}] Wrote {pair_path} ({len(pairwise_df)} rows)")
    print(f"[{split}] Wrote {sdg_path} ({len(sdg_summary_df)} rows)")
    print(f"[{split}] Wrote {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    for split in args.splits:
        summaries.append(
            run_split(
                teacher_dir=args.teacher_dir,
                processed_dir=args.processed_dir,
                out_dir=args.out_dir,
                year=args.year,
                split=split,
                include_text=not args.no_text,
            )
        )

    print("\nSummary")
    for item in summaries:
        print(
            f"- {item['split']}: duplicate_groups={item['duplicate_group_count']}, "
            f"differing_groups={item['groups_with_teacher_vector_difference']}, "
            f"groups_with_any_hard_flip={item['groups_with_any_hard_flip']}, "
            f"median_max_pairwise_max_abs_delta_p1={item['median_max_pairwise_max_abs_delta_p1']:.4f}"
        )


if __name__ == "__main__":
    main()
