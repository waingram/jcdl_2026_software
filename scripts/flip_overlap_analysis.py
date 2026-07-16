#!/usr/bin/env python3

# Description: Analyze overlap, reversion, and opposite-direction flip
# patterns across three labeling runs and enrich them with logits-based
# uncertainty diagnostics.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/flip_overlap_analysis.py [--justification PATH --binary-label PATH --binary-bit PATH --logits PATH --out-dir DIR --top-k N]

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


LABEL_R = "Relevant"
LABEL_NR = "Non-Relevant"


def _read_run_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"row_id", "SDG", "parsed_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    # Keep DOI if present (for reporting), but do not use it as a join key.
    keep_cols = ["row_id", "SDG", "parsed_label"]
    if "DOI" in df.columns:
        keep_cols.insert(1, "DOI")

    df = df[keep_cols].copy()
    df["row_id"] = df["row_id"].astype(int)
    df["SDG"] = df["SDG"].astype(str)
    df["parsed_label"] = df["parsed_label"].astype(str)

    # Basic normalization guard: we only accept the canonical labels.
    bad = ~df["parsed_label"].isin([LABEL_R, LABEL_NR])
    if bad.any():
        bad_vals = sorted(df.loc[bad, "parsed_label"].unique().tolist())
        raise ValueError(f"{path} has unexpected parsed_label values: {bad_vals}")

    return df


def _read_logits_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"row_id", "SDG", "teacher_logit", "p1", "logit_0", "logit_1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df[["row_id", "SDG", "teacher_logit", "p1", "logit_0", "logit_1"]].copy()
    df["row_id"] = df["row_id"].astype(int)
    df["SDG"] = df["SDG"].astype(str)
    for c in ["teacher_logit", "p1", "logit_0", "logit_1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["abs_margin"] = df["teacher_logit"].abs()
    df["dist_to_0_5"] = (df["p1"] - 0.5).abs()
    return df


def _flip_type(a: str, b: str) -> str:
    if a == b:
        return "no_flip"
    if a == LABEL_R and b == LABEL_NR:
        return "R->NR"
    if a == LABEL_NR and b == LABEL_R:
        return "NR->R"
    return "unknown"


def main() -> None:
    """Run the three-run flip-overlap analysis and write summary artifacts."""

    p = argparse.ArgumentParser(
        description=(
            "Analyze overlap between flips across two prompt transitions and enrich with logits."
        )
    )
    p.add_argument("--justification", type=Path, required=True)
    p.add_argument("--binary-label", type=Path, required=True)
    p.add_argument("--binary-bit", type=Path, required=True)
    p.add_argument("--logits", type=Path, required=True, help="binary_bit_with_probs CSV")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=200, help="Top-k lowest-margin rows to print and save.")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a = _read_run_csv(args.justification).rename(columns={"parsed_label": "label_A"})
    df_b = _read_run_csv(args.binary_label).rename(columns={"parsed_label": "label_B"})
    df_c = _read_run_csv(args.binary_bit).rename(columns={"parsed_label": "label_C"})
    df_l = _read_logits_csv(args.logits)

    # Join by row_id + SDG (SDG is a guardrail; row_id is the real key).
    key = ["row_id", "SDG"]

    # Merge A/B/C, preferring DOI from A then B then C if present.
    merged = (
        df_a.merge(df_b, on=key, how="inner", suffixes=("", "_b"))
        .merge(df_c, on=key, how="inner", suffixes=("", "_c"))
    )

    # Unify DOI if present; do not depend on DOI for joining.
    doi_cols = [c for c in merged.columns if c.startswith("DOI")]
    if doi_cols:
        # Choose first non-empty DOI across available DOI columns.
        doi_series = None
        for c in doi_cols:
            s = merged[c].fillna("").astype(str).str.strip()
            if doi_series is None:
                doi_series = s
            else:
                doi_series = doi_series.where(doi_series != "", s)
        merged["DOI"] = doi_series
        # Drop the intermediate DOI columns.
        for c in doi_cols:
            if c != "DOI":
                merged.drop(columns=[c], inplace=True, errors="ignore")

    # Enrich with logits from L
    merged = merged.merge(df_l, on=key, how="left")

    # Compute flip indicators and flip types
    merged["flip_AB"] = merged["label_A"] != merged["label_B"]
    merged["flip_BC"] = merged["label_B"] != merged["label_C"]
    merged["flip_AB_type"] = merged.apply(lambda r: _flip_type(r["label_A"], r["label_B"]), axis=1)
    merged["flip_BC_type"] = merged.apply(lambda r: _flip_type(r["label_B"], r["label_C"]), axis=1)

    flip_ab = merged[merged["flip_AB"]].copy()
    flip_bc = merged[merged["flip_BC"]].copy()

    # (1) Same rows flip in both transitions
    both = merged[merged["flip_AB"] & merged["flip_BC"]].copy()

    # (2) Reversion: A == C, but A != B
    reverted = merged[(merged["label_A"] == merged["label_C"]) & (merged["label_A"] != merged["label_B"])].copy()

    # (3) Double-change (non-reverting): flipped twice and ended different than start
    double_nonrevert = merged[(merged["flip_AB"] & merged["flip_BC"]) & (merged["label_A"] != merged["label_C"])].copy()

    # Specific neat case: one way then the other (R->NR then NR->R, or NR->R then R->NR)
    opposite_direction = merged[
        (merged["flip_AB"] & merged["flip_BC"]) &
        (merged["flip_AB_type"].isin(["R->NR", "NR->R"])) &
        (merged["flip_BC_type"].isin(["R->NR", "NR->R"])) &
        (merged["flip_AB_type"] != merged["flip_BC_type"])
    ].copy()

    # Sort by smallest absolute margin (uncertainty proxy) where available
    for df_ in (both, reverted, double_nonrevert, opposite_direction):
        if "abs_margin" in df_.columns:
            df_.sort_values(["abs_margin", "dist_to_0_5"], ascending=True, inplace=True)

    # Save CSVs
    merged_out = out_dir / "flip_overlap__merged_ABC_with_logits.csv"
    both_out = out_dir / "flip_overlap__flipAB_and_flipBC.csv"
    reverted_out = out_dir / "flip_overlap__reverted_A_eq_C.csv"
    double_out = out_dir / "flip_overlap__double_nonrevert.csv"
    opp_out = out_dir / "flip_overlap__opposite_direction.csv"
    top_out = out_dir / "flip_overlap__opposite_direction_topk_low_margin.csv"

    merged.to_csv(merged_out, index=False)
    both.to_csv(both_out, index=False)
    reverted.to_csv(reverted_out, index=False)
    double_nonrevert.to_csv(double_out, index=False)
    opposite_direction.to_csv(opp_out, index=False)

    topk = int(args.top_k)
    opposite_direction.head(topk).to_csv(top_out, index=False)

    # Summary stats
    summary: Dict[str, object] = {
        "files": {
            "A_justification": str(args.justification),
            "B_binary_label": str(args.binary_label),
            "C_binary_bit": str(args.binary_bit),
            "L_logits": str(args.logits),
        },
        "counts": {
            "n_total": int(len(merged)),
            "n_flip_AB": int(len(flip_ab)),
            "n_flip_BC": int(len(flip_bc)),
            "n_both_flips": int(len(both)),
            "n_reverted_A_eq_C": int(len(reverted)),
            "n_double_nonrevert": int(len(double_nonrevert)),
            "n_opposite_direction": int(len(opposite_direction)),
        },
        "rates": {
            "flip_AB_rate": float(len(flip_ab) / len(merged)) if len(merged) else 0.0,
            "flip_BC_rate": float(len(flip_bc) / len(merged)) if len(merged) else 0.0,
            "both_flips_rate": float(len(both) / len(merged)) if len(merged) else 0.0,
            "reverted_rate": float(len(reverted) / len(merged)) if len(merged) else 0.0,
            "opposite_direction_rate": float(len(opposite_direction) / len(merged)) if len(merged) else 0.0,
        },
        "flip_type_tables": {
            "AB": flip_ab["flip_AB_type"].value_counts(dropna=False).to_dict(),
            "BC": flip_bc["flip_BC_type"].value_counts(dropna=False).to_dict(),
            "both_AB_type": both["flip_AB_type"].value_counts(dropna=False).to_dict(),
            "both_BC_type": both["flip_BC_type"].value_counts(dropna=False).to_dict(),
            "opposite_direction_AB_type": opposite_direction["flip_AB_type"].value_counts(dropna=False).to_dict(),
            "opposite_direction_BC_type": opposite_direction["flip_BC_type"].value_counts(dropna=False).to_dict(),
        },
        "outputs": {
            "merged": merged_out.name,
            "both_flips": both_out.name,
            "reverted": reverted_out.name,
            "double_nonrevert": double_out.name,
            "opposite_direction": opp_out.name,
            "opposite_direction_topk_low_margin": top_out.name,
        },
    }

    summary_path = out_dir / "flip_overlap__summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Print a short console report
    print(json.dumps(summary["counts"], indent=2))
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
