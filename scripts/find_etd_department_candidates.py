#!/usr/bin/env python3

# Description: Identify likely duplicate or misspelled Virginia Tech ETD
# department labels by comparing normalized department names with
# Levenshtein similarity and token-overlap heuristics.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/find_etd_department_candidates.py [--input PATH --output PATH --min-levenshtein FLOAT --min-token-jaccard FLOAT --min-combined-score FLOAT --limit N]

from __future__ import annotations

import argparse
import csv
import html
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


STRUCTURAL_STOPWORDS = {
    "and",
    "of",
    "the",
    "department",
    "school",
    "center",
    "centre",
    "for",
    "with",
}

CANDIDATE_FIELDNAMES = [
    "candidate_reason",
    "combined_score",
    "levenshtein_ratio",
    "token_jaccard",
    "department_a",
    "record_count_a",
    "raw_variant_count_a",
    "raw_variants_a",
    "department_b",
    "record_count_b",
    "raw_variant_count_b",
    "raw_variants_b",
]


@dataclass(frozen=True)
class DepartmentRecord:
    """Normalized department row used for similarity comparison."""

    department_normalized: str
    record_count: int
    doctoral_count: int
    masters_count: int
    raw_department_variant_count: int
    raw_department_variants: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ETD department candidate generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a review table of likely duplicate department labels from "
            "vt_etds_department_summary.csv using Levenshtein and token-overlap heuristics."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/vtechworks/vt_etds_department_summary.csv"),
        help="Department summary CSV created by create_etd_dataset.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/vtechworks/vt_etds_department_similarity_candidates.csv"),
        help="Output CSV path for candidate duplicate department pairs.",
    )
    parser.add_argument(
        "--min-levenshtein",
        type=float,
        default=0.78,
        help="Minimum Levenshtein ratio for a candidate pair.",
    )
    parser.add_argument(
        "--min-token-jaccard",
        type=float,
        default=0.50,
        help="Minimum token Jaccard overlap for a candidate pair.",
    )
    parser.add_argument(
        "--min-combined-score",
        type=float,
        default=0.78,
        help="Minimum weighted combined score for a candidate pair.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of candidate rows to write after sorting.",
    )
    return parser.parse_args()


def load_department_summary(path: Path) -> list[DepartmentRecord]:
    """Load normalized ETD department summary rows from CSV."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                DepartmentRecord(
                    department_normalized=row["department_normalized"],
                    record_count=int(row["record_count"]),
                    doctoral_count=int(row["doctoral_count"]),
                    masters_count=int(row["masters_count"]),
                    raw_department_variant_count=int(row["raw_department_variant_count"]),
                    raw_department_variants=row["raw_department_variants"],
                )
            )
    if not rows:
        raise ValueError(f"No department rows found in {path}")
    return rows


def normalize_surface(text: str) -> str:
    """Normalize department text for surface-form similarity comparison."""

    normalized = html.unescape(text)
    normalized = normalized.replace("and#38;", "and")
    normalized = normalized.replace("&#38;", "and")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_tokens(text: str) -> list[str]:
    """Normalize department text into content-bearing tokens."""

    surface = normalize_surface(text)
    tokens = []
    for token in surface.split():
        if token in STRUCTURAL_STOPWORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance between two strings."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_left != char_right)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def levenshtein_ratio(left: str, right: str) -> float:
    """Compute a max-length-normalized Levenshtein similarity ratio."""

    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(left, right) / max_len)


def token_jaccard(left_tokens: list[str], right_tokens: list[str]) -> float:
    """Compute Jaccard overlap between two token lists."""

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def candidate_reason(
    *,
    left_surface: str,
    right_surface: str,
    left_tokens: list[str],
    right_tokens: list[str],
    lev_ratio: float,
    jaccard: float,
) -> str:
    """Classify why a department pair was surfaced for manual review."""

    if left_tokens == right_tokens:
        return "same_content_tokens"
    if lev_ratio >= 0.95:
        return "very_high_levenshtein"
    if jaccard >= 0.80 and lev_ratio >= 0.70:
        return "high_token_overlap"
    if left_surface in right_surface or right_surface in left_surface:
        return "substring_match"
    return "mixed_similarity"


def build_candidate_rows(
    rows: list[DepartmentRecord],
    *,
    min_levenshtein: float,
    min_token_jaccard: float,
    min_combined_score: float,
) -> list[dict[str, str | int | float]]:
    """Build review rows for likely duplicate department labels."""

    candidates: list[dict[str, str | int | float]] = []
    for left, right in combinations(rows, 2):
        left_surface = normalize_surface(left.department_normalized)
        right_surface = normalize_surface(right.department_normalized)
        left_tokens = normalize_tokens(left.department_normalized)
        right_tokens = normalize_tokens(right.department_normalized)

        lev_ratio = levenshtein_ratio(left_surface, right_surface)
        jaccard = token_jaccard(left_tokens, right_tokens)
        combined = (0.7 * lev_ratio) + (0.3 * jaccard)

        if lev_ratio < min_levenshtein:
            continue
        if jaccard < min_token_jaccard and left_tokens != right_tokens:
            continue
        if combined < min_combined_score:
            continue

        candidates.append(
            {
                "candidate_reason": candidate_reason(
                    left_surface=left_surface,
                    right_surface=right_surface,
                    left_tokens=left_tokens,
                    right_tokens=right_tokens,
                    lev_ratio=lev_ratio,
                    jaccard=jaccard,
                ),
                "combined_score": round(combined, 4),
                "levenshtein_ratio": round(lev_ratio, 4),
                "token_jaccard": round(jaccard, 4),
                "department_a": left.department_normalized,
                "record_count_a": left.record_count,
                "raw_variant_count_a": left.raw_department_variant_count,
                "raw_variants_a": left.raw_department_variants,
                "department_b": right.department_normalized,
                "record_count_b": right.record_count,
                "raw_variant_count_b": right.raw_department_variant_count,
                "raw_variants_b": right.raw_department_variants,
            }
        )

    candidates.sort(
        key=lambda row: (
            -float(row["combined_score"]),
            -float(row["levenshtein_ratio"]),
            -float(row["token_jaccard"]),
            min(int(row["record_count_a"]), int(row["record_count_b"])),
            str(row["department_a"]),
            str(row["department_b"]),
        )
    )
    return candidates


def write_candidate_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Write candidate department pairs to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate and write candidate ETD department duplicate pairs."""

    args = parse_args()
    rows = load_department_summary(args.input)
    candidates = build_candidate_rows(
        rows,
        min_levenshtein=args.min_levenshtein,
        min_token_jaccard=args.min_token_jaccard,
        min_combined_score=args.min_combined_score,
    )
    if args.limit is not None:
        candidates = candidates[: args.limit]
    write_candidate_rows(args.output, candidates)
    print(f"Wrote {len(candidates)} candidate pairs to {args.output}")


if __name__ == "__main__":
    main()
