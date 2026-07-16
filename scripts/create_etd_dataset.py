#!/usr/bin/env python3

# Description: Harvest Virginia Tech ETD records from VTechWorks OAI-PMH DIM
# feeds, convert the requested metadata fields to CSV, normalize department
# labels for reporting, and create department-stratified train/test splits.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/create_etd_dataset.py [--doctoral-url URL --masters-url URL --raw-dir DIR --processed-dir DIR --combined-name FILE --train-name FILE --test-name FILE --department-summary-name FILE --department-mapping-name FILE --department-candidates-name FILE --test-fraction FLOAT --seed INT --timeout-sec INT --skip-harvest --keep-empty-abstracts]

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


OAI_NS = "http://www.openarchives.org/OAI/2.0/"
DIM_NS = "http://www.dspace.org/xmlns/dspace/dim"
NS = {"oai": OAI_NS, "dim": DIM_NS}
LIST_DELIM = " || "
SUMMARY_LIST_DELIM = " ;; "
MISSING_DEPARTMENT_LABEL = "Missing Department"
DEFAULT_DEPARTMENT_MAPPING_NAME = "vt_etds_department_mapping_audit.csv"
DEFAULT_DEPARTMENT_CANDIDATES_NAME = "vt_etds_department_similarity_candidates.csv"

# Conservative manual merges approved from the similarity-candidate review.
# Keys are the rule-based department keys produced by _department_key().
APPROVED_DEPARTMENT_OVERRIDES = {
    "agriculture and extension education": "Agricultural and Extension Education",
    "animal sciences": "Animal Science",
    "career counseling and student personnel": "Counseling and Student Personnel Services",
    "career counseling and student personnel services": "Counseling and Student Personnel Services",
    "clothing textiles and related art": "Clothing, Textiles, and Related Arts",
    "computer science and application": "Computer Science and Applications",
    "computer sciences": "Computer Science",
    "counseling and college student personnel": "Counseling and Student Personnel Services",
    "counseling student personnel services": "Counseling and Student Personnel Services",
    "counselor education and student personnel": "Counselor Education and Student Personnel Services",
    "education administration": "Educational Administration",
    "education research and evaluation": "Educational Research and Evaluation",
    "educational policy research and evaluation": "Educational Research and Evaluation",
    "environmental science and engineering": "Environmental Sciences and Engineering",
    "family and child devolopment": "Family and Child Development",
    "fisheries and wildlife science": "Fisheries and Wildlife Sciences",
    "forestry and forestry products": "Forestry and Forest Products",
    "geological science": "Geological Sciences",
    "hotel restaurant and institution management": "Hotel, Restaurant, and Institutional Management",
    "industrial engineering and operation research": "Industrial Engineering and Operations Research",
    "industrial systems engineering": "Industrial and Systems Engineering",
    "materials and engineering science": "Materials Engineering Science",
    "materials engineering and science": "Materials Engineering Science",
    "public administration public affairs": "Public Administration and Public Affairs",
    "vocational technical education": "Vocational and Technical Education",
    "wildlife science": "Wildlife Sciences",
}

DEFAULT_DOCTORAL_URL = (
    "https://vtechworks.lib.vt.edu/server/oai/request"
    "?verb=ListRecords&metadataPrefix=dim&set=col_10919_11041"
)
DEFAULT_MASTERS_URL = (
    "https://vtechworks.lib.vt.edu/server/oai/request"
    "?verb=ListRecords&metadataPrefix=dim&set=col_10919_9291"
)
DEFAULT_VT_MAJORS_PATH = Path(__file__).resolve().parents[3] / "docs" / "vt_majors.md"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for VT ETD harvesting and CSV export."""

    parser = argparse.ArgumentParser(
        description=(
            "Harvest Virginia Tech ETD OAI-PMH records, convert DIM XML metadata "
            "to CSV, and create department-stratified train/test splits."
        )
    )
    parser.add_argument("--doctoral-url", default=DEFAULT_DOCTORAL_URL)
    parser.add_argument("--masters-url", default=DEFAULT_MASTERS_URL)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/vtechworks_oai"),
        help="Directory where raw OAI-PMH XML pages will be stored.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/vtechworks"),
        help="Directory where combined and split CSVs will be written.",
    )
    parser.add_argument(
        "--combined-name",
        default="vt_etds_all.csv",
        help="Filename for the combined harvested CSV.",
    )
    parser.add_argument(
        "--train-name",
        default="vt_etds_train.csv",
        help="Filename for the training split CSV.",
    )
    parser.add_argument(
        "--test-name",
        default="vt_etds_test.csv",
        help="Filename for the test split CSV.",
    )
    parser.add_argument(
        "--department-summary-name",
        default="vt_etds_department_summary.csv",
        help="Filename for the normalized department summary CSV.",
    )
    parser.add_argument(
        "--department-mapping-name",
        default=DEFAULT_DEPARTMENT_MAPPING_NAME,
        help="Filename for the raw-to-canonical department mapping audit CSV.",
    )
    parser.add_argument(
        "--department-candidates-name",
        default=DEFAULT_DEPARTMENT_CANDIDATES_NAME,
        help="Filename for the fuzzy department similarity candidate CSV.",
    )
    parser.add_argument(
        "--vt-majors-path",
        type=Path,
        default=DEFAULT_VT_MAJORS_PATH,
        help="Path to the VT majors reference list used for summary matching.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of each department assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="Random seed for the stratified split.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="Per-request timeout in seconds for OAI-PMH harvest calls.",
    )
    parser.add_argument(
        "--skip-harvest",
        action="store_true",
        help="Reuse previously downloaded XML in --raw-dir instead of harvesting again.",
    )
    parser.add_argument(
        "--keep-empty-abstracts",
        action="store_true",
        help="Keep records with no abstract text instead of filtering them out before export.",
    )
    return parser.parse_args()


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _request_base(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _resumption_url(base_url: str, token: str) -> str:
    return f"{base_url}?{urlencode({'verb': 'ListRecords', 'resumptionToken': token})}"


def _download_page(url: str, timeout_sec: int) -> bytes:
    request = Request(url, headers={"User-Agent": "sdg-distillation/0.1"})
    with urlopen(request, timeout=timeout_sec) as response:
        return response.read()


def _extract_resumption_token(root: ET.Element) -> str:
    token_el = root.find(".//oai:resumptionToken", NS)
    if token_el is None or token_el.text is None:
        return ""
    return token_el.text.strip()


def harvest_oai_pages(
    initial_url: str,
    raw_dir: Path,
    prefix: str,
    timeout_sec: int,
) -> list[Path]:
    """Download all paged OAI-PMH responses for one endpoint into raw_dir."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[Path] = []
    request_url = initial_url
    base_url = _request_base(initial_url)
    page_no = 1

    while True:
        payload = _download_page(request_url, timeout_sec=timeout_sec)
        out_path = raw_dir / f"{prefix}_page_{page_no:04d}.xml"
        out_path.write_bytes(payload)
        page_paths.append(out_path)

        root = ET.fromstring(payload)
        error_el = root.find(".//oai:error", NS)
        if error_el is not None:
            raise RuntimeError(f"OAI-PMH error for {request_url}: {_normalize_text(error_el.text)}")

        token = _extract_resumption_token(root)
        if not token:
            break
        request_url = _resumption_url(base_url, token)
        page_no += 1

    return page_paths


def _match_field(
    field: ET.Element,
    *,
    mdschema: str,
    element: str,
    qualifier: str | None,
    lang: str | None,
) -> bool:
    if field.attrib.get("mdschema") != mdschema:
        return False
    if field.attrib.get("element") != element:
        return False
    field_qualifier = field.attrib.get("qualifier")
    if qualifier is None:
        if field_qualifier not in (None, ""):
            return False
    elif field_qualifier != qualifier:
        return False
    if lang is None:
        return True
    return field.attrib.get("lang") == lang


def _extract_dim_values(
    record: ET.Element,
    *,
    mdschema: str,
    element: str,
    qualifier: str | None = None,
    lang: str | None = "en",
) -> list[str]:
    fields = record.findall(".//oai:metadata/dim:dim/dim:field", NS)
    exact = [
        _normalize_text(field.text)
        for field in fields
        if _match_field(
            field,
            mdschema=mdschema,
            element=element,
            qualifier=qualifier,
            lang=lang,
        )
    ]
    exact = _dedupe_preserve_order(exact)
    if exact:
        return exact

    # Fallback to no-language match if the English field is absent.
    fallback = [
        _normalize_text(field.text)
        for field in fields
        if _match_field(
            field,
            mdschema=mdschema,
            element=element,
            qualifier=qualifier,
            lang=None,
        )
    ]
    return _dedupe_preserve_order(fallback)


def _join_values(values: list[str]) -> str:
    return LIST_DELIM.join(values)


def _split_joined_values(value: str) -> list[str]:
    if not value:
        return []
    return _dedupe_preserve_order(part.strip() for part in value.split(LIST_DELIM))


def _clean_department_display(value: str | None) -> str:
    text = html.unescape(value or "")
    text = text.replace("and#38;", "and")
    text = text.replace("&#38;", "and")
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\b(?:with|and)\s+a\s+major\s+in\s+[^,;|]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = _normalize_text(text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip(" ,")


def _department_key(value: str | None) -> str:
    text = _clean_department_display(value).lower()
    text = re.sub(r"\bdept\.?\b", "department", text)
    text = text.replace("-", " ")
    text = re.sub(r"[/;(),]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_vt_majors(path: Path) -> list[str]:
    if not path.exists():
        return []

    majors: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.replace("\u200b", "").strip()
        if not line or line == "Under Construction":
            continue
        while re.search(r"\([^)]*\)\s*$", line):
            line = re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
        if line:
            majors.append(line)
    return majors


def _normalize_major_surface(text: str) -> str:
    normalized = html.unescape(text).replace("\u200b", "")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.lower()
    normalized = re.sub(r"\btech\b", "technology", normalized)
    normalized = re.sub(r"\bengr\b", "engineering", normalized)
    normalized = re.sub(r"\becono\b", "economics", normalized)
    normalized = re.sub(r"\bmgt\b", "management", normalized)
    normalized = re.sub(r"\bresour\b", "resource", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_major_tokens(text: str) -> list[str]:
    tokens = _normalize_major_surface(text).split()
    out: list[str] = []
    for token in tokens:
        if token in {"and", "of", "the"}:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        out.append(token)
    return out


def _levenshtein_distance(left: str, right: str) -> int:
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


def _levenshtein_ratio(left: str, right: str) -> float:
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(left, right) / max_len)


def _token_jaccard(left_tokens: list[str], right_tokens: list[str]) -> float:
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def best_vt_major_match(
    department_label: str,
    vt_majors: list[str],
) -> tuple[str, float, str]:
    if not vt_majors:
        return "", 0.0, "no_reference_list"

    department_surface = _normalize_major_surface(department_label)
    department_tokens = _normalize_major_tokens(department_label)

    best_major = ""
    best_score = 0.0
    best_type = "none"

    for major in vt_majors:
        major_surface = _normalize_major_surface(major)
        major_tokens = _normalize_major_tokens(major)
        lev_ratio = _levenshtein_ratio(department_surface, major_surface)
        token_overlap = _token_jaccard(department_tokens, major_tokens)
        combined = (0.7 * lev_ratio) + (0.3 * token_overlap)

        if department_surface == major_surface:
            return major, 1.0, "exact"

        if combined > best_score:
            best_major = major
            best_score = combined
            best_type = "fuzzy"

    if best_score >= 0.90:
        return best_major, round(best_score, 4), "strong_fuzzy"
    if best_score >= 0.82:
        return best_major, round(best_score, 4), "review_fuzzy"
    return "", round(best_score, 4), "none"


def _normalize_single_department(
    department: str | None,
    canonical_map: dict[str, str],
) -> tuple[str, str, str, str]:
    raw_department = _normalize_text(department)
    if not raw_department:
        return MISSING_DEPARTMENT_LABEL, "", "", "missing_department"

    cleaned_department = _clean_department_display(raw_department)
    key = _department_key(cleaned_department)
    if not key:
        return MISSING_DEPARTMENT_LABEL, cleaned_department, "", "missing_department"

    if key in APPROVED_DEPARTMENT_OVERRIDES:
        return (
            APPROVED_DEPARTMENT_OVERRIDES[key],
            cleaned_department,
            key,
            "approved_override",
        )

    canonical_department = canonical_map.get(key, cleaned_department)
    if canonical_department != cleaned_department:
        return canonical_department, cleaned_department, key, "canonical_key_merge"
    if cleaned_department != raw_department:
        return canonical_department, cleaned_department, key, "rule_based_cleanup"
    return canonical_department, cleaned_department, key, "identity"


def build_department_canonical_map(rows: list[dict[str, str]]) -> dict[str, str]:
    """Choose a canonical display label for each normalized department key."""

    variants: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for department in _split_joined_values(row.get("department", "")):
            key = _department_key(department)
            if not key:
                continue
            display = _clean_department_display(department)
            variants[key][display] += 1

    canonical_map: dict[str, str] = {}
    for key, counts in variants.items():
        canonical_map[key] = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
    return canonical_map


def normalize_department_label(value: str | None, canonical_map: dict[str, str]) -> str:
    """Normalize a raw department string to a canonical reporting label."""

    departments = _split_joined_values(value or "")
    if not departments:
        return MISSING_DEPARTMENT_LABEL

    normalized_departments: list[str] = []
    for department in departments:
        canonical_department, _, _, _ = _normalize_single_department(
            department,
            canonical_map,
        )
        if canonical_department == MISSING_DEPARTMENT_LABEL:
            continue
        normalized_departments.append(canonical_department)

    normalized_departments = _dedupe_preserve_order(normalized_departments)
    if not normalized_departments:
        return MISSING_DEPARTMENT_LABEL
    return _join_values(normalized_departments)


def add_normalized_department_column(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Add department_normalized to each row and return the canonical mapping."""

    canonical_map = build_department_canonical_map(rows)
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_row: dict[str, str] = {}
        for key, value in row.items():
            normalized_row[key] = value
            if key == "department":
                normalized_row["department_normalized"] = normalize_department_label(
                    value,
                    canonical_map,
                )
        normalized_rows.append(normalized_row)
    return normalized_rows, canonical_map


def parse_oai_records(xml_path: Path, source_set: str) -> list[dict[str, str]]:
    """Parse one harvested DIM XML page into row dictionaries."""

    root = ET.fromstring(xml_path.read_bytes())
    records: list[dict[str, str]] = []
    for record in root.findall(".//oai:record", NS):
        header = record.find("oai:header", NS)
        if header is None:
            continue
        if header.attrib.get("status") == "deleted":
            continue

        oai_identifier = _normalize_text(header.findtext("oai:identifier", default="", namespaces=NS))

        authors = _extract_dim_values(record, mdschema="dc", element="contributor", qualifier="author")
        departments = _extract_dim_values(
            record,
            mdschema="dc",
            element="contributor",
            qualifier="department",
        )
        committee_chairs = _extract_dim_values(
            record,
            mdschema="dc",
            element="contributor",
            qualifier="committeechair",
        )
        committee_cochairs = _extract_dim_values(
            record,
            mdschema="dc",
            element="contributor",
            qualifier="committeecochair",
        )
        abstracts = _extract_dim_values(
            record,
            mdschema="dc",
            element="description",
            qualifier="abstract",
        )
        titles = _extract_dim_values(record, mdschema="dc", element="title", qualifier=None)
        uris = _extract_dim_values(
            record,
            mdschema="dc",
            element="identifier",
            qualifier="uri",
        )
        degrees = _extract_dim_values(
            record,
            mdschema="dc",
            element="description",
            qualifier="degree",
        )
        date_issued = _extract_dim_values(
            record,
            mdschema="dc",
            element="date",
            qualifier="issued",
        )

        records.append(
            {
                "source_set": source_set,
                "oai_identifier": oai_identifier,
                "date_issued": _join_values(date_issued),
                "author": _join_values(authors),
                "department": _join_values(departments),
                "committeechair": _join_values(committee_chairs),
                "committeecochair": _join_values(committee_cochairs),
                "abstract": _join_values(abstracts),
                "title": _join_values(titles),
                "uri": _join_values(uris),
                "degree": _join_values(degrees),
            }
        )
    return records


def build_etd_records(raw_dir: Path) -> list[dict[str, str]]:
    """Parse and deduplicate all harvested VT ETD XML files in raw_dir."""

    records: list[dict[str, str]] = []
    for xml_path in sorted(raw_dir.glob("*.xml")):
        source_set = "doctoral" if xml_path.name.startswith("doctoral_") else "masters"
        records.extend(parse_oai_records(xml_path, source_set=source_set))

    if not records:
        raise ValueError(f"No OAI-PMH records parsed from {raw_dir}")

    deduped: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in records:
        dedupe_key = (row.get("oai_identifier", ""), row.get("uri", ""))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(row)
    return deduped


def filter_rows(
    rows: list[dict[str, str]],
    *,
    keep_empty_abstracts: bool,
) -> tuple[list[dict[str, str]], int]:
    """Optionally drop records that do not contain abstract text."""

    if keep_empty_abstracts:
        return rows, 0

    filtered = [row for row in rows if _normalize_text(row.get("abstract", ""))]
    removed = len(rows) - len(filtered)
    return filtered, removed


def stratified_split_by_department(
    rows: list[dict[str, str]],
    *,
    department_col: str,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic train/test split stratified by department column."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    group_to_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        key = _normalize_text(row.get(department_col, "")) or "__MISSING_DEPARTMENT__"
        group_to_indices.setdefault(key, []).append(idx)

    test_indices: set[int] = set()
    train_indices: set[int] = set()

    for indices in group_to_indices.values():
        n_rows = len(indices)
        if n_rows == 1:
            train_indices.update(indices)
            continue
        test_n = int(round(n_rows * test_fraction))
        test_n = max(1, test_n)
        test_n = min(test_n, n_rows - 1)
        picked = set(rng.sample(indices, k=test_n))
        for idx in indices:
            if idx in picked:
                test_indices.add(idx)
            else:
                train_indices.add(idx)

    train_rows = [rows[idx] for idx in sorted(train_indices)]
    test_rows = [rows[idx] for idx in sorted(test_indices)]
    return train_rows, test_rows


def write_rows_to_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a list of row dictionaries to CSV using the first row's schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def department_counts(rows: list[dict[str, str]], department_col: str) -> dict[str, int]:
    """Count rows by department label for reporting and split diagnostics."""

    counts: dict[str, int] = {}
    for row in rows:
        key = _normalize_text(row.get(department_col, "")) or "__MISSING_DEPARTMENT__"
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_department_mapping_audit(
    rows: list[dict[str, str]],
    canonical_map: dict[str, str],
    *,
    vt_majors: list[str] | None = None,
) -> list[dict[str, int | str]]:
    """Build a readable audit table from raw department strings to canonical labels."""

    audit: dict[str, dict[str, int | str]] = {}
    for row in rows:
        departments = _split_joined_values(row.get("department", ""))
        if not departments:
            departments = [""]

        for raw_department in departments:
            bucket = audit.get(raw_department)
            if bucket is None:
                canonical_department, cleaned_department, key, mapping_source = (
                    _normalize_single_department(raw_department, canonical_map)
                )
                vt_major_match, vt_major_score, vt_major_match_type = best_vt_major_match(
                    canonical_department,
                    vt_majors or [],
                )
                bucket = {
                    "raw_department": raw_department,
                    "cleaned_department": cleaned_department,
                    "department_key": key,
                    "canonical_department": canonical_department,
                    "mapping_source": mapping_source,
                    "vt_major_match": vt_major_match,
                    "vt_major_match_score": vt_major_score,
                    "vt_major_match_type": vt_major_match_type,
                    "record_count": 0,
                    "doctoral_count": 0,
                    "masters_count": 0,
                }
                audit[raw_department] = bucket
            bucket["record_count"] = int(bucket["record_count"]) + 1
            source_set = row.get("source_set", "")
            if source_set == "doctoral":
                bucket["doctoral_count"] = int(bucket["doctoral_count"]) + 1
            elif source_set == "masters":
                bucket["masters_count"] = int(bucket["masters_count"]) + 1

    return sorted(
        audit.values(),
        key=lambda row: (
            str(row["canonical_department"]),
            -int(row["record_count"]),
            str(row["raw_department"]),
        ),
    )


def summarize_departments(
    rows: list[dict[str, str]],
    *,
    vt_majors: list[str] | None = None,
) -> list[dict[str, int | str | float]]:
    """Build a normalized department summary table for Chapter 3 reporting."""

    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        department = _normalize_text(row.get("department_normalized", "")) or MISSING_DEPARTMENT_LABEL
        stats = summary.setdefault(
            department,
            {
                "department_normalized": department,
                "record_count": 0,
                "doctoral_count": 0,
                "masters_count": 0,
                "raw_department_variants": Counter(),
            },
        )
        stats["record_count"] = int(stats["record_count"]) + 1
        source_set = row.get("source_set", "")
        if source_set == "doctoral":
            stats["doctoral_count"] = int(stats["doctoral_count"]) + 1
        elif source_set == "masters":
            stats["masters_count"] = int(stats["masters_count"]) + 1

        raw_department = _normalize_text(row.get("department", ""))
        if raw_department:
            raw_counts = stats["raw_department_variants"]
            assert isinstance(raw_counts, Counter)
            raw_counts[raw_department] += 1

    summary_rows: list[dict[str, int | str]] = []
    for department, stats in sorted(
        summary.items(),
        key=lambda item: (-int(item[1]["record_count"]), item[0]),
    ):
        raw_counts = stats["raw_department_variants"]
        assert isinstance(raw_counts, Counter)
        vt_major_match, vt_major_score, vt_major_match_type = best_vt_major_match(
            department,
            vt_majors or [],
        )
        summary_rows.append(
            {
                "department_normalized": department,
                "record_count": int(stats["record_count"]),
                "doctoral_count": int(stats["doctoral_count"]),
                "masters_count": int(stats["masters_count"]),
                "raw_department_variant_count": len(raw_counts),
                "raw_department_variants": SUMMARY_LIST_DELIM.join(
                    [name for name, _ in sorted(
                        raw_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )]
                ),
                "vt_major_match": vt_major_match,
                "vt_major_match_score": vt_major_score,
                "vt_major_match_type": vt_major_match_type,
            }
        )
    return summary_rows


def write_department_candidate_report(
    *,
    summary_path: Path,
    output_path: Path,
) -> int:
    """Write fuzzy candidate department pairs for later manual review."""

    try:
        from scripts.find_etd_department_candidates import (
            build_candidate_rows,
            load_department_summary,
            write_candidate_rows,
        )
    except ModuleNotFoundError:
        module_path = Path(__file__).with_name("find_etd_department_candidates.py")
        spec = importlib.util.spec_from_file_location(
            "find_etd_department_candidates",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load department candidate module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        build_candidate_rows = module.build_candidate_rows
        load_department_summary = module.load_department_summary
        write_candidate_rows = module.write_candidate_rows

    summary_rows = load_department_summary(summary_path)
    candidate_rows = build_candidate_rows(
        summary_rows,
        min_levenshtein=0.78,
        min_token_jaccard=0.50,
        min_combined_score=0.78,
    )
    write_candidate_rows(output_path, candidate_rows)
    return len(candidate_rows)


def main() -> None:
    """Harvest or reuse VT ETD XML, then write normalized dataset exports."""

    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_harvest:
        doctoral_paths = sorted(args.raw_dir.glob("doctoral_*.xml"))
        masters_paths = sorted(args.raw_dir.glob("masters_*.xml"))
        if not doctoral_paths or not masters_paths:
            raise ValueError(
                "--skip-harvest was provided, but the expected XML files were not found "
                f"in {args.raw_dir}"
            )
        print(
            f"Reusing {len(doctoral_paths)} doctoral pages and {len(masters_paths)} masters pages "
            f"from {args.raw_dir}"
        )
    else:
        doctoral_paths = harvest_oai_pages(
            args.doctoral_url,
            raw_dir=args.raw_dir,
            prefix="doctoral",
            timeout_sec=args.timeout_sec,
        )
        masters_paths = harvest_oai_pages(
            args.masters_url,
            raw_dir=args.raw_dir,
            prefix="masters",
            timeout_sec=args.timeout_sec,
        )
        print(
            f"Downloaded {len(doctoral_paths)} doctoral pages and {len(masters_paths)} masters pages "
            f"to {args.raw_dir}"
        )

    rows = build_etd_records(args.raw_dir)
    rows, removed_empty_abstracts = filter_rows(
        rows,
        keep_empty_abstracts=args.keep_empty_abstracts,
    )
    rows, canonical_map = add_normalized_department_column(rows)
    vt_majors = _load_vt_majors(args.vt_majors_path)
    combined_path = args.processed_dir / args.combined_name
    write_rows_to_csv(combined_path, rows)
    print(f"Wrote combined CSV: {combined_path} ({len(rows)} rows)")
    if removed_empty_abstracts:
        print(f"Filtered out {removed_empty_abstracts} records with empty abstracts.")

    train_rows, test_rows = stratified_split_by_department(
        rows,
        department_col="department_normalized",
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    train_path = args.processed_dir / args.train_name
    test_path = args.processed_dir / args.test_name
    write_rows_to_csv(train_path, train_rows)
    write_rows_to_csv(test_path, test_rows)

    print(f"Wrote train CSV: {train_path} ({len(train_rows)} rows)")
    print(f"Wrote test CSV: {test_path} ({len(test_rows)} rows)")

    department_summary_rows = summarize_departments(rows, vt_majors=vt_majors)
    department_summary_path = args.processed_dir / args.department_summary_name
    write_rows_to_csv(department_summary_path, department_summary_rows)
    print(
        f"Wrote department summary CSV: {department_summary_path} "
        f"({len(department_summary_rows)} rows)"
    )

    department_mapping_rows = build_department_mapping_audit(
        rows,
        canonical_map,
        vt_majors=vt_majors,
    )
    department_mapping_path = args.processed_dir / args.department_mapping_name
    write_rows_to_csv(department_mapping_path, department_mapping_rows)
    print(
        f"Wrote department mapping audit CSV: {department_mapping_path} "
        f"({len(department_mapping_rows)} rows)"
    )

    department_candidates_path = args.processed_dir / args.department_candidates_name
    candidate_count = write_department_candidate_report(
        summary_path=department_summary_path,
        output_path=department_candidates_path,
    )
    print(
        f"Wrote department similarity candidates CSV: {department_candidates_path} "
        f"({candidate_count} rows)"
    )

    dept_counts = department_counts(rows, "department_normalized")
    singleton_departments = sum(1 for count in dept_counts.values() if count == 1)
    print(
        f"Departments: {len(dept_counts)} total, {singleton_departments} singleton departments "
        "retained in train only."
    )


if __name__ == "__main__":
    main()
