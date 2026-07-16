from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "find_etd_department_candidates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "find_etd_department_candidates",
    SCRIPT_PATH,
)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalize_tokens_handles_structural_words_and_pluralization() -> None:
    tokens = MODULE.normalize_tokens("Department of Statistics")
    assert tokens == ["statistic"]


def test_build_candidate_rows_surfaces_clear_misspelling() -> None:
    rows = [
        MODULE.DepartmentRecord(
            department_normalized="Environmental Sciences and Engineering",
            record_count=100,
            doctoral_count=60,
            masters_count=40,
            raw_department_variant_count=1,
            raw_department_variants="Environmental Sciences and Engineering",
        ),
        MODULE.DepartmentRecord(
            department_normalized="Environmental Science and Engineering",
            record_count=10,
            doctoral_count=6,
            masters_count=4,
            raw_department_variant_count=1,
            raw_department_variants="Environmental Science and Engineering",
        ),
        MODULE.DepartmentRecord(
            department_normalized="Mechanical Engineering",
            record_count=200,
            doctoral_count=80,
            masters_count=120,
            raw_department_variant_count=1,
            raw_department_variants="Mechanical Engineering",
        ),
    ]

    candidates = MODULE.build_candidate_rows(
        rows,
        min_levenshtein=0.78,
        min_token_jaccard=0.50,
        min_combined_score=0.78,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["department_a"] == "Environmental Sciences and Engineering"
    assert candidate["department_b"] == "Environmental Science and Engineering"
    assert candidate["candidate_reason"] in {
        "same_content_tokens",
        "very_high_levenshtein",
    }


def test_write_candidate_rows_writes_header_for_empty_candidate_set(tmp_path: Path) -> None:
    output_path = tmp_path / "candidates.csv"

    MODULE.write_candidate_rows(output_path, [])

    content = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert content == [",".join(MODULE.CANDIDATE_FIELDNAMES)]
