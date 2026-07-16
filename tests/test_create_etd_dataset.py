from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "create_etd_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("create_etd_dataset", SCRIPT_PATH)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


SAMPLE_XML = """\
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:dim="http://www.dspace.org/xmlns/dspace/dim">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:test:1</identifier>
        <datestamp>2026-03-01</datestamp>
        <setSpec>col_10919_11041</setSpec>
      </header>
      <metadata>
        <dim:dim>
          <dim:field mdschema="dc" element="contributor" qualifier="author" lang="en">Alice Author</dim:field>
          <dim:field mdschema="dc" element="contributor" qualifier="department" lang="en">Computer Science</dim:field>
          <dim:field mdschema="dc" element="contributor" qualifier="committeechair" lang="en">Chair One</dim:field>
          <dim:field mdschema="dc" element="contributor" qualifier="committeecochair" lang="en">Cochair A</dim:field>
          <dim:field mdschema="dc" element="contributor" qualifier="committeecochair" lang="en">Cochair B</dim:field>
          <dim:field mdschema="dc" element="description" qualifier="abstract" lang="en">An abstract.</dim:field>
          <dim:field mdschema="dc" element="title" lang="en">A Title</dim:field>
          <dim:field mdschema="dc" element="identifier" qualifier="uri" lang="en">http://example.org/item/1</dim:field>
          <dim:field mdschema="dc" element="date" qualifier="issued" lang="en">2024-05-06</dim:field>
          <dim:field mdschema="dc" element="description" qualifier="degree" lang="en">Doctor of Philosophy</dim:field>
        </dim:dim>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
"""


def test_parse_oai_records_extracts_requested_fields(tmp_path: Path) -> None:
    xml_path = tmp_path / "doctoral_page_0001.xml"
    xml_path.write_text(SAMPLE_XML)

    records = MODULE.parse_oai_records(xml_path, source_set="doctoral")
    assert len(records) == 1
    row = records[0]
    assert row["source_set"] == "doctoral"
    assert row["author"] == "Alice Author"
    assert row["department"] == "Computer Science"
    assert row["committeechair"] == "Chair One"
    assert row["committeecochair"] == "Cochair A || Cochair B"
    assert row["abstract"] == "An abstract."
    assert row["title"] == "A Title"
    assert row["uri"] == "http://example.org/item/1"
    assert row["date_issued"] == "2024-05-06"
    assert row["degree"] == "Doctor of Philosophy"
    assert "oai_datestamp" not in row
    assert "set_spec" not in row


def test_stratified_split_by_department_preserves_department_presence() -> None:
    rows = [
        {"department": "CS", "title": "t0"},
        {"department": "CS", "title": "t1"},
        {"department": "CS", "title": "t2"},
        {"department": "Math", "title": "t3"},
        {"department": "Math", "title": "t4"},
        {"department": "Physics", "title": "t5"},
    ]
    train_rows, test_rows = MODULE.stratified_split_by_department(
        rows,
        department_col="department",
        test_fraction=0.2,
        seed=200,
    )

    train_departments = {row["department"] for row in train_rows}
    test_departments = {row["department"] for row in test_rows}

    assert len(train_rows) + len(test_rows) == len(rows)
    assert "Physics" in train_departments
    assert "Physics" not in test_departments
    assert "CS" in train_departments
    assert "CS" in test_departments
    assert "Math" in train_departments
    assert "Math" in test_departments


def test_filter_rows_drops_empty_abstracts_by_default() -> None:
    rows = [
        {"title": "has abstract", "abstract": "Useful text", "department": "CS"},
        {"title": "missing abstract", "abstract": "   ", "department": "Math"},
        {"title": "also missing", "abstract": "", "department": "Physics"},
    ]

    filtered, removed = MODULE.filter_rows(rows, keep_empty_abstracts=False)

    assert removed == 2
    assert [row["title"] for row in filtered] == ["has abstract"]


def test_filter_rows_can_keep_empty_abstracts() -> None:
    rows = [
        {"title": "has abstract", "abstract": "Useful text"},
        {"title": "missing abstract", "abstract": ""},
    ]

    filtered, removed = MODULE.filter_rows(rows, keep_empty_abstracts=True)

    assert removed == 0
    assert filtered == rows


def test_add_normalized_department_column_collapses_common_variants() -> None:
    rows = [
        {"department": "Computer Science & Applications", "title": "t0"},
        {"department": "Computer Science and#38; Applications", "title": "t1"},
        {"department": "Computer Science and Applications", "title": "t2"},
        {"department": "", "title": "t3"},
    ]

    normalized_rows, canonical_map = MODULE.add_normalized_department_column(rows)

    assert canonical_map[MODULE._department_key("Computer Science & Applications")] == (
        "Computer Science and Applications"
    )
    assert normalized_rows[0]["department_normalized"] == "Computer Science and Applications"
    assert normalized_rows[1]["department_normalized"] == "Computer Science and Applications"
    assert normalized_rows[2]["department_normalized"] == "Computer Science and Applications"
    assert normalized_rows[3]["department_normalized"] == MODULE.MISSING_DEPARTMENT_LABEL


def test_department_normalization_strips_parenthetical_qualifiers() -> None:
    rows = [
        {
            "department": "Family and Child Development (Marriage and Family Therapy)",
            "title": "t0",
        },
        {
            "department": "Family and Child Development (Marriage and Family Therapy Emphasis)",
            "title": "t1",
        },
    ]

    normalized_rows, _ = MODULE.add_normalized_department_column(rows)

    assert normalized_rows[0]["department_normalized"] == "Family and Child Development"
    assert normalized_rows[1]["department_normalized"] == "Family and Child Development"


def test_department_normalization_applies_approved_overrides() -> None:
    rows = [
        {"department": "Environmental Science and Engineering", "title": "t0"},
        {"department": "Environmental Sciences and Engineering", "title": "t1"},
    ]

    normalized_rows, _ = MODULE.add_normalized_department_column(rows)

    assert normalized_rows[0]["department_normalized"] == "Environmental Sciences and Engineering"
    assert normalized_rows[1]["department_normalized"] == "Environmental Sciences and Engineering"


def test_department_normalization_applies_approved_candidate_merges() -> None:
    rows = [
        {"department": "Forestry and Forest Products", "title": "t0"},
        {"department": "Forestry and Forestry Products", "title": "t1"},
        {"department": "Career Counseling and Student Personnel", "title": "t2"},
        {"department": "Counseling and Student Personnel Services", "title": "t3"},
    ]

    normalized_rows, _ = MODULE.add_normalized_department_column(rows)

    assert normalized_rows[0]["department_normalized"] == "Forestry and Forest Products"
    assert normalized_rows[1]["department_normalized"] == "Forestry and Forest Products"
    assert normalized_rows[2]["department_normalized"] == "Counseling and Student Personnel Services"
    assert normalized_rows[3]["department_normalized"] == "Counseling and Student Personnel Services"


def test_department_normalization_strips_major_phrases_to_base_unit() -> None:
    rows = [
        {"department": "Business with a major in Accounting", "title": "t0"},
        {"department": "Business and a major in Accounting", "title": "t1"},
        {"department": "Business (Accounting)", "title": "t2"},
        {"department": "Business", "title": "t3"},
    ]

    normalized_rows, _ = MODULE.add_normalized_department_column(rows)

    assert all(
        row["department_normalized"] == "Business"
        for row in normalized_rows
    )


def test_summarize_departments_counts_normalized_labels() -> None:
    rows = [
        {
            "department": "Human Nutrition & Foods",
            "department_normalized": "Human Nutrition and Foods",
            "source_set": "doctoral",
        },
        {
            "department": "Human Nutrition and Foods",
            "department_normalized": "Human Nutrition and Foods",
            "source_set": "masters",
        },
        {
            "department": "",
            "department_normalized": MODULE.MISSING_DEPARTMENT_LABEL,
            "source_set": "masters",
        },
    ]

    summary_rows = MODULE.summarize_departments(rows)

    assert summary_rows[0]["department_normalized"] == "Human Nutrition and Foods"
    assert summary_rows[0]["record_count"] == 2
    assert summary_rows[0]["doctoral_count"] == 1
    assert summary_rows[0]["masters_count"] == 1
    assert summary_rows[0]["raw_department_variant_count"] == 2
    assert summary_rows[1]["department_normalized"] == MODULE.MISSING_DEPARTMENT_LABEL
    assert summary_rows[1]["record_count"] == 1


def test_summarize_departments_adds_vt_major_reference_match() -> None:
    rows = [
        {
            "department": "Business with a major in Accounting",
            "department_normalized": "Business",
            "source_set": "doctoral",
        },
    ]

    summary_rows = MODULE.summarize_departments(
        rows,
        vt_majors=["Business", "Accounting"],
    )

    assert summary_rows[0]["department_normalized"] == "Business"
    assert summary_rows[0]["vt_major_match"] == "Business"
    assert summary_rows[0]["vt_major_match_score"] == 1.0
    assert summary_rows[0]["vt_major_match_type"] == "exact"


def test_build_department_mapping_audit_records_mapping_source() -> None:
    rows = [
        {
            "department": "Environmental Science and Engineering",
            "source_set": "doctoral",
        },
        {
            "department": "Computer Science and#38; Applications",
            "source_set": "masters",
        },
        {
            "department": "",
            "source_set": "masters",
        },
    ]

    _, canonical_map = MODULE.add_normalized_department_column(rows)
    audit_rows = MODULE.build_department_mapping_audit(
        rows,
        canonical_map,
        vt_majors=["Environmental Science", "Computer Science and Applications"],
    )
    by_raw = {row["raw_department"]: row for row in audit_rows}

    assert by_raw["Environmental Science and Engineering"]["canonical_department"] == (
        "Environmental Sciences and Engineering"
    )
    assert by_raw["Environmental Science and Engineering"]["mapping_source"] == "approved_override"
    assert by_raw["Environmental Science and Engineering"]["vt_major_match"] == ""
    assert by_raw["Environmental Science and Engineering"]["vt_major_match_type"] == "none"
    assert by_raw["Computer Science and#38; Applications"]["canonical_department"] == (
        "Computer Science and Applications"
    )
    assert by_raw["Computer Science and#38; Applications"]["mapping_source"] == "rule_based_cleanup"
    assert (
        by_raw["Computer Science and#38; Applications"]["vt_major_match"]
        == "Computer Science and Applications"
    )
    assert by_raw[""]["canonical_department"] == MODULE.MISSING_DEPARTMENT_LABEL
