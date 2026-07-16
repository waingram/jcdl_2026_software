from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from sdg_distillation.reporting.etd_institutional_profile import build_report


def _make_prediction_row(sample_id: str, department: str, *, sdg13_prob: float, sdg04_prob: float) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "department_normalized": department,
    }
    for sdg in range(1, 18):
        prob = 0.01
        if sdg == 13:
            prob = sdg13_prob
        elif sdg == 4:
            prob = sdg04_prob
        row[f"student_prob_sdg{sdg:02d}"] = prob
        row[f"student_label_sdg{sdg:02d}"] = int(prob >= 0.5)
    return row


def test_build_report_writes_expected_outputs(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True)
    pred_df = pd.DataFrame(
        [
            _make_prediction_row("a1", "Dept A", sdg13_prob=0.90, sdg04_prob=0.10),
            _make_prediction_row("a2", "Dept A", sdg13_prob=0.85, sdg04_prob=0.15),
            _make_prediction_row("b1", "Dept B", sdg13_prob=0.10, sdg04_prob=0.80),
            _make_prediction_row("b2", "Dept B", sdg13_prob=0.20, sdg04_prob=0.75),
        ]
    )
    pred_df.to_parquet(report_dir / "student_predictions_merged.parquet", index=False)

    config = {
        "output": {"analysis_dir": str(tmp_path / "out"), "report_name": "inst_profile"},
        "latex": {"caption": "Synthetic profile", "label": "tab:synthetic_profile"},
        "analysis": {
            "prediction_threshold": 0.5,
            "badge_threshold": 0.15,
            "profile_sdg": 13,
            "profile_sdg_name": "Climate Action",
            "ranked_top_k": 2,
            "selected_departments": [
                {"department": "Dept A", "short_label": "Dept A", "family": "A"},
                {"department": "Dept B", "short_label": "Dept B", "family": "B"},
            ],
        },
        "model": {"label": "Adapted", "report_dir": str(report_dir)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary_df, out_dir = build_report(config_path)

    assert list(summary_df["department_short_label"]) == ["Dept A", "Dept B"]
    assert summary_df.loc[summary_df["department_short_label"] == "Dept A", "badge_sdg_list"].iloc[0] == "13"
    assert summary_df.loc[summary_df["department_short_label"] == "Dept B", "badge_sdg_list"].iloc[0] == "4"
    assert (out_dir / "department_profile_summary.csv").exists()
    assert (out_dir / "department_profile_summary.tex").exists()
    tex = (out_dir / "department_profile_summary.tex").read_text(encoding="utf-8")
    assert "Dept A" in tex
    assert "\\label{tab:synthetic_profile}" in tex
