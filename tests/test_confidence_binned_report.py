from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from sdg_distillation.reporting.confidence_bins import (
    build_report,
    melt_predictions,
    render_latex_table,
    summarize_condition_bins,
)


def test_melt_predictions_and_bin_summary(tmp_path: Path) -> None:
    predictions_df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "pred_prob_sdg01": [0.52, 0.85],
            "pred_label_sdg01": [1, 1],
            "teacher_prob_sdg01": [0.51, 0.90],
            "hard_target_sdg01": [1, 1],
            "pred_prob_sdg02": [0.49, 0.20],
            "pred_label_sdg02": [0, 0],
            "teacher_prob_sdg02": [0.49, 0.10],
            "hard_target_sdg02": [0, 0],
        }
    )
    long_df = melt_predictions(predictions_df)
    assert len(long_df) == 4
    summary_df = summarize_condition_bins(
        long_df,
        condition_key="plain",
        label="Plain",
        prediction_threshold=0.5,
        bin_edges=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    assert set(summary_df["confidence_bin"]) == {"[0.00, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.00]"}
    populated = summary_df[summary_df["n_instances"] > 0]
    assert len(populated) == 2
    assert populated["soft_bce_p1"].notna().all()


def test_render_latex_table_bolds_per_bin_winners() -> None:
    rows_df = pd.DataFrame(
        [
            {
                "label": "Plain BCE",
                "confidence_bin": "[0.00, 0.25)",
                "bin_index": 0,
                "n_instances": 100,
                "mean_teacher_confidence": 0.12,
                "soft_bce_p1": 0.1000,
                "brier_p1": 0.0200,
                "f1": 0.7000,
                "auroc": 0.9000,
            },
            {
                "label": "Weighted BCE",
                "confidence_bin": "[0.00, 0.25)",
                "bin_index": 0,
                "n_instances": 100,
                "mean_teacher_confidence": 0.12,
                "soft_bce_p1": 0.1100,
                "brier_p1": 0.0300,
                "f1": 0.6900,
                "auroc": 0.8900,
            },
            {
                "label": "Plain BCE",
                "confidence_bin": "[0.25, 0.50)",
                "bin_index": 1,
                "n_instances": 120,
                "mean_teacher_confidence": 0.33,
                "soft_bce_p1": 0.2100,
                "brier_p1": 0.0500,
                "f1": 0.8100,
                "auroc": 0.9400,
            },
            {
                "label": "Weighted BCE",
                "confidence_bin": "[0.25, 0.50)",
                "bin_index": 1,
                "n_instances": 120,
                "mean_teacher_confidence": 0.33,
                "soft_bce_p1": 0.1900,
                "brier_p1": 0.0400,
                "f1": 0.8300,
                "auroc": 0.9500,
            },
        ]
    )

    latex = render_latex_table(rows_df, caption="Confidence bins", label="tab:confidence_bins")

    assert "\\textbf{0.1000}" in latex
    assert "\\textbf{0.0200}" in latex
    assert "\\textbf{0.7000}" in latex
    assert "\\textbf{0.9000}" in latex
    assert "\\textbf{0.1900}" in latex
    assert "\\textbf{0.0400}" in latex
    assert "\\textbf{0.8300}" in latex
    assert "\\textbf{0.9500}" in latex


def test_build_report_writes_outputs(tmp_path: Path) -> None:
    report_dir = tmp_path / "eval" / "plain"
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "pred_prob_sdg01": [0.6, 0.2],
            "pred_label_sdg01": [1, 0],
            "teacher_prob_sdg01": [0.7, 0.1],
            "hard_target_sdg01": [1, 0],
        }
    )
    predictions_df.to_parquet(report_dir / "predictions.parquet", index=False)

    config = {
        "output": {"output_dir": str(tmp_path / "out"), "report_name": "bins"},
        "analysis": {"prediction_threshold": 0.5, "bin_edges": [0.0, 0.5, 1.0]},
        "latex": {"caption": "Confidence bins", "label": "tab:confidence_bins"},
        "conditions": [
            {
                "key": "plain",
                "label": "Plain",
                "report_dir": str(report_dir),
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    rows_df, out_dir = build_report(config_path)

    assert len(rows_df) == 2
    assert (out_dir / "confidence_bin_summary.csv").exists()
    assert (out_dir / "confidence_bin_summary.md").exists()
    assert (out_dir / "confidence_bin_summary.json").exists()
    assert (out_dir / "confidence_bin_summary.tex").exists()
    assert (out_dir / "manifest.json").exists()
