from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from sdg_distillation.reporting.etd_transfer_summary import (
    build_report,
    collect_condition_rows,
    load_report_config,
    render_latex_table,
)


def _write_summary(report_dir: Path, *, checkpoint_run_id: str, summary: dict[str, float]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_model_name": "microsoft/deberta-v3-base",
        "checkpoint_path": f"outputs/wandb/sdg-distillation/{checkpoint_run_id}/checkpoints/model.ckpt",
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_target_mode": "teacher_logit",
        "reference_path": "/tmp/test.csv",
        "inference_input_path": "/tmp/test.csv",
        "prediction_output_path": "/tmp/preds.parquet",
        "split": "test",
        "summary": summary,
    }
    (report_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_report_collects_rows_and_writes_outputs(tmp_path: Path) -> None:
    test_dir = tmp_path / "etd_test"
    _write_summary(
        test_dir,
        checkpoint_run_id="plain1234",
        summary={
            "macro_auroc": 0.9544,
            "micro_auroc": 0.9613,
            "macro_f1": 0.6799,
            "micro_f1": 0.7225,
            "macro_precision": 0.7092,
            "micro_precision": 0.7316,
            "macro_recall": 0.6596,
            "micro_recall": 0.7135,
            "macro_ece": 0.0221,
            "micro_ece": 0.0213,
            "soft_mae_p1": 0.0677,
            "soft_rmse_p1": 0.1826,
            "num_samples": 8178,
            "num_labels": 17,
            "prediction_threshold": 0.5,
            "hard_label_threshold": 0.5,
            "label_cardinality_pred": 1.82,
            "label_cardinality_true": 1.87,
        },
    )

    config = {
        "output": {"output_dir": str(tmp_path / "out"), "report_name": "etd_report"},
        "latex": {"caption": "Synthetic ETD report", "label": "tab:etd_synth"},
        "conditions": [
            {"key": "etd_test", "label": "ETD test", "report_dir": str(test_dir)},
        ],
    }
    config_path = tmp_path / "etd.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary_df, out_dir = build_report(config_path)

    assert list(summary_df["condition_key"]) == ["etd_test"]
    assert summary_df.iloc[0]["checkpoint_run_id"] == "plain1234"
    assert (out_dir / "condition_summary.csv").exists()
    assert (out_dir / "condition_summary.md").exists()
    assert (out_dir / "condition_summary.json").exists()
    tex = (out_dir / "condition_summary.tex").read_text(encoding="utf-8")
    assert "ETD test" in tex
    assert "\\label{tab:etd_synth}" in tex


def test_render_latex_table_uses_expected_values_for_single_row() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "label": "ETD test",
                "macro_auroc": 0.9544,
                "micro_auroc": 0.9613,
                "macro_f1": 0.6799,
                "micro_f1": 0.7225,
                "macro_precision": 0.7092,
                "micro_precision": 0.7316,
                "macro_recall": 0.6596,
                "micro_recall": 0.7135,
            }
        ]
    )
    latex = render_latex_table(summary_df, caption="ETD", label="tab:etd")
    assert "ETD test" in latex
    assert "0.9544" in latex
    assert "\\label{tab:etd}" in latex
    assert "\\textbf{0.9544}" not in latex


def test_render_latex_table_bolds_winners_for_multi_row_comparison() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "label": "Zero-shot",
                "macro_auroc": 0.9544,
                "micro_auroc": 0.9613,
                "macro_f1": 0.6799,
                "micro_f1": 0.7225,
                "macro_precision": 0.7092,
                "micro_precision": 0.7316,
                "macro_recall": 0.6596,
                "micro_recall": 0.7135,
            },
            {
                "label": "Adapted",
                "macro_auroc": 0.9600,
                "micro_auroc": 0.9650,
                "macro_f1": 0.7000,
                "micro_f1": 0.7350,
                "macro_precision": 0.7200,
                "micro_precision": 0.7400,
                "macro_recall": 0.6800,
                "micro_recall": 0.7200,
            },
        ]
    )
    latex = render_latex_table(summary_df, caption="ETD", label="tab:etd")
    assert "\\textbf{0.9600}" in latex
    assert "\\textbf{0.7350}" in latex


def test_collect_condition_rows_raises_when_summary_is_missing(tmp_path: Path) -> None:
    config = {
        "output": {"output_dir": str(tmp_path / "out"), "report_name": "etd_report"},
        "latex": {"caption": "Synthetic ETD report", "label": "tab:etd_synth"},
        "conditions": [
            {"key": "etd_test", "label": "ETD test", "report_dir": str(tmp_path / "missing")},
        ],
    }
    config_path = tmp_path / "etd.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    try:
        collect_condition_rows(load_report_config(config_path))
    except FileNotFoundError as exc:
        assert str(tmp_path / "missing" / "summary.json") in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
