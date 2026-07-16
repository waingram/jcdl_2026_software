from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from sdg_distillation.reporting.factorial_ablation import build_report, render_latex_table, summarize_condition_rows


def _write_run(
    root: Path,
    *,
    run_dir_name: str,
    group: str,
    run_name: str,
    seed: int,
    target_mode: str,
    loss_mode: str,
    teacher_temperature: float,
    confidence_threshold: float,
    confidence_gamma: float,
    train_path: str,
    val_path: str,
    val_fraction: float,
    reference_val_path: str,
    monitor: str,
    summary: dict[str, float],
) -> None:
    files_dir = root / run_dir_name / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "resolved_config": {
            "value": {
                "seed": seed,
                "target_mode": target_mode,
                "data": {
                    "train_path": train_path,
                    "val_path": val_path,
                    "val_fraction": val_fraction,
                },
                "distillation": {
                    "loss_mode": loss_mode,
                    "teacher_temperature": teacher_temperature,
                    "confidence_threshold": confidence_threshold,
                    "confidence_gamma": confidence_gamma,
                },
                "evaluation": {
                    "reference_val_path": reference_val_path,
                },
                "trainer": {
                    "monitor": monitor,
                },
                "logging": {
                    "group": group,
                    "name": run_name,
                },
            }
        }
    }
    (files_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (files_dir / "wandb-summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_build_report_aggregates_and_writes_outputs(tmp_path: Path) -> None:
    runs_root = tmp_path / "wandb"
    summary_a = {
        "val_soft_bce_p1": 0.10,
        "val_brier_p1": 0.02,
        "val_micro_f1": 0.90,
        "val_macro_f1": 0.80,
        "val_micro_auroc": 0.95,
        "val_macro_auroc": 0.94,
    }
    summary_b = {
        "val_soft_bce_p1": 0.12,
        "val_brier_p1": 0.03,
        "val_micro_f1": 0.88,
        "val_macro_f1": 0.79,
        "val_micro_auroc": 0.96,
        "val_macro_auroc": 0.95,
    }
    summary_c = {
        "val_soft_bce_p1": 0.20,
        "val_brier_p1": 0.05,
        "val_micro_f1": 0.70,
        "val_macro_f1": 0.60,
        "val_micro_auroc": 0.85,
        "val_macro_auroc": 0.84,
    }
    summary_d = {
        "val_soft_bce_p1": 0.18,
        "val_brier_p1": 0.04,
        "val_micro_f1": 0.72,
        "val_macro_f1": 0.62,
        "val_micro_auroc": 0.86,
        "val_macro_auroc": 0.85,
    }

    common = {
        "group": "factorial_ablation_sbce_split_v1",
        "target_mode": "teacher_logit",
        "loss_mode": "plain_bce",
        "teacher_temperature": 1.0,
        "confidence_threshold": 0.0,
        "confidence_gamma": 1.0,
        "train_path": "/tmp/train.parquet",
        "val_path": "/tmp/val.parquet",
        "val_fraction": 0.0,
        "reference_val_path": "/tmp/val.parquet",
        "monitor": "val_soft_bce_p1",
    }
    _write_run(runs_root, run_dir_name="run-20260316_000001-aaa", run_name="sbce_split_plain_bce_t1_v1__s17", seed=17, summary=summary_a, **common)
    _write_run(runs_root, run_dir_name="run-20260316_000002-bbb", run_name="sbce_split_plain_bce_t1_v1__s23", seed=23, summary=summary_b, **common)
    common_bad = dict(common)
    common_bad["loss_mode"] = "masked_bce"
    common_bad["confidence_threshold"] = 0.1
    common_bad["confidence_gamma"] = 1.0
    _write_run(runs_root, run_dir_name="run-20260316_000003-ccc", run_name="sbce_split_masked_bce_t1_v1__s17", seed=17, summary=summary_c, **common_bad)
    _write_run(runs_root, run_dir_name="run-20260316_000004-ddd", run_name="sbce_split_masked_bce_t1_v1__s23", seed=23, summary=summary_d, **common_bad)

    config = {
        "output": {"output_dir": str(tmp_path / "out"), "report_name": "factorial_report"},
        "source": {"runs_root": str(runs_root), "group": "factorial_ablation_sbce_split_v1", "duplicate_policy": "latest"},
        "latex": {"caption": "Synthetic report", "label": "tab:synthetic"},
        "conditions": [
            {
                "key": "plain_bce_t1_v1",
                "label": "Plain BCE (T=1.0)",
                "run_name_prefix": "sbce_split_plain_bce_t1_v1__s",
                "seeds": [17, 23],
                "expectations": {
                    "target_mode": "teacher_logit",
                    "loss_mode": "plain_bce",
                    "teacher_temperature": 1.0,
                    "confidence_threshold": 0.0,
                    "confidence_gamma": 1.0,
                    "train_path": "/tmp/train.parquet",
                    "val_path": "/tmp/val.parquet",
                    "val_fraction": 0.0,
                    "reference_val_path": "/tmp/val.parquet",
                    "monitor": "val_soft_bce_p1",
                },
            },
            {
                "key": "masked_bce_t1_v1",
                "label": "Masked BCE (T=1.0)",
                "run_name_prefix": "sbce_split_masked_bce_t1_v1__s",
                "seeds": [17, 23],
                "expectations": {
                    "target_mode": "teacher_logit",
                    "loss_mode": "masked_bce",
                    "teacher_temperature": 1.0,
                    "confidence_threshold": 0.1,
                    "confidence_gamma": 1.0,
                    "train_path": "/tmp/train.parquet",
                    "val_path": "/tmp/val.parquet",
                    "val_fraction": 0.0,
                    "reference_val_path": "/tmp/val.parquet",
                    "monitor": "val_soft_bce_p1",
                },
            },
        ],
    }
    config_path = tmp_path / "report.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    per_run_df, summary_df, out_dir = build_report(config_path)

    assert len(per_run_df) == 4
    assert list(summary_df["condition_key"]) == ["plain_bce_t1_v1", "masked_bce_t1_v1"]
    assert summary_df.iloc[0]["val_soft_bce_p1_mean"] < summary_df.iloc[1]["val_soft_bce_p1_mean"]
    assert (out_dir / "per_run_metrics.csv").exists()
    assert (out_dir / "condition_summary.csv").exists()
    assert (out_dir / "condition_summary.md").exists()
    assert (out_dir / "condition_summary.json").exists()
    tex = (out_dir / "condition_summary.tex").read_text(encoding="utf-8")
    assert "Plain BCE (T=1.0)" in tex
    assert "\\label{tab:synthetic}" in tex


def test_render_latex_table_includes_expected_rows() -> None:
    summary_df = pd.DataFrame(
        [
            {
                "label": "Plain BCE (T=1.0)",
                "val_soft_bce_p1_mean": 0.1,
                "val_soft_bce_p1_sd": 0.01,
                "val_brier_p1_mean": 0.02,
                "val_brier_p1_sd": 0.002,
                "val_micro_f1_mean": 0.9,
                "val_micro_f1_sd": 0.01,
                "val_macro_f1_mean": 0.8,
                "val_macro_f1_sd": 0.02,
                "val_micro_auroc_mean": 0.95,
                "val_micro_auroc_sd": 0.01,
                "val_macro_auroc_mean": 0.94,
                "val_macro_auroc_sd": 0.01,
            }
        ]
    )
    latex = render_latex_table(summary_df, caption="Caption", label="tab:test")
    assert "Plain BCE (T=1.0)" in latex
    assert "0.1000 $\\pm$ 0.0100" in latex
    assert "\\label{tab:test}" in latex


def test_build_report_ignores_unrelated_runs_with_missing_optional_sections(tmp_path: Path) -> None:
    runs_root = tmp_path / "wandb"
    files_dir = runs_root / "run-20260316_999999-zzz" / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "resolved_config": {
            "value": {
                "seed": 999,
                "logging": {
                    "group": "some_other_group",
                    "name": "unrelated_run",
                },
            }
        }
    }
    (files_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (files_dir / "wandb-summary.json").write_text(json.dumps({}), encoding="utf-8")

    common = {
        "group": "factorial_ablation_sbce_split_v1",
        "target_mode": "teacher_logit",
        "loss_mode": "plain_bce",
        "teacher_temperature": 1.0,
        "confidence_threshold": 0.0,
        "confidence_gamma": 1.0,
        "train_path": "/tmp/train.parquet",
        "val_path": "/tmp/val.parquet",
        "val_fraction": 0.0,
        "reference_val_path": "/tmp/val.parquet",
        "monitor": "val_soft_bce_p1",
    }
    _write_run(
        runs_root,
        run_dir_name="run-20260316_000001-aaa",
        run_name="sbce_split_plain_bce_t1_v1__s17",
        seed=17,
        summary={
            "val_soft_bce_p1": 0.10,
            "val_brier_p1": 0.02,
            "val_micro_f1": 0.90,
            "val_macro_f1": 0.80,
            "val_micro_auroc": 0.95,
            "val_macro_auroc": 0.94,
        },
        **common,
    )

    config = {
        "output": {"output_dir": str(tmp_path / "out"), "report_name": "factorial_report"},
        "source": {"runs_root": str(runs_root), "group": "factorial_ablation_sbce_split_v1", "duplicate_policy": "latest"},
        "latex": {"caption": "Synthetic report", "label": "tab:synthetic"},
        "conditions": [
            {
                "key": "plain_bce_t1_v1",
                "label": "Plain BCE (T=1.0)",
                "run_name_prefix": "sbce_split_plain_bce_t1_v1__s",
                "seeds": [17],
                "expectations": {
                    "target_mode": "teacher_logit",
                    "loss_mode": "plain_bce",
                    "teacher_temperature": 1.0,
                    "confidence_threshold": 0.0,
                    "confidence_gamma": 1.0,
                    "train_path": "/tmp/train.parquet",
                    "val_path": "/tmp/val.parquet",
                    "val_fraction": 0.0,
                    "reference_val_path": "/tmp/val.parquet",
                    "monitor": "val_soft_bce_p1",
                },
            }
        ],
    }
    config_path = tmp_path / "report.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    per_run_df, summary_df, _ = build_report(config_path)

    assert len(per_run_df) == 1
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["condition_key"] == "plain_bce_t1_v1"
