from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

METRIC_COLUMNS = [
    "macro_auroc",
    "micro_auroc",
    "macro_f1",
    "micro_f1",
    "macro_precision",
    "micro_precision",
    "macro_recall",
    "micro_recall",
    "macro_ece",
    "micro_ece",
    "soft_mae_p1",
    "soft_rmse_p1",
    "label_cardinality_pred",
    "label_cardinality_true",
]

LATEX_METRIC_COLUMNS = [
    "macro_auroc",
    "micro_auroc",
    "macro_f1",
    "micro_f1",
    "macro_precision",
    "micro_precision",
    "macro_recall",
    "micro_recall",
]

LATEX_COLUMN_LABELS = {
    "label": "Split",
    "macro_auroc": "Macro AUROC",
    "micro_auroc": "Micro AUROC",
    "macro_f1": "Macro F1",
    "micro_f1": "Micro F1",
    "macro_precision": "Macro Prec.",
    "micro_precision": "Micro Prec.",
    "macro_recall": "Macro Rec.",
    "micro_recall": "Micro Rec.",
}

DISCRIMINATION_MAIN_TEXT_METRICS = [
    "macro_auroc",
    "micro_auroc",
    "macro_f1",
    "micro_f1",
]

PRECISION_RECALL_MAIN_TEXT_METRICS = [
    "macro_precision",
    "micro_precision",
    "macro_recall",
    "micro_recall",
]

METRIC_DIRECTIONS = {
    "macro_auroc": "max",
    "micro_auroc": "max",
    "macro_f1": "max",
    "micro_f1": "max",
    "macro_precision": "max",
    "micro_precision": "max",
    "macro_recall": "max",
    "micro_recall": "max",
    "macro_ece": "min",
    "micro_ece": "min",
    "soft_mae_p1": "min",
    "soft_rmse_p1": "min",
    "label_cardinality_pred": "min",
    "label_cardinality_true": "min",
}


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    report_dir: Path


@dataclass(frozen=True)
class ETDTransferConfig:
    output_dir: Path
    report_name: str
    latex_caption: str
    latex_label: str
    conditions: tuple[ConditionSpec, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {path}")
    return raw


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _normalize_condition(raw: dict[str, Any]) -> ConditionSpec:
    return ConditionSpec(
        key=str(raw["key"]),
        label=str(raw["label"]),
        report_dir=Path(str(raw["report_dir"])),
    )


def load_report_config(path: Path) -> ETDTransferConfig:
    raw = _load_yaml(path)
    output_cfg = raw["output"]
    latex_cfg = raw["latex"]
    conditions = tuple(_normalize_condition(item) for item in raw["conditions"])
    return ETDTransferConfig(
        output_dir=Path(str(output_cfg["output_dir"])),
        report_name=str(output_cfg["report_name"]),
        latex_caption=str(latex_cfg["caption"]),
        latex_label=str(latex_cfg["label"]),
        conditions=conditions,
    )


def _load_summary_payload(report_dir: Path) -> dict[str, Any]:
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object payload in {summary_path}")
    return raw


def collect_condition_rows(cfg: ETDTransferConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in cfg.conditions:
        payload = _load_summary_payload(spec.report_dir)
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"Expected summary mapping in {spec.report_dir / 'summary.json'}")
        row: dict[str, Any] = {
            "condition_key": spec.key,
            "label": spec.label,
            "report_dir": str(spec.report_dir),
            "checkpoint_model_name": payload.get("checkpoint_model_name"),
            "checkpoint_path": payload.get("checkpoint_path"),
            "checkpoint_run_id": payload.get("checkpoint_run_id"),
            "checkpoint_target_mode": payload.get("checkpoint_target_mode"),
            "reference_path": payload.get("reference_path"),
            "inference_input_path": payload.get("inference_input_path"),
            "prediction_output_path": payload.get("prediction_output_path"),
            "split": payload.get("split"),
            "num_samples": _to_float(summary.get("num_samples")),
            "num_labels": _to_float(summary.get("num_labels")),
            "prediction_threshold": _to_float(summary.get("prediction_threshold")),
            "hard_label_threshold": _to_float(summary.get("hard_label_threshold")),
            "label_cardinality_pred": _to_float(summary.get("label_cardinality_pred")),
            "label_cardinality_true": _to_float(summary.get("label_cardinality_true")),
            "mean_pred_prob": _to_float(summary.get("mean_pred_prob")),
            "mean_teacher_prob": _to_float(summary.get("mean_teacher_prob")),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = _to_float(summary.get(metric))
        rows.append(row)
    return pd.DataFrame(rows)


def _winner_values(summary_df: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    if len(summary_df.index) <= 1:
        return {}
    winners: dict[str, float] = {}
    for metric in metrics:
        series = summary_df[metric].dropna().astype(float)
        if series.empty:
            continue
        direction = METRIC_DIRECTIONS[metric]
        winners[metric] = float(series.min()) if direction == "min" else float(series.max())
    return winners


def _is_winner(value: float, winner: float) -> bool:
    return math.isclose(float(value), float(winner), rel_tol=1e-12, abs_tol=1e-12)


def _summary_subset(summary_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    columns = ["condition_key", "label", "checkpoint_run_id"] + metrics
    return summary_df[columns].copy()


def render_latex_table_for_metrics(
    summary_df: pd.DataFrame,
    *,
    metrics: list[str],
    caption: str,
    label: str,
) -> str:
    winner_values = _winner_values(summary_df, metrics)
    column_spec = "@{} l " + " ".join("r" for _ in metrics) + " @{}"
    header = " & ".join(
        [rf"\textbf{{{LATEX_COLUMN_LABELS['label']}}}"]
        + [rf"\textbf{{{LATEX_COLUMN_LABELS[metric]}}}" for metric in metrics]
    )
    lines = [
        r"\begin{table}[tb]",
        r"  \centering",
        f"  \\caption[{caption}]{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \small",
        f"  \\begin{{tabular}}{{{column_spec}}}",
        r"    \toprule",
        "    " + header + r" \\",
        r"    \midrule",
    ]
    for _, row in summary_df.iterrows():
        formatted = [row["label"]]
        for metric in metrics:
            value = row[metric]
            cell = "" if value is None or pd.isna(value) else f"{float(value):.4f}"
            winner = winner_values.get(metric)
            if cell and winner is not None and _is_winner(value, winner):
                cell = f"\\textbf{{{cell}}}"
            formatted.append(cell)
        lines.append("    " + " & ".join(formatted) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def render_latex_table(summary_df: pd.DataFrame, *, caption: str, label: str) -> str:
    return render_latex_table_for_metrics(summary_df, metrics=LATEX_METRIC_COLUMNS, caption=caption, label=label)


def write_report(cfg: ETDTransferConfig, summary_df: pd.DataFrame) -> Path:
    out_dir = cfg.output_dir / cfg.report_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "condition_summary.csv"
    discrimination_csv = out_dir / "condition_summary_discrimination.csv"
    precision_recall_csv = out_dir / "condition_summary_precision_recall.csv"
    summary_md = out_dir / "condition_summary.md"
    summary_json = out_dir / "condition_summary.json"
    summary_tex = out_dir / "condition_summary.tex"
    discrimination_tex = out_dir / "condition_summary_discrimination.tex"
    precision_recall_tex = out_dir / "condition_summary_precision_recall.tex"
    manifest_json = out_dir / "manifest.json"

    discrimination_summary_df = _summary_subset(summary_df, DISCRIMINATION_MAIN_TEXT_METRICS)
    precision_recall_summary_df = _summary_subset(summary_df, PRECISION_RECALL_MAIN_TEXT_METRICS)

    summary_df.to_csv(summary_csv, index=False)
    discrimination_summary_df.to_csv(discrimination_csv, index=False)
    precision_recall_summary_df.to_csv(precision_recall_csv, index=False)
    summary_md.write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    summary_tex.write_text(
        render_latex_table(summary_df, caption=cfg.latex_caption, label=cfg.latex_label),
        encoding="utf-8",
    )
    discrimination_tex.write_text(
        render_latex_table_for_metrics(
            summary_df,
            metrics=DISCRIMINATION_MAIN_TEXT_METRICS,
            caption=f"{cfg.latex_caption} Discrimination and F1 metrics.",
            label=f"{cfg.latex_label}_discrimination",
        ),
        encoding="utf-8",
    )
    precision_recall_tex.write_text(
        render_latex_table_for_metrics(
            summary_df,
            metrics=PRECISION_RECALL_MAIN_TEXT_METRICS,
            caption=f"{cfg.latex_caption} Precision and recall metrics.",
            label=f"{cfg.latex_label}_precision_recall",
        ),
        encoding="utf-8",
    )
    manifest = {
        "condition_count": len(cfg.conditions),
        "metric_columns": METRIC_COLUMNS,
        "latex_metric_columns": LATEX_METRIC_COLUMNS,
        "main_text_tables": {
            "discrimination": {
                "metrics": DISCRIMINATION_MAIN_TEXT_METRICS,
                "csv": discrimination_csv.name,
                "tex": discrimination_tex.name,
            },
            "precision_recall": {
                "metrics": PRECISION_RECALL_MAIN_TEXT_METRICS,
                "csv": precision_recall_csv.name,
                "tex": precision_recall_tex.name,
            },
        },
        "conditions": [
            {
                "key": spec.key,
                "label": spec.label,
                "report_dir": str(spec.report_dir),
            }
            for spec in cfg.conditions
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def build_report(config_path: Path) -> tuple[pd.DataFrame, Path]:
    cfg = load_report_config(config_path)
    summary_df = collect_condition_rows(cfg)
    out_dir = write_report(cfg, summary_df)
    return summary_df, out_dir
