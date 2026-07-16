from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

METRIC_COLUMNS = [
    "val_soft_bce_p1",
    "val_brier_p1",
    "val_micro_f1",
    "val_macro_f1",
    "val_micro_auroc",
    "val_macro_auroc",
]

METRIC_DIRECTIONS = {
    "val_soft_bce_p1": "min",
    "val_brier_p1": "min",
    "val_micro_f1": "max",
    "val_macro_f1": "max",
    "val_micro_auroc": "max",
    "val_macro_auroc": "max",
}

LATEX_COLUMN_LABELS = {
    "label": "Condition",
    "val_soft_bce_p1": "Soft BCE$_{p_1}$",
    "val_brier_p1": "Brier$_{p_1}$",
    "val_micro_f1": "Micro F1",
    "val_macro_f1": "Macro F1",
    "val_micro_auroc": "Micro AUROC",
    "val_macro_auroc": "Macro AUROC",
}

PRIMARY_MAIN_TEXT_METRICS = [
    "val_soft_bce_p1",
    "val_brier_p1",
]

SECONDARY_MAIN_TEXT_METRICS = [
    "val_micro_f1",
    "val_macro_f1",
    "val_micro_auroc",
    "val_macro_auroc",
]


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    run_name_prefix: str
    seeds: tuple[int, ...]
    expectations: dict[str, Any]


@dataclass(frozen=True)
class RunRecord:
    run_name: str
    run_dir: Path
    group: str | None
    seed: int | None
    target_mode: str | None
    loss_mode: str | None
    teacher_temperature: float | None
    confidence_threshold: float | None
    confidence_gamma: float | None
    train_path: str | None
    val_path: str | None
    val_fraction: float | None
    reference_val_path: str | None
    monitor: str | None
    metrics: dict[str, float | None]


@dataclass(frozen=True)
class FactorialReportConfig:
    output_dir: Path
    report_name: str
    runs_root: Path
    group: str
    duplicate_policy: str
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


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_condition(raw: dict[str, Any]) -> ConditionSpec:
    return ConditionSpec(
        key=str(raw["key"]),
        label=str(raw["label"]),
        run_name_prefix=str(raw["run_name_prefix"]),
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        expectations=dict(raw.get("expectations", {})),
    )


def load_report_config(path: Path) -> FactorialReportConfig:
    raw = _load_yaml(path)
    output_cfg = raw["output"]
    source_cfg = raw["source"]
    latex_cfg = raw["latex"]
    conditions = tuple(_normalize_condition(item) for item in raw["conditions"])
    return FactorialReportConfig(
        output_dir=Path(str(output_cfg["output_dir"])),
        report_name=str(output_cfg["report_name"]),
        runs_root=Path(str(source_cfg["runs_root"])),
        group=str(source_cfg["group"]),
        duplicate_policy=str(source_cfg.get("duplicate_policy", "latest")),
        latex_caption=str(latex_cfg["caption"]),
        latex_label=str(latex_cfg["label"]),
        conditions=conditions,
    )


def _load_run_record(run_dir: Path) -> RunRecord:
    cfg = _load_yaml(run_dir / "files" / "config.yaml")
    summary = json.loads((run_dir / "files" / "wandb-summary.json").read_text(encoding="utf-8"))
    resolved = cfg["resolved_config"]["value"]
    logging_cfg = resolved.get("logging", {})
    distill_cfg = resolved.get("distillation", {})
    data_cfg = resolved.get("data", {})
    eval_cfg = resolved.get("evaluation", {})
    trainer_cfg = resolved.get("trainer", {})

    metrics = {metric: _to_float(summary.get(metric)) for metric in METRIC_COLUMNS}
    return RunRecord(
        run_name=str(logging_cfg.get("name")),
        run_dir=run_dir,
        group=str(logging_cfg.get("group")) if logging_cfg.get("group") is not None else None,
        seed=_to_int(resolved.get("seed")),
        target_mode=str(resolved.get("target_mode")) if resolved.get("target_mode") is not None else None,
        loss_mode=str(distill_cfg.get("loss_mode")) if distill_cfg.get("loss_mode") is not None else None,
        teacher_temperature=_to_float(distill_cfg.get("teacher_temperature")),
        confidence_threshold=_to_float(distill_cfg.get("confidence_threshold")),
        confidence_gamma=_to_float(distill_cfg.get("confidence_gamma")),
        train_path=str(data_cfg.get("train_path")) if data_cfg.get("train_path") is not None else None,
        val_path=str(data_cfg.get("val_path")) if data_cfg.get("val_path") is not None else None,
        val_fraction=_to_float(data_cfg.get("val_fraction")),
        reference_val_path=str(eval_cfg.get("reference_val_path")) if eval_cfg.get("reference_val_path") is not None else None,
        monitor=str(trainer_cfg.get("monitor")) if trainer_cfg.get("monitor") is not None else None,
        metrics=metrics,
    )


def _iter_group_runs(cfg: FactorialReportConfig) -> list[RunRecord]:
    records: list[RunRecord] = []
    for run_dir in sorted(cfg.runs_root.glob("run-*")):
        config_path = run_dir / "files" / "config.yaml"
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        record = _load_run_record(run_dir)
        if record.group == cfg.group:
            records.append(record)
    return records


def _dedupe_records(records: list[RunRecord], duplicate_policy: str) -> list[RunRecord]:
    by_name: dict[str, list[RunRecord]] = {}
    for record in records:
        by_name.setdefault(record.run_name, []).append(record)
    deduped: list[RunRecord] = []
    for run_name, items in by_name.items():
        if len(items) == 1:
            deduped.append(items[0])
            continue
        items = sorted(items, key=lambda item: item.run_dir.name)
        if duplicate_policy == "error":
            raise ValueError(f"Duplicate run name in group: {run_name}")
        if duplicate_policy != "latest":
            raise ValueError(f"Unsupported duplicate_policy={duplicate_policy!r}")
        deduped.append(items[-1])
    return sorted(deduped, key=lambda item: item.run_name)


def _expected_run_name(prefix: str, seed: int) -> str:
    return f"{prefix}{seed}"


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        actual_float = _to_float(actual)
        return actual_float is not None and math.isclose(actual_float, expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def _record_expectation_map(record: RunRecord) -> dict[str, Any]:
    return {
        "target_mode": record.target_mode,
        "loss_mode": record.loss_mode,
        "teacher_temperature": record.teacher_temperature,
        "confidence_threshold": record.confidence_threshold,
        "confidence_gamma": record.confidence_gamma,
        "train_path": record.train_path,
        "val_path": record.val_path,
        "val_fraction": record.val_fraction,
        "reference_val_path": record.reference_val_path,
        "monitor": record.monitor,
    }


def collect_condition_rows(cfg: FactorialReportConfig) -> pd.DataFrame:
    records = _dedupe_records(_iter_group_runs(cfg), cfg.duplicate_policy)
    by_name = {record.run_name: record for record in records}
    rows: list[dict[str, Any]] = []

    for spec in cfg.conditions:
        for seed in spec.seeds:
            run_name = _expected_run_name(spec.run_name_prefix, seed)
            if run_name not in by_name:
                raise FileNotFoundError(f"Missing run for condition {spec.key}: {run_name}")
            record = by_name[run_name]
            actual = _record_expectation_map(record)
            for key, expected in spec.expectations.items():
                if key not in actual:
                    raise KeyError(f"Unknown expectation key {key!r}")
                if not _values_equal(actual[key], expected):
                    raise ValueError(
                        f"Expectation mismatch for {run_name}: {key} expected {expected!r}, got {actual[key]!r}"
                    )
            row = {
                "condition_key": spec.key,
                "label": spec.label,
                "seed": seed,
                "run_name": record.run_name,
                "run_dir": str(record.run_dir),
                "group": record.group,
                "target_mode": record.target_mode,
                "loss_mode": record.loss_mode,
                "teacher_temperature": record.teacher_temperature,
                "confidence_threshold": record.confidence_threshold,
                "confidence_gamma": record.confidence_gamma,
                "train_path": record.train_path,
                "val_path": record.val_path,
                "val_fraction": record.val_fraction,
                "reference_val_path": record.reference_val_path,
                "monitor": record.monitor,
            }
            row.update(record.metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_condition_rows(per_run_df: pd.DataFrame, metric_columns: list[str] | None = None) -> pd.DataFrame:
    metrics = metric_columns or METRIC_COLUMNS
    grouped = per_run_df.groupby(["condition_key", "label"], sort=False)
    rows: list[dict[str, Any]] = []
    for (condition_key, label), group in grouped:
        row: dict[str, Any] = {
            "condition_key": condition_key,
            "label": label,
            "n_runs": int(len(group)),
        }
        for metric in metrics:
            series = group[metric].astype(float)
            row[f"{metric}_mean"] = float(series.mean())
            row[f"{metric}_sd"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["val_soft_bce_p1_mean", "label"], kind="stable").reset_index(drop=True)
        summary_df["rank"] = range(1, len(summary_df) + 1)
    return summary_df


def _winner_values(summary_df: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    winners: dict[str, float] = {}
    for metric in metrics:
        series = summary_df[f"{metric}_mean"].dropna().astype(float)
        if series.empty:
            continue
        direction = METRIC_DIRECTIONS[metric]
        winners[metric] = float(series.min()) if direction == "min" else float(series.max())
    return winners


def _is_winner(value: float, winner: float) -> bool:
    return math.isclose(float(value), float(winner), rel_tol=1e-12, abs_tol=1e-12)


def _summary_subset(summary_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    columns = ["condition_key", "label", "n_runs", "rank"]
    for metric in metrics:
        columns.extend([f"{metric}_mean", f"{metric}_sd"])
    return summary_df[columns].copy()


def render_latex_table_for_metrics(
    summary_df: pd.DataFrame,
    *,
    metrics: list[str],
    caption: str,
    label: str,
) -> str:
    winner_values = _winner_values(summary_df, metrics)
    column_spec = "@{} l " + " ".join("c" for _ in metrics) + " @{}"
    header = " & ".join(
        [rf"\textbf{{{LATEX_COLUMN_LABELS['label']}}}"]
        + [rf"\textbf{{{LATEX_COLUMN_LABELS[metric]}}}" for metric in metrics]
    )
    lines = [
        r"\begin{table}[htb]",
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
            cell = f"{row[f'{metric}_mean']:.4f} $\\pm$ {row[f'{metric}_sd']:.4f}"
            winner = winner_values.get(metric)
            if winner is not None and _is_winner(row[f"{metric}_mean"], winner):
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
    return render_latex_table_for_metrics(summary_df, metrics=METRIC_COLUMNS, caption=caption, label=label)


def write_report(cfg: FactorialReportConfig, per_run_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    out_dir = cfg.output_dir / cfg.report_name
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_csv = out_dir / "per_run_metrics.csv"
    summary_csv = out_dir / "condition_summary.csv"
    primary_csv = out_dir / "condition_summary_primary.csv"
    secondary_csv = out_dir / "condition_summary_secondary.csv"
    summary_md = out_dir / "condition_summary.md"
    summary_json = out_dir / "condition_summary.json"
    summary_tex = out_dir / "condition_summary.tex"
    primary_tex = out_dir / "condition_summary_primary.tex"
    secondary_tex = out_dir / "condition_summary_secondary.tex"
    manifest_json = out_dir / "manifest.json"

    primary_summary_df = _summary_subset(summary_df, PRIMARY_MAIN_TEXT_METRICS)
    secondary_summary_df = _summary_subset(summary_df, SECONDARY_MAIN_TEXT_METRICS)

    per_run_df.to_csv(per_run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    primary_summary_df.to_csv(primary_csv, index=False)
    secondary_summary_df.to_csv(secondary_csv, index=False)
    summary_md.write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    summary_tex.write_text(
        render_latex_table(summary_df, caption=cfg.latex_caption, label=cfg.latex_label),
        encoding="utf-8",
    )
    primary_tex.write_text(
        render_latex_table_for_metrics(
            summary_df,
            metrics=PRIMARY_MAIN_TEXT_METRICS,
            caption=f"{cfg.latex_caption} Primary soft-fidelity metrics.",
            label=f"{cfg.latex_label}_primary",
        ),
        encoding="utf-8",
    )
    secondary_tex.write_text(
        render_latex_table_for_metrics(
            summary_df,
            metrics=SECONDARY_MAIN_TEXT_METRICS,
            caption=f"{cfg.latex_caption} Secondary thresholded proxy metrics.",
            label=f"{cfg.latex_label}_secondary",
        ),
        encoding="utf-8",
    )
    manifest = {
        "group": cfg.group,
        "runs_root": str(cfg.runs_root),
        "condition_count": len(cfg.conditions),
        "metric_columns": METRIC_COLUMNS,
        "main_text_tables": {
            "primary": {
                "metrics": PRIMARY_MAIN_TEXT_METRICS,
                "csv": primary_csv.name,
                "tex": primary_tex.name,
            },
            "secondary": {
                "metrics": SECONDARY_MAIN_TEXT_METRICS,
                "csv": secondary_csv.name,
                "tex": secondary_tex.name,
            },
        },
        "conditions": [
            {
                "key": spec.key,
                "label": spec.label,
                "run_name_prefix": spec.run_name_prefix,
                "seeds": list(spec.seeds),
                "expectations": spec.expectations,
            }
            for spec in cfg.conditions
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def build_report(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    cfg = load_report_config(config_path)
    per_run_df = collect_condition_rows(cfg)
    summary_df = summarize_condition_rows(per_run_df)
    out_dir = write_report(cfg, per_run_df, summary_df)
    return per_run_df, summary_df, out_dir
