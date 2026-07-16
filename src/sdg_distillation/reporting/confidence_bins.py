from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

METRIC_DIRECTIONS = {
    "soft_bce_p1": "min",
    "brier_p1": "min",
    "f1": "max",
    "auroc": "max",
}

LATEX_WINNER_COLUMNS = ("soft_bce_p1", "brier_p1", "f1", "auroc")


@dataclass(frozen=True)
class ConditionReport:
    key: str
    label: str
    report_dir: Path


@dataclass(frozen=True)
class ConfidenceBinConfig:
    output_dir: Path
    report_name: str
    prediction_threshold: float
    bin_edges: tuple[float, ...]
    latex_caption: str
    latex_label: str
    conditions: tuple[ConditionReport, ...]


REQUIRED_PREDICTION_PREFIXES = ("pred_prob_", "teacher_prob_", "hard_target_")


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {path}")
    return raw


def _normalize_condition(raw: dict[str, Any]) -> ConditionReport:
    return ConditionReport(
        key=str(raw["key"]),
        label=str(raw["label"]),
        report_dir=Path(str(raw["report_dir"])),
    )


def load_config(path: Path) -> ConfidenceBinConfig:
    raw = _load_yaml(path)
    output_cfg = raw["output"]
    analysis_cfg = raw["analysis"]
    latex_cfg = raw.get("latex", {})
    bin_edges = tuple(float(x) for x in analysis_cfg["bin_edges"])
    if len(bin_edges) < 2:
        raise ValueError("analysis.bin_edges must contain at least two edges")
    if any(b2 <= b1 for b1, b2 in zip(bin_edges, bin_edges[1:])):
        raise ValueError("analysis.bin_edges must be strictly increasing")
    return ConfidenceBinConfig(
        output_dir=Path(str(output_cfg["output_dir"])),
        report_name=str(output_cfg["report_name"]),
        prediction_threshold=float(analysis_cfg.get("prediction_threshold", 0.5)),
        bin_edges=bin_edges,
        latex_caption=str(
            latex_cfg.get(
                "caption",
                "Held-out confidence-binned comparison across teacher-confidence regions for the retained stable student conditions.",
            )
        ),
        latex_label=str(latex_cfg.get("label", "tab:heldout_confidence_bins")),
        conditions=tuple(_normalize_condition(item) for item in raw["conditions"]),
    )


def _label_names(df: pd.DataFrame) -> list[str]:
    labels = sorted(col[len("pred_prob_") :] for col in df.columns if col.startswith("pred_prob_"))
    if not labels:
        raise ValueError("predictions dataframe does not contain pred_prob_* columns")
    for label in labels:
        for prefix in REQUIRED_PREDICTION_PREFIXES:
            col = f"{prefix}{label}"
            if col not in df.columns:
                raise ValueError(f"predictions dataframe missing required column {col!r}")
    return labels


def _clip_probs(values: np.ndarray, *, eps: float = 1e-7) -> np.ndarray:
    return np.clip(values.astype(np.float64), eps, 1.0 - eps)


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)

    i = 0
    while i < len(y_score):
        j = i + 1
        while j < len(y_score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return float((sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg))


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _bin_edges_for_cut(bin_edges: tuple[float, ...]) -> list[float]:
    edges = list(bin_edges)
    edges[-1] = float(np.nextafter(edges[-1], math.inf))
    return edges


def _bin_labels(bin_edges: tuple[float, ...]) -> list[str]:
    labels: list[str] = []
    for idx, (lo, hi) in enumerate(zip(bin_edges, bin_edges[1:])):
        right = "]" if idx == len(bin_edges) - 2 else ")"
        labels.append(f"[{lo:.2f}, {hi:.2f}{right}")
    return labels


def melt_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    labels = _label_names(predictions_df)
    pieces: list[pd.DataFrame] = []
    for label in labels:
        piece = pd.DataFrame(
            {
                "sample_id": predictions_df["sample_id"].astype(str),
                "label": label,
                "pred_prob": predictions_df[f"pred_prob_{label}"].astype(float),
                "teacher_prob": predictions_df[f"teacher_prob_{label}"].astype(float),
                "hard_target": predictions_df[f"hard_target_{label}"].astype(int),
            }
        )
        pieces.append(piece)
    long_df = pd.concat(pieces, ignore_index=True)
    teacher_prob = long_df["teacher_prob"].to_numpy(dtype=np.float64)
    pred_prob = long_df["pred_prob"].to_numpy(dtype=np.float64)
    long_df["teacher_confidence"] = 2.0 * np.abs(teacher_prob - 0.5)
    clipped_teacher = _clip_probs(teacher_prob)
    long_df["teacher_abs_logit"] = np.abs(np.log(clipped_teacher / (1.0 - clipped_teacher)))
    long_df["pred_label"] = (pred_prob >= 0.5).astype(int)
    return long_df


def summarize_condition_bins(
    long_df: pd.DataFrame,
    *,
    condition_key: str,
    label: str,
    prediction_threshold: float,
    bin_edges: tuple[float, ...],
) -> pd.DataFrame:
    df = long_df.copy()
    df["pred_label"] = (df["pred_prob"].to_numpy(dtype=np.float64) >= float(prediction_threshold)).astype(int)

    labels = _bin_labels(bin_edges)
    df["confidence_bin"] = pd.cut(
        df["teacher_confidence"],
        bins=_bin_edges_for_cut(bin_edges),
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows: list[dict[str, Any]] = []
    for bin_index, bin_label in enumerate(labels):
        group = df[df["confidence_bin"] == bin_label]
        row: dict[str, Any] = {
            "condition_key": condition_key,
            "label": label,
            "confidence_bin": bin_label,
            "bin_index": bin_index,
            "bin_lower": float(bin_edges[bin_index]),
            "bin_upper": float(bin_edges[bin_index + 1]),
            "n_instances": int(len(group)),
        }
        if group.empty:
            for col in (
                "mean_teacher_confidence",
                "mean_teacher_abs_logit",
                "mean_teacher_prob",
                "mean_pred_prob",
                "soft_bce_p1",
                "brier_p1",
                "prevalence",
                "precision",
                "recall",
                "f1",
                "auroc",
            ):
                row[col] = float("nan")
            rows.append(row)
            continue

        teacher_prob = group["teacher_prob"].to_numpy(dtype=np.float64)
        pred_prob = group["pred_prob"].to_numpy(dtype=np.float64)
        hard_target = group["hard_target"].to_numpy(dtype=np.int64)
        pred_label = group["pred_label"].to_numpy(dtype=np.int64)
        clipped_pred = _clip_probs(pred_prob)

        soft_bce = -(teacher_prob * np.log(clipped_pred) + (1.0 - teacher_prob) * np.log(1.0 - clipped_pred)).mean()
        precision, recall, f1 = _precision_recall_f1(hard_target, pred_label)

        row.update(
            {
                "mean_teacher_confidence": float(group["teacher_confidence"].mean()),
                "mean_teacher_abs_logit": float(group["teacher_abs_logit"].mean()),
                "mean_teacher_prob": float(group["teacher_prob"].mean()),
                "mean_pred_prob": float(group["pred_prob"].mean()),
                "soft_bce_p1": float(soft_bce),
                "brier_p1": float(np.mean((pred_prob - teacher_prob) ** 2)),
                "prevalence": float(hard_target.mean()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auroc": _binary_auroc(hard_target, pred_prob),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def collect_confidence_bin_rows(cfg: ConfidenceBinConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition in cfg.conditions:
        pred_path = condition.report_dir / "predictions.parquet"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        predictions_df = pd.read_parquet(pred_path)
        long_df = melt_predictions(predictions_df)
        summary_df = summarize_condition_bins(
            long_df,
            condition_key=condition.key,
            label=condition.label,
            prediction_threshold=cfg.prediction_threshold,
            bin_edges=cfg.bin_edges,
        )
        frames.append(summary_df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["bin_index", "soft_bce_p1", "label"], kind="stable").reset_index(drop=True)


def _winner_values(rows_df: pd.DataFrame) -> dict[tuple[int, str], float]:
    winners: dict[tuple[int, str], float] = {}
    for bin_index, group in rows_df.groupby("bin_index", sort=True):
        for metric in LATEX_WINNER_COLUMNS:
            series = group[metric].dropna().astype(float)
            if series.empty:
                continue
            direction = METRIC_DIRECTIONS[metric]
            winners[(int(bin_index), metric)] = float(series.min()) if direction == "min" else float(series.max())
    return winners


def _is_winner(value: float, winner: float) -> bool:
    return math.isclose(float(value), float(winner), rel_tol=1e-12, abs_tol=1e-12)


def render_latex_table(rows_df: pd.DataFrame, *, caption: str, label: str) -> str:
    winner_values = _winner_values(rows_df)
    lines = [
        r"\begin{table}[htb]",
        r"  \centering",
        f"  \\caption[{caption}]{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \scriptsize",
        r"  \begin{tabular}{@{} l l r c c c c c @{}}",
        r"    \toprule",
        r"    \textbf{Condition} & \textbf{Confidence Bin} & \textbf{N} & \textbf{Mean Conf.} & \textbf{Soft BCE$_{p_1}$} & \textbf{Brier$_{p_1}$} & \textbf{F1} & \textbf{AUROC} \\",
        r"    \midrule",
    ]
    for idx, (_, row) in enumerate(rows_df.iterrows()):
        bin_index = int(row["bin_index"])
        formatted = [
            row["label"],
            row["confidence_bin"],
            str(int(row["n_instances"])),
            f"{float(row['mean_teacher_confidence']):.3f}" if pd.notna(row["mean_teacher_confidence"]) else "",
        ]
        for metric in LATEX_WINNER_COLUMNS:
            value = row[metric]
            cell = "" if value is None or pd.isna(value) else f"{float(value):.4f}"
            winner = winner_values.get((bin_index, metric))
            if cell and winner is not None and _is_winner(value, winner):
                cell = f"\\textbf{{{cell}}}"
            formatted.append(cell)
        lines.append("    " + " & ".join(formatted) + r" \\")
        next_bin = int(rows_df.iloc[idx + 1]["bin_index"]) if idx + 1 < len(rows_df) else None
        if next_bin is not None and next_bin != bin_index:
            lines.append(r"    \addlinespace")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def write_report(cfg: ConfidenceBinConfig, rows_df: pd.DataFrame) -> Path:
    out_dir = cfg.output_dir / cfg.report_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "confidence_bin_summary.csv"
    md_path = out_dir / "confidence_bin_summary.md"
    json_path = out_dir / "confidence_bin_summary.json"
    tex_path = out_dir / "confidence_bin_summary.tex"
    manifest_path = out_dir / "manifest.json"

    rows_df.to_csv(csv_path, index=False)
    md_path.write_text(rows_df.to_markdown(index=False), encoding="utf-8")
    json_path.write_text(rows_df.to_json(orient="records", indent=2), encoding="utf-8")
    tex_path.write_text(
        render_latex_table(rows_df, caption=cfg.latex_caption, label=cfg.latex_label),
        encoding="utf-8",
    )

    manifest = {
        "prediction_threshold": cfg.prediction_threshold,
        "bin_edges": list(cfg.bin_edges),
        "latex_caption": cfg.latex_caption,
        "latex_label": cfg.latex_label,
        "conditions": [
            {"key": c.key, "label": c.label, "report_dir": str(c.report_dir)} for c in cfg.conditions
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def build_report(config_path: Path) -> tuple[pd.DataFrame, Path]:
    cfg = load_config(config_path)
    rows_df = collect_confidence_bin_rows(cfg)
    out_dir = write_report(cfg, rows_df)
    return rows_df, out_dir
