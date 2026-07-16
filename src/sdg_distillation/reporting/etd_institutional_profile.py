from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SDG_IDS = tuple(range(1, 18))


@dataclass(frozen=True)
class DepartmentSpec:
    department: str
    short_label: str
    family: str


@dataclass(frozen=True)
class ModelSpec:
    label: str
    report_dir: Path


@dataclass(frozen=True)
class InstitutionalProfileConfig:
    analysis_dir: Path
    report_name: str
    latex_caption: str
    latex_label: str
    prediction_threshold: float
    badge_threshold: float
    profile_sdg: int
    profile_sdg_name: str
    ranked_top_k: int
    model: ModelSpec
    selected_departments: tuple[DepartmentSpec, ...]


PROB_COLS = [f"student_prob_sdg{i:02d}" for i in SDG_IDS]
LABEL_COLS = [f"student_label_sdg{i:02d}" for i in SDG_IDS]


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {path}")
    return raw


def load_report_config(path: Path) -> InstitutionalProfileConfig:
    raw = _load_yaml(path)
    output_cfg = raw["output"]
    latex_cfg = raw["latex"]
    analysis_cfg = raw["analysis"]
    model_cfg = raw["model"]
    selected_departments = tuple(
        DepartmentSpec(
            department=str(item["department"]),
            short_label=str(item["short_label"]),
            family=str(item["family"]),
        )
        for item in analysis_cfg["selected_departments"]
    )
    profile_sdg = int(analysis_cfg["profile_sdg"])
    if profile_sdg not in SDG_IDS:
        raise ValueError(f"profile_sdg must be in 1..17, got {profile_sdg}")
    return InstitutionalProfileConfig(
        analysis_dir=Path(str(output_cfg["analysis_dir"])),
        report_name=str(output_cfg["report_name"]),
        latex_caption=str(latex_cfg["caption"]),
        latex_label=str(latex_cfg["label"]),
        prediction_threshold=float(analysis_cfg.get("prediction_threshold", 0.5)),
        badge_threshold=float(analysis_cfg["badge_threshold"]),
        profile_sdg=profile_sdg,
        profile_sdg_name=str(analysis_cfg.get("profile_sdg_name", f"SDG {profile_sdg}")),
        ranked_top_k=int(analysis_cfg.get("ranked_top_k", len(selected_departments))),
        model=ModelSpec(
            label=str(model_cfg["label"]),
            report_dir=Path(str(model_cfg["report_dir"])),
        ),
        selected_departments=selected_departments,
    )


def _load_predictions(report_dir: Path) -> pd.DataFrame:
    pred_path = report_dir / "student_predictions_merged.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_parquet(pred_path)
    required = {"sample_id", "department_normalized", *PROB_COLS, *LABEL_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction artifact missing columns: {', '.join(missing)}")
    return df


def _format_sdg_list(values: list[int], *, sep: str) -> str:
    if not values:
        return "--"
    return sep.join(str(v) for v in values)


def build_department_profile_summary(cfg: InstitutionalProfileConfig) -> pd.DataFrame:
    pred_df = _load_predictions(cfg.model.report_dir)
    pred_df["department_normalized"] = pred_df["department_normalized"].fillna("Missing department")

    rows: list[dict[str, Any]] = []
    order_map = {spec.department: idx for idx, spec in enumerate(cfg.selected_departments)}
    for spec in cfg.selected_departments:
        dept_df = pred_df.loc[pred_df["department_normalized"] == spec.department].copy()
        if dept_df.empty:
            raise ValueError(f"Selected department missing from prediction artifact: {spec.department}")

        mean_probs = {sdg: float(dept_df[f"student_prob_sdg{sdg:02d}"].mean()) for sdg in SDG_IDS}
        positive_rates = {sdg: float(dept_df[f"student_label_sdg{sdg:02d}"].mean()) for sdg in SDG_IDS}
        top3_sdgs = sorted(SDG_IDS, key=lambda sdg: (-mean_probs[sdg], sdg))[:3]
        badge_sdgs = [sdg for sdg in SDG_IDS if positive_rates[sdg] >= cfg.badge_threshold]
        mean_confidence = float((2.0 * (dept_df[PROB_COLS] - 0.5).abs()).to_numpy(dtype=float).mean())
        mean_predicted_cardinality = float(dept_df[LABEL_COLS].sum(axis=1).mean())

        row: dict[str, Any] = {
            "department_order": order_map[spec.department],
            "department_normalized": spec.department,
            "department_short_label": spec.short_label,
            "department_family": spec.family,
            "n_documents": int(len(dept_df.index)),
            "badge_threshold": float(cfg.badge_threshold),
            "profile_sdg": int(cfg.profile_sdg),
            "profile_sdg_name": cfg.profile_sdg_name,
            "profile_sdg_rate": positive_rates[cfg.profile_sdg],
            "mean_confidence": mean_confidence,
            "mean_predicted_cardinality": mean_predicted_cardinality,
            "badge_sdg_list": _format_sdg_list(badge_sdgs, sep=", "),
            "top3_sdg_list": _format_sdg_list(top3_sdgs, sep=" / "),
            "model_label": cfg.model.label,
            "report_dir": str(cfg.model.report_dir),
        }
        for sdg in SDG_IDS:
            row[f"sdg{sdg:02d}_mean_prob"] = mean_probs[sdg]
            row[f"sdg{sdg:02d}_rate"] = positive_rates[sdg]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("department_order", kind="stable").reset_index(drop=True)


def render_latex_table(summary_df: pd.DataFrame, *, caption: str, label: str, badge_threshold: float) -> str:
    badge_header = rf"\shortstack{{\textbf{{SDG badges}}\\($\rho={badge_threshold:.2f}$)}}"
    lines = [
        r"\begin{table}[tb]",
        r"  \centering",
        f"  \\caption[{caption}]{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \footnotesize",
        r"  \begin{tabular}{@{} p{0.31\linewidth} r p{0.20\linewidth} p{0.18\linewidth} r @{}}",
        r"    \toprule",
        "    " + " & ".join(
            [
                r"\textbf{Department}",
                r"\textbf{ETDs}",
                badge_header,
                r"\textbf{Top-3 SDGs}",
                r"\textbf{Mean conf.}",
            ]
        ) + r" \\",
        r"    \midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "    " + " & ".join(
                [
                    str(row["department_short_label"]),
                    str(int(row["n_documents"])),
                    str(row["badge_sdg_list"]),
                    str(row["top3_sdg_list"]),
                    f"{float(row['mean_confidence']):.2f}",
                ]
            ) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def _render_markdown(summary_df: pd.DataFrame, *, badge_threshold: float) -> str:
    display_df = summary_df[
        [
            "department_short_label",
            "n_documents",
            "badge_sdg_list",
            "top3_sdg_list",
            "mean_confidence",
            "mean_predicted_cardinality",
            "profile_sdg_rate",
        ]
    ].copy()
    display_df = display_df.rename(
        columns={
            "department_short_label": "department",
            "n_documents": "etds",
            "badge_sdg_list": f"badges_rho_{badge_threshold:.2f}",
            "top3_sdg_list": "top3_sdgs",
            "mean_confidence": "mean_confidence",
            "mean_predicted_cardinality": "mean_predicted_cardinality",
            "profile_sdg_rate": "profile_sdg_rate",
        }
    )
    return "# ETD institutional profile\n\n```\n" + display_df.to_string(index=False) + "\n```\n"


def write_report(cfg: InstitutionalProfileConfig, summary_df: pd.DataFrame) -> Path:
    out_dir = cfg.analysis_dir / cfg.report_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "department_profile_summary.csv"
    summary_json = out_dir / "department_profile_summary.json"
    summary_md = out_dir / "department_profile_summary.md"
    summary_tex = out_dir / "department_profile_summary.tex"
    manifest_path = out_dir / "manifest.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary_df, badge_threshold=cfg.badge_threshold), encoding="utf-8")
    summary_tex.write_text(
        render_latex_table(
            summary_df,
            caption=cfg.latex_caption,
            label=cfg.latex_label,
            badge_threshold=cfg.badge_threshold,
        ),
        encoding="utf-8",
    )

    manifest = {
        "report_name": cfg.report_name,
        "badge_threshold": cfg.badge_threshold,
        "profile_sdg": cfg.profile_sdg,
        "profile_sdg_name": cfg.profile_sdg_name,
        "model_label": cfg.model.label,
        "report_dir": str(cfg.model.report_dir),
        "selected_departments": [spec.department for spec in cfg.selected_departments],
        "artifacts": {
            "csv": str(summary_csv),
            "json": str(summary_json),
            "markdown": str(summary_md),
            "latex": str(summary_tex),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out_dir


def build_report(config_path: Path) -> tuple[pd.DataFrame, Path]:
    cfg = load_report_config(config_path)
    summary_df = build_department_profile_summary(cfg)
    out_dir = write_report(cfg, summary_df)
    return summary_df, out_dir
