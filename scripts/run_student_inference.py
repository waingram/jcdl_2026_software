#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sdg_distillation.inference import TextInferenceDataModule
from sdg_distillation.training.data_contract import sdg_target_columns
from sdg_distillation.training.evaluation import (
    build_prediction_frame,
    compute_multilabel_report,
    load_sdg_titles,
    write_evaluation_report,
)
from sdg_distillation.training.model import DistillationLitModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a distilled student checkpoint over an unlabeled corpus or over a teacher-wide "
            "reference artifact. If p1_* columns are present, also compute teacher-consistency metrics."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True, help="CSV or parquet file to score.")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/inference"))
    parser.add_argument("--report-name", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-run-id", type=str, default=None)
    parser.add_argument("--id-col", type=str, default="sample_id")
    parser.add_argument("--text-col", type=str, default="abstract")
    parser.add_argument("--title-col", type=str, default=None)
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--hard-label-threshold", type=float, default=0.5)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--sdgs-path", type=Path, default=Path("./configs/sdgs.yaml"))
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type for {path}")


def _resolve_checkpoint_path(checkpoint_path: Path | None, checkpoint_run_id: str | None) -> Path:
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        return checkpoint_path

    if checkpoint_run_id:
        ckpt_dir = Path("./outputs/wandb/sdg-distillation") / checkpoint_run_id / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(ckpt_dir)
        matches = sorted(ckpt_dir.glob("*.ckpt"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple checkpoints found for run_id={checkpoint_run_id!r}; set --checkpoint-path explicitly."
            )
        return matches[0]

    raise ValueError("Set either --checkpoint-path or --checkpoint-run-id.")


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _prediction_columns(prefix: str) -> list[str]:
    return [f"{prefix}_sdg{i:02d}" for i in range(1, 18)]


def main() -> None:
    args = parse_args()

    ckpt_path = _resolve_checkpoint_path(args.checkpoint_path, args.checkpoint_run_id)
    model = DistillationLitModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()

    input_df = _read_table(args.input_path).reset_index(drop=True)
    if args.id_col not in input_df.columns:
        raise ValueError(f"Missing id column {args.id_col!r} in {args.input_path}")
    if args.text_col not in input_df.columns:
        raise ValueError(f"Missing text column {args.text_col!r} in {args.input_path}")

    dm = TextInferenceDataModule(
        predict_path=args.input_path,
        tokenizer_name=str(model.model_name),
        id_col=args.id_col,
        text_col=args.text_col,
        title_col=args.title_col,
        include_title=bool(args.include_title),
        metadata_cols=[],
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        persistent_workers=bool(args.persistent_workers),
        pin_memory=bool(args.pin_memory),
    )
    dm.setup("predict")
    loader = dm.predict_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sample_ids: list[str] = []
    logits_parts: list[np.ndarray] = []
    probs_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with _autocast_context(device):
                logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            logits_parts.append(logits.detach().cpu().numpy())
            probs_parts.append(probs.detach().cpu().numpy())
            sample_ids.extend([str(x) for x in batch["sample_id"]])

    logits_np = np.concatenate(logits_parts, axis=0)
    probs_np = np.concatenate(probs_parts, axis=0)

    expected_ids = input_df[args.id_col].astype(str).tolist()
    if sample_ids != expected_ids:
        raise RuntimeError("Prediction sample order mismatch between dataloader and input dataframe")

    pred_hard = (probs_np >= float(args.prediction_threshold)).astype(np.int64)
    pred_df = pd.DataFrame({args.id_col: sample_ids})
    for i in range(17):
        label = f"sdg{i + 1:02d}"
        pred_df[f"student_logit_{label}"] = logits_np[:, i]
        pred_df[f"student_prob_{label}"] = probs_np[:, i]
        pred_df[f"student_label_{label}"] = pred_hard[:, i]

    merged_df = input_df.merge(pred_df, on=args.id_col, how="left", validate="one_to_one")

    report_name = args.report_name
    if report_name in (None, "", "null"):
        run_id = args.checkpoint_run_id or ckpt_path.parent.parent.name or ckpt_path.stem
        report_name = f"{run_id}__{args.input_path.stem}"

    output_dir = args.output_dir / str(report_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "student_predictions_merged.parquet"
    merged_df.to_parquet(merged_path, index=False)
    print(f"Wrote {merged_path}")

    teacher_prob_cols = sdg_target_columns("p1")
    if all(col in input_df.columns for col in teacher_prob_cols):
        teacher_probs = input_df[teacher_prob_cols].to_numpy(dtype=np.float32)
        hard_targets = (teacher_probs >= float(args.hard_label_threshold)).astype(np.int64)
        sdg_titles = load_sdg_titles(args.sdgs_path)
        summary, per_label_df = compute_multilabel_report(
            pred_probs=probs_np,
            teacher_probs=teacher_probs,
            hard_targets=hard_targets,
            prediction_threshold=float(args.prediction_threshold),
            calibration_bins=int(args.calibration_bins),
            sdg_titles=sdg_titles,
        )
        summary["prediction_threshold"] = float(args.prediction_threshold)
        summary["hard_label_threshold"] = float(args.hard_label_threshold)

        comparison_pred_df = build_prediction_frame(
            sample_ids=sample_ids,
            pred_probs=probs_np,
            hard_targets=hard_targets,
            teacher_probs=teacher_probs,
            prediction_threshold=float(args.prediction_threshold),
        )
        report_paths = write_evaluation_report(
            output_dir=args.output_dir,
            report_name=str(report_name),
            model_name=str(model.model_name),
            target_mode=str(model.target_mode),
            split=str(input_df["split"].iloc[0]) if "split" in input_df.columns and input_df["split"].nunique() == 1 else "predict",
            summary=summary,
            per_label_df=per_label_df,
            reference_path=str(args.input_path),
            checkpoint_path=str(ckpt_path),
            checkpoint_run_id=args.checkpoint_run_id or ckpt_path.parent.parent.name or ckpt_path.stem,
            predictions_df=comparison_pred_df,
            extra_payload={
                "inference_input_path": str(args.input_path),
                "prediction_output_path": str(merged_path),
            },
        )
        print(
            "Teacher-consistency summary: "
            f"macro_auroc={summary['macro_auroc']:.6f} "
            f"micro_auroc={summary['micro_auroc']:.6f} "
            f"macro_f1={summary['macro_f1']:.6f} "
            f"micro_f1={summary['micro_f1']:.6f}"
        )
        print(f"Wrote {report_paths['summary_path']}")
        print(f"Wrote {report_paths['per_label_path']}")

    manifest = {
        "input_path": str(args.input_path),
        "output_dir": str(output_dir),
        "prediction_output_path": str(merged_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_run_id": args.checkpoint_run_id,
        "model_name": str(model.model_name),
        "target_mode": str(model.target_mode),
        "id_col": args.id_col,
        "text_col": args.text_col,
        "title_col": args.title_col,
        "include_title": bool(args.include_title),
        "prediction_threshold": float(args.prediction_threshold),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
