from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from .data_contract import sdg_target_columns
from .datamodule import DistillationDataModule
from .dataset import DistillationDataset
from .model import DistillationLitModule


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: str | None = None
    checkpoint_run_id: str | None = None
    split: str = "test"  # train | val | test
    reference_train_path: str | None = None
    reference_val_path: str | None = None
    reference_test_path: str | None = None
    prediction_threshold: float = 0.5
    hard_label_threshold: float = 0.5
    calibration_bins: int = 10
    output_dir: str = "./outputs/evaluation"
    report_name: str | None = None
    save_predictions: bool = False
    sdgs_path: str = "./configs/sdgs.yaml"


def _resolve_eval_dataset(cfg: Any, model: DistillationLitModule) -> tuple[DistillationDataset, str]:
    split = str(cfg.evaluation.split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("evaluation.split must be one of {'train', 'val', 'test'}")

    dm = DistillationDataModule(
        train_path=Path(str(cfg.data.train_path)),
        test_path=Path(str(cfg.data.test_path)),
        val_path=Path(str(cfg.data.val_path)) if cfg.data.val_path is not None else None,
        tokenizer_name=str(model.model_name),
        target_mode=str(model.target_mode),
        max_length=int(cfg.data.max_length),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        persistent_workers=bool(cfg.data.persistent_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=False,
        val_fraction=float(cfg.data.val_fraction),
        seed=int(cfg.seed),
    )

    if split == "test":
        dm.setup("test")
        if dm.test_dataset is None:
            raise RuntimeError("test dataset was not initialized")
        return dm.test_dataset, split

    dm.setup("fit")
    if split == "train":
        if dm.train_dataset is None:
            raise RuntimeError("train dataset was not initialized")
        return dm.train_dataset, split
    if dm.val_dataset is None:
        raise RuntimeError("val dataset is not available; set data.val_path or data.val_fraction > 0")
    return dm.val_dataset, split


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type {suffix!r} for {path}")


def _resolve_reference_df(cfg: Any, *, split: str, dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    attr = f"reference_{split}_path"
    ref_path = getattr(cfg.evaluation, attr, None)
    if ref_path in (None, "", "null"):
        return dataset_df.reset_index(drop=True), None

    ref_df = _read_table(Path(str(ref_path))).reset_index(drop=True)
    required_cols = ["sample_id", *sdg_target_columns("p1")]
    missing = [c for c in required_cols if c not in ref_df.columns]
    if missing:
        raise ValueError(f"Reference evaluation file missing required columns: {missing}")
    if ref_df["sample_id"].duplicated().any():
        n_dup = int(ref_df["sample_id"].duplicated().sum())
        raise ValueError(f"Reference evaluation file has duplicated sample_id rows: {n_dup}")

    ref_df = ref_df.set_index(ref_df["sample_id"].astype(str), drop=False)
    expected_ids = dataset_df["sample_id"].astype(str).tolist()
    missing_ids = [sid for sid in expected_ids if sid not in ref_df.index]
    if missing_ids:
        raise ValueError(
            f"Reference evaluation file is missing sample_ids for split={split}: {missing_ids[:5]}"
        )
    aligned = ref_df.loc[expected_ids].reset_index(drop=True)
    return aligned, str(ref_path)


def _resolve_checkpoint_path(cfg: Any) -> Path:
    checkpoint_path = cfg.evaluation.checkpoint_path
    checkpoint_run_id = cfg.evaluation.checkpoint_run_id

    if checkpoint_path not in (None, "", "null"):
        ckpt_path = Path(str(checkpoint_path))
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        return ckpt_path

    if checkpoint_run_id not in (None, "", "null"):
        run_id = str(checkpoint_run_id)
        ckpt_dir = Path("./outputs/wandb/sdg-distillation") / run_id / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(ckpt_dir)
        matches = sorted(ckpt_dir.glob("*.ckpt"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple checkpoints found for run_id={run_id!r}; set evaluation.checkpoint_path explicitly."
            )
        return matches[0]

    raise ValueError("Set either evaluation.checkpoint_path or evaluation.checkpoint_run_id for mode=evaluate")


def _build_eval_loader(cfg: Any, dataset: DistillationDataset) -> DataLoader:
    num_workers = int(cfg.data.num_workers)
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=bool(cfg.data.persistent_workers) if num_workers > 0 else False,
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=False,
    )


def _autocast_context(precision: int | str, device_type: str):
    precision_str = str(precision)
    if device_type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if precision_str == "16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision_str == "bf16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", enabled=False)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


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


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, *, bins: int) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    ece = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (float(mask.sum()) / total) * abs(conf - acc)
    return float(ece)


def _micro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return _binary_auroc(y_true.reshape(-1), y_prob.reshape(-1))


def load_sdg_titles(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    sdgs = raw.get("sdgs", {}) if isinstance(raw, dict) else {}
    titles: dict[int, str] = {}
    for k, v in sdgs.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if isinstance(v, dict):
            titles[idx] = str(v.get("title", "")).strip()
    return titles


def compute_multilabel_report(
    *,
    pred_probs: np.ndarray,
    teacher_probs: np.ndarray,
    hard_targets: np.ndarray,
    prediction_threshold: float,
    calibration_bins: int,
    sdg_titles: dict[int, str],
) -> tuple[dict[str, float], pd.DataFrame]:
    if pred_probs.shape != teacher_probs.shape or pred_probs.shape != hard_targets.shape:
        raise ValueError("pred_probs, teacher_probs, and hard_targets must have the same shape")

    n_samples, n_labels = pred_probs.shape
    pred_hard = (pred_probs >= prediction_threshold).astype(np.int64)

    per_label_rows: list[dict[str, Any]] = []
    aucs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    briers: list[float] = []
    eces: list[float] = []

    for label_idx in range(n_labels):
        label_no = label_idx + 1
        y_true = hard_targets[:, label_idx]
        y_prob = pred_probs[:, label_idx]
        y_teacher = teacher_probs[:, label_idx]
        y_pred = pred_hard[:, label_idx]

        auroc = _binary_auroc(y_true, y_prob)
        precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
        brier = _brier_score(y_true, y_prob)
        ece = _ece(y_true, y_prob, bins=calibration_bins)

        per_label_rows.append(
            {
                "label_index": label_no,
                "label": f"sdg{label_no:02d}",
                "title": sdg_titles.get(label_no, ""),
                "auroc": auroc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support_positive": int(y_true.sum()),
                "support_negative": int((1 - y_true).sum()),
                "prevalence": float(y_true.mean()),
                "pred_positive_rate": float(y_pred.mean()),
                "mean_pred_prob": float(y_prob.mean()),
                "mean_teacher_prob": float(y_teacher.mean()),
                "brier": brier,
                "ece": ece,
            }
        )

        aucs.append(auroc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        briers.append(brier)
        eces.append(ece)

    per_label_df = pd.DataFrame(per_label_rows)

    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
        hard_targets.reshape(-1),
        pred_hard.reshape(-1),
    )

    macro_auroc = float(np.nanmean(np.asarray(aucs, dtype=np.float64)))
    macro_precision = float(np.mean(np.asarray(precisions, dtype=np.float64)))
    macro_recall = float(np.mean(np.asarray(recalls, dtype=np.float64)))
    macro_f1 = float(np.mean(np.asarray(f1s, dtype=np.float64)))

    summary = {
        "num_samples": float(n_samples),
        "num_labels": float(n_labels),
        "macro_auroc": macro_auroc,
        "micro_auroc": _micro_auroc(hard_targets, pred_probs),
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "label_cardinality_true": float(hard_targets.sum(axis=1).mean()),
        "label_cardinality_pred": float(pred_hard.sum(axis=1).mean()),
        "mean_pred_prob": float(pred_probs.mean()),
        "mean_teacher_prob": float(teacher_probs.mean()),
        "soft_mae_p1": float(np.mean(np.abs(pred_probs - teacher_probs))),
        "soft_rmse_p1": float(np.sqrt(np.mean((pred_probs - teacher_probs) ** 2))),
        "macro_brier": float(np.mean(np.asarray(briers, dtype=np.float64))),
        "micro_brier": _brier_score(hard_targets.reshape(-1), pred_probs.reshape(-1)),
        "macro_ece": float(np.nanmean(np.asarray(eces, dtype=np.float64))),
        "micro_ece": _ece(hard_targets.reshape(-1), pred_probs.reshape(-1), bins=calibration_bins),
    }
    return summary, per_label_df


def build_prediction_frame(
    *,
    sample_ids: list[str],
    pred_probs: np.ndarray,
    hard_targets: np.ndarray,
    teacher_probs: np.ndarray,
    prediction_threshold: float,
) -> pd.DataFrame:
    out: dict[str, Any] = {"sample_id": sample_ids}
    pred_hard = (pred_probs >= prediction_threshold).astype(np.int64)
    for i in range(pred_probs.shape[1]):
        label = f"sdg{i + 1:02d}"
        out[f"pred_prob_{label}"] = pred_probs[:, i]
        out[f"pred_label_{label}"] = pred_hard[:, i]
        out[f"teacher_prob_{label}"] = teacher_probs[:, i]
        out[f"hard_target_{label}"] = hard_targets[:, i]
    return pd.DataFrame(out)


def flatten_metrics_for_logging(summary: dict[str, float], per_label_df: pd.DataFrame, *, split: str) -> dict[str, float]:
    metrics = {f"eval/{split}/{k}": float(v) for k, v in summary.items()}
    for row in per_label_df.itertuples(index=False):
        label = str(row.label)
        for metric_name in ("auroc", "f1", "precision", "recall", "prevalence"):
            value = getattr(row, metric_name)
            if pd.isna(value):
                continue
            metrics[f"eval/{split}/{label}/{metric_name}"] = float(value)
    return metrics


def write_evaluation_report(
    *,
    output_dir: Path,
    report_name: str,
    model_name: str,
    target_mode: str,
    split: str,
    summary: dict[str, Any],
    per_label_df: pd.DataFrame,
    reference_path: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_run_id: str | None = None,
    predictions_df: pd.DataFrame | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Path | None]:
    output_dir = output_dir / str(report_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_target_mode": target_mode,
        "checkpoint_model_name": model_name,
        "split": split,
        "reference_path": reference_path,
        "summary": summary,
    }
    if extra_payload:
        summary_payload.update(extra_payload)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    per_label_path = output_dir / "per_label_metrics.csv"
    per_label_df.to_csv(per_label_path, index=False)

    predictions_path: Path | None = None
    if predictions_df is not None:
        predictions_path = output_dir / "predictions.parquet"
        predictions_df.to_parquet(predictions_path, index=False)

    return {
        "summary_path": summary_path,
        "per_label_path": per_label_path,
        "predictions_path": predictions_path,
    }


def evaluate_checkpoint(cfg: Any) -> dict[str, Any]:
    ckpt_path = _resolve_checkpoint_path(cfg)

    model = DistillationLitModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    dataset, split = _resolve_eval_dataset(cfg, model)
    loader = _build_eval_loader(cfg, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sample_ids: list[str] = []
    pred_probs_parts: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with _autocast_context(cfg.trainer.precision, device.type):
                logits = model(input_ids, attention_mask)
                loss, _ = model._compute_loss(logits, targets)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            pred_probs_parts.append(probs)
            sample_ids.extend([str(x) for x in batch["sample_id"]])

            batch_size = int(input_ids.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_examples += batch_size

    pred_probs = np.concatenate(pred_probs_parts, axis=0)
    raw_df = dataset.df.reset_index(drop=True)
    expected_sample_ids = raw_df["sample_id"].astype(str).tolist()
    if sample_ids != expected_sample_ids:
        raise RuntimeError("Evaluation sample order mismatch between dataloader and dataset dataframe")

    reference_df, reference_path = _resolve_reference_df(cfg, split=split, dataset_df=raw_df)

    teacher_probs = reference_df[sdg_target_columns("p1")].to_numpy(dtype=np.float32)
    hard_targets = (teacher_probs >= float(cfg.evaluation.hard_label_threshold)).astype(np.int64)

    sdg_titles = load_sdg_titles(Path(str(cfg.evaluation.sdgs_path)))
    summary, per_label_df = compute_multilabel_report(
        pred_probs=pred_probs,
        teacher_probs=teacher_probs,
        hard_targets=hard_targets,
        prediction_threshold=float(cfg.evaluation.prediction_threshold),
        calibration_bins=int(cfg.evaluation.calibration_bins),
        sdg_titles=sdg_titles,
    )

    summary["eval_loss"] = total_loss / max(total_examples, 1)
    summary["hard_label_threshold"] = float(cfg.evaluation.hard_label_threshold)
    summary["prediction_threshold"] = float(cfg.evaluation.prediction_threshold)

    run_id = ckpt_path.parent.parent.name if ckpt_path.parent.parent.name else ckpt_path.stem
    report_name = cfg.evaluation.report_name
    if report_name in (None, "", "null"):
        report_name = f"{run_id}__{split}"
    pred_df: pd.DataFrame | None = None
    if bool(cfg.evaluation.save_predictions):
        pred_df = build_prediction_frame(
            sample_ids=sample_ids,
            pred_probs=pred_probs,
            hard_targets=hard_targets,
            teacher_probs=teacher_probs,
            prediction_threshold=float(cfg.evaluation.prediction_threshold),
        )
    report_paths = write_evaluation_report(
        output_dir=Path(str(cfg.evaluation.output_dir)),
        report_name=str(report_name),
        model_name=str(model.model_name),
        target_mode=str(model.target_mode),
        split=split,
        summary=summary,
        per_label_df=per_label_df,
        reference_path=reference_path,
        checkpoint_path=str(ckpt_path),
        checkpoint_run_id=run_id,
        predictions_df=pred_df,
    )

    print(
        "Evaluation summary: "
        f"checkpoint_run_id={run_id} split={split} "
        f"eval_loss={summary['eval_loss']:.6f} "
        f"macro_auroc={summary['macro_auroc']:.6f} "
        f"micro_auroc={summary['micro_auroc']:.6f} "
        f"macro_f1={summary['macro_f1']:.6f} "
        f"micro_f1={summary['micro_f1']:.6f}"
    )
    print(f"Wrote {report_paths['summary_path']}")
    print(f"Wrote {report_paths['per_label_path']}")
    if report_paths["predictions_path"] is not None:
        print(f"Wrote {report_paths['predictions_path']}")

    return {
        "summary": summary,
        "per_label_df": per_label_df,
        "summary_path": report_paths["summary_path"],
        "per_label_path": report_paths["per_label_path"],
        "predictions_path": report_paths["predictions_path"],
        "checkpoint_run_id": run_id,
        "checkpoint_target_mode": str(model.target_mode),
        "checkpoint_model_name": str(model.model_name),
        "split": split,
        "reference_path": reference_path,
    }
