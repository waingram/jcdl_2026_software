from __future__ import annotations

from typing import Any

import numpy as np


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def micro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return binary_auroc(y_true.reshape(-1), y_prob.reshape(-1))


def compute_hard_target_summary(
    *,
    pred_probs: np.ndarray,
    hard_targets: np.ndarray,
    prediction_threshold: float,
) -> dict[str, float]:
    if pred_probs.shape != hard_targets.shape:
        raise ValueError("pred_probs and hard_targets must have the same shape")

    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    hard_targets = np.asarray(hard_targets, dtype=np.int64)
    pred_hard = (pred_probs >= float(prediction_threshold)).astype(np.int64)

    aucs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label_idx in range(pred_probs.shape[1]):
        y_true = hard_targets[:, label_idx]
        y_prob = pred_probs[:, label_idx]
        y_pred = pred_hard[:, label_idx]

        aucs.append(binary_auroc(y_true, y_prob))
        precision, recall, f1 = precision_recall_f1(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    micro_precision, micro_recall, micro_f1 = precision_recall_f1(
        hard_targets.reshape(-1),
        pred_hard.reshape(-1),
    )

    return {
        "macro_auroc": float(np.nanmean(np.asarray(aucs, dtype=np.float64))),
        "micro_auroc": micro_auroc(hard_targets, pred_probs),
        "macro_precision": float(np.mean(np.asarray(precisions, dtype=np.float64))),
        "micro_precision": micro_precision,
        "macro_recall": float(np.mean(np.asarray(recalls, dtype=np.float64))),
        "micro_recall": micro_recall,
        "macro_f1": float(np.mean(np.asarray(f1s, dtype=np.float64))),
        "micro_f1": micro_f1,
    }


def hard_targets_from_targets(targets: np.ndarray, *, target_mode: str) -> np.ndarray:
    targets = np.asarray(targets)
    if target_mode == "teacher_logit":
        return (targets >= 0.0).astype(np.int64)
    if target_mode in {"p1", "hard_label"}:
        return (targets >= 0.5).astype(np.int64)
    raise ValueError(f"Unsupported target_mode={target_mode!r}")
