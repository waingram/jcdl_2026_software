from __future__ import annotations

import math

import numpy as np

from sdg_distillation.training.metric_utils import compute_hard_target_summary, hard_targets_from_targets


def test_hard_targets_from_teacher_logits_uses_sign() -> None:
    targets = np.array([[-0.2, 0.0, 0.3]], dtype=float)
    hard = hard_targets_from_targets(targets, target_mode="teacher_logit")
    assert hard.tolist() == [[0, 1, 1]]


def test_hard_targets_from_prob_targets_uses_half_threshold() -> None:
    targets = np.array([[0.49, 0.5, 0.9]], dtype=float)
    hard = hard_targets_from_targets(targets, target_mode="p1")
    assert hard.tolist() == [[0, 1, 1]]


def test_compute_hard_target_summary_basic_metrics() -> None:
    pred_probs = np.array(
        [
            [0.9, 0.2],
            [0.8, 0.1],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    hard_targets = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=int,
    )

    summary = compute_hard_target_summary(
        pred_probs=pred_probs,
        hard_targets=hard_targets,
        prediction_threshold=0.5,
    )

    assert math.isclose(summary["macro_auroc"], 1.0)
    assert math.isclose(summary["micro_auroc"], 1.0)
    assert math.isclose(summary["macro_f1"], 1.0)
    assert math.isclose(summary["micro_f1"], 1.0)
