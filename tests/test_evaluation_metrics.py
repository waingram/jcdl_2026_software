from __future__ import annotations

import math

import numpy as np

from sdg_distillation.training.evaluation import compute_multilabel_report


def test_compute_multilabel_report_basic_metrics() -> None:
    pred_probs = np.array(
        [
            [0.9, 0.2],
            [0.8, 0.1],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    teacher_probs = pred_probs.copy()
    hard_targets = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=int,
    )

    summary, per_label = compute_multilabel_report(
        pred_probs=pred_probs,
        teacher_probs=teacher_probs,
        hard_targets=hard_targets,
        prediction_threshold=0.5,
        calibration_bins=5,
        sdg_titles={1: "A", 2: "B"},
    )

    assert math.isclose(summary["macro_auroc"], 1.0)
    assert math.isclose(summary["micro_auroc"], 1.0)
    assert math.isclose(summary["macro_f1"], 1.0)
    assert math.isclose(summary["micro_f1"], 1.0)
    assert math.isclose(summary["label_cardinality_true"], 1.0)
    assert math.isclose(summary["label_cardinality_pred"], 1.0)
    assert list(per_label["label"]) == ["sdg01", "sdg02"]
    assert per_label["title"].tolist() == ["A", "B"]
    assert per_label["support_positive"].tolist() == [2, 2]
