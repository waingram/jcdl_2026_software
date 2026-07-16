from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillationLossConfig:
    mode: str = "plain_bce"
    confidence_threshold: float = 0.0
    confidence_gamma: float = 1.0
    teacher_temperature: float = 1.0
    epsilon: float = 1.0e-6


@dataclass(frozen=True)
class DistillationLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _teacher_to_probs(
    teacher_logits: torch.Tensor,
    *,
    teacher_temperature: float,
) -> torch.Tensor:
    temperature = max(float(teacher_temperature), 1.0e-8)
    return torch.sigmoid(teacher_logits / temperature)


def _confidence_from_probs(target_probs: torch.Tensor) -> torch.Tensor:
    # 0 => maximum uncertainty (p=0.5), 1 => maximum confidence (p=0/1).
    return (target_probs - 0.5).abs() * 2.0


def distillation_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_mode: str,
    config: DistillationLossConfig,
) -> DistillationLossOutput:
    if target_mode == "p1":
        target_probs = targets.clamp(0.0, 1.0)
    elif target_mode == "hard_label":
        target_probs = targets.clamp(0.0, 1.0)
    elif target_mode == "teacher_logit":
        target_probs = _teacher_to_probs(
            targets,
            teacher_temperature=float(config.teacher_temperature),
        )
    else:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")

    per_elem_loss = F.binary_cross_entropy_with_logits(
        student_logits,
        target_probs,
        reduction="none",
    )
    confidence = _confidence_from_probs(target_probs)

    mode = str(config.mode)
    threshold = float(config.confidence_threshold)
    gamma = float(config.confidence_gamma)

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if gamma < 0.0:
        raise ValueError("confidence_gamma must be >= 0")

    if mode == "plain_bce":
        weights = torch.ones_like(confidence)
    elif mode == "masked_bce":
        weights = (confidence >= threshold).to(per_elem_loss.dtype)
    elif mode == "weighted_bce":
        base_weights = confidence.pow(gamma)
        if threshold > 0.0:
            base_weights = base_weights * (confidence >= threshold).to(per_elem_loss.dtype)
        weights = base_weights
    else:
        raise ValueError(f"Unsupported distillation loss mode={mode!r}")

    denom = weights.sum().clamp_min(float(config.epsilon))
    loss = (per_elem_loss * weights).sum() / denom

    metrics = {
        "distill/target_prob_mean": target_probs.mean().detach(),
        "distill/confidence_mean": confidence.mean().detach(),
        "distill/active_frac": (weights > 0).to(per_elem_loss.dtype).mean().detach(),
        "distill/weight_mean": weights.mean().detach(),
    }
    return DistillationLossOutput(loss=loss, metrics=metrics)
