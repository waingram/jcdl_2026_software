from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as L
except ImportError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as L  # type: ignore[no-redef]

from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

from sdg_distillation.hf_local import resolve_local_hf_path

from .losses import DistillationLossConfig, distillation_loss
from .metric_utils import compute_hard_target_summary


class DistillationLitModule(L.LightningModule):
    """
    Encoder-only student for 17-label SDG distillation.

    Inputs: tokenized abstract text
    Targets: 17-dim targets (`p1`, `teacher_logit`, or `hard_label`)
    """

    def __init__(
        self,
        *,
        model_name: str,
        target_mode: str,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        dropout: float,
        distillation_loss_mode: str,
        confidence_threshold: float,
        confidence_gamma: float,
        teacher_temperature: float,
        validation_prediction_threshold: float = 0.5,
        validation_hard_label_threshold: float = 0.5,
        num_labels: int = 17,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.target_mode = target_mode
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.validation_prediction_threshold = float(validation_prediction_threshold)
        self.validation_hard_label_threshold = float(validation_hard_label_threshold)
        self.num_labels = int(num_labels)
        self.loss_config = DistillationLossConfig(
            mode=str(distillation_loss_mode),
            confidence_threshold=float(confidence_threshold),
            confidence_gamma=float(confidence_gamma),
            teacher_temperature=float(teacher_temperature),
        )
        self._val_logit_batches: list[torch.Tensor] = []
        self._val_p1_reference_batches: list[torch.Tensor] = []

        model_path = resolve_local_hf_path(model_name)
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            model_path,
            config=cfg,
            local_files_only=True,
        )
        # Keep training numerically stable: this checkpoint often loads in fp16
        # in this environment, which can produce NaNs under non-mixed precision.
        self.encoder = self.encoder.to(torch.float32)
        self.encoder.train()

        hidden_size = int(getattr(cfg, "hidden_size"))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation from last hidden state.
        pooled = out.last_hidden_state[:, 0, :]
        if pooled.dtype != self.classifier.weight.dtype:
            pooled = pooled.to(self.classifier.weight.dtype)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = distillation_loss(
            logits,
            targets,
            target_mode=self.target_mode,
            config=self.loss_config,
        )
        return out.loss, out.metrics

    def _resolve_p1_reference(self, batch: dict[str, Any]) -> torch.Tensor:
        if "p1_reference" in batch:
            return batch["p1_reference"].detach().to(torch.float32).cpu()

        targets = batch["targets"].detach().to(torch.float32).cpu()
        if self.target_mode == "p1":
            return targets
        if self.target_mode == "teacher_logit":
            return torch.sigmoid(targets)
        raise RuntimeError(
            "Validation requires canonical p1_reference values for hard-label training. "
            "Attach p1 columns or configure evaluation.reference_*_path."
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss, metrics = self._compute_loss(logits, batch["targets"])
        bsz = int(batch["input_ids"].shape[0])
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=bsz,
        )
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_step=True,
            on_epoch=True,
            batch_size=bsz,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_logit_batches.clear()
        self._val_p1_reference_batches.clear()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss, metrics = self._compute_loss(logits, batch["targets"])
        bsz = int(batch["input_ids"].shape[0])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=bsz)
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=bsz,
        )

        self._val_logit_batches.append(logits.detach().to(torch.float32).cpu())
        self._val_p1_reference_batches.append(self._resolve_p1_reference(batch))

    def on_validation_epoch_end(self) -> None:
        if not self._val_logit_batches:
            return

        logits = torch.cat(self._val_logit_batches, dim=0).to(torch.float32)
        p1_reference = torch.cat(self._val_p1_reference_batches, dim=0).to(torch.float32)
        pred_probs = torch.sigmoid(logits)

        soft_bce_p1 = F.binary_cross_entropy_with_logits(logits, p1_reference)
        brier_p1 = torch.mean((pred_probs - p1_reference) ** 2)

        pred_probs_np = pred_probs.numpy().astype(np.float64, copy=False)
        hard_targets_np = (
            p1_reference.numpy().astype(np.float64, copy=False)
            >= self.validation_hard_label_threshold
        ).astype(np.int64)
        hard_metrics = compute_hard_target_summary(
            pred_probs=pred_probs_np,
            hard_targets=hard_targets_np,
            prediction_threshold=self.validation_prediction_threshold,
        )

        self.log("val_soft_bce_p1", soft_bce_p1, prog_bar=True, logger=True)
        self.log("val_brier_p1", brier_p1, prog_bar=False, logger=True)
        self.log("val_micro_f1", hard_metrics["micro_f1"], prog_bar=False, logger=True)
        self.log("val_macro_f1", hard_metrics["macro_f1"], prog_bar=False, logger=True)
        self.log("val_micro_auroc", hard_metrics["micro_auroc"], prog_bar=False, logger=True)
        self.log("val_macro_auroc", hard_metrics["macro_auroc"], prog_bar=False, logger=True)

        self._val_logit_batches.clear()
        self._val_p1_reference_batches.clear()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss, metrics = self._compute_loss(logits, batch["targets"])
        bsz = int(batch["input_ids"].shape[0])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=bsz,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
