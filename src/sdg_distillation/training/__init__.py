"""Training components (datasets, datamodules, models, losses)."""

from .data_contract import TargetMode, target_columns_for_mode, validate_student_ready_df
from .dataset import DistillationDataset, TokenizationConfig
from .losses import DistillationLossConfig, DistillationLossOutput, distillation_loss
from .model import DistillationLitModule

try:
    from .datamodule import DistillationDataModule
except ImportError:  # pragma: no cover
    DistillationDataModule = None  # type: ignore[assignment]

__all__ = [
    "TargetMode",
    "target_columns_for_mode",
    "validate_student_ready_df",
    "DistillationDataset",
    "TokenizationConfig",
    "DistillationLitModule",
    "DistillationLossConfig",
    "DistillationLossOutput",
    "distillation_loss",
]

if DistillationDataModule is not None:
    __all__.append("DistillationDataModule")
