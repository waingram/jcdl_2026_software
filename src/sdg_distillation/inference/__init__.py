"""Inference-side datasets and data modules for unlabeled corpora."""

from .datamodule import TextInferenceDataModule
from .dataset import TextInferenceDataset

__all__ = [
    "TextInferenceDataset",
    "TextInferenceDataModule",
]
