from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from torch.utils.data import DataLoader
try:
    import lightning as L
except ImportError:  # pragma: no cover - compatibility fallback
    try:
        import pytorch_lightning as L  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TextInferenceDataModule requires `lightning` (or `pytorch-lightning`). "
            "Install with: uv add lightning"
        ) from exc

from sdg_distillation.hf_local import load_local_tokenizer
from sdg_distillation.training.dataset import TokenizationConfig

from .dataset import InferenceTextConfig, TextInferenceDataset


class TextInferenceDataModule(L.LightningDataModule):
    """LightningDataModule for unlabeled text inference over ETD-like corpora."""

    def __init__(
        self,
        *,
        predict_path: Path,
        tokenizer_name: str,
        id_col: str = "uri",
        text_col: str = "abstract",
        title_col: str | None = None,
        include_title: bool = False,
        metadata_cols: Sequence[str] | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.predict_path = Path(predict_path)
        self.tokenizer_name = tokenizer_name
        self.text_config = InferenceTextConfig(
            id_col=id_col,
            text_col=text_col,
            title_col=title_col,
            include_title=include_title,
        )
        self.metadata_cols = list(metadata_cols or [])
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.persistent_workers = bool(persistent_workers)
        self.pin_memory = bool(pin_memory)

        self.tokenizer = None
        self.predict_dataset: Optional[TextInferenceDataset] = None

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file type {suffix!r} for {path}")

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "predict"):
            return
        if self.tokenizer is None:
            self.tokenizer = load_local_tokenizer(self.tokenizer_name)

        predict_df = self._read_table(self.predict_path)
        self.predict_dataset = TextInferenceDataset(
            predict_df,
            tokenizer=self.tokenizer,
            text_config=self.text_config,
            tokenization=TokenizationConfig(max_length=self.max_length),
            metadata_cols=self.metadata_cols,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_dataset is None:
            raise RuntimeError("setup('predict') must be called before predict_dataloader().")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
