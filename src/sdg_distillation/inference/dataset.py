from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from sdg_distillation.training.dataset import TokenizationConfig


def _require_column(df: pd.DataFrame, column: str, *, name: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Missing {name} column {column!r}")


@dataclass(frozen=True)
class InferenceTextConfig:
    id_col: str = "uri"
    text_col: str = "abstract"
    title_col: str | None = None
    include_title: bool = False


class TextInferenceDataset(Dataset[dict[str, Any]]):
    """
    Tokenized text-only dataset for ETD or other unlabeled corpus inference.

    Each item returns:
      - input_ids: LongTensor[max_length]
      - attention_mask: LongTensor[max_length]
      - sample_id: str
      - text: str
      - optional metadata columns requested at init time
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        tokenizer: PreTrainedTokenizerBase,
        text_config: InferenceTextConfig | None = None,
        tokenization: TokenizationConfig | None = None,
        metadata_cols: Sequence[str] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.text_config = text_config or InferenceTextConfig()
        self.tokenization = tokenization or TokenizationConfig()
        self.metadata_cols = list(metadata_cols or [])

        _require_column(self.df, self.text_config.id_col, name="identifier")
        _require_column(self.df, self.text_config.text_col, name="text")
        if self.text_config.include_title and self.text_config.title_col is None:
            raise ValueError("include_title=True requires title_col to be set")
        if self.text_config.title_col is not None:
            _require_column(self.df, self.text_config.title_col, name="title")
        for col in self.metadata_cols:
            _require_column(self.df, col, name="metadata")

        self.df[self.text_config.id_col] = (
            self.df[self.text_config.id_col].fillna("").astype(str).str.strip()
        )
        self.df[self.text_config.text_col] = self.df[self.text_config.text_col].fillna("").astype(str)
        if self.text_config.title_col is not None:
            self.df[self.text_config.title_col] = (
                self.df[self.text_config.title_col].fillna("").astype(str)
            )

    def __len__(self) -> int:
        return len(self.df)

    def _build_text(self, row: pd.Series) -> str:
        abstract = str(row[self.text_config.text_col])
        if not self.text_config.include_title or self.text_config.title_col is None:
            return abstract
        title = str(row[self.text_config.title_col]).strip()
        if not title:
            return abstract
        if not abstract:
            return title
        return f"Title: {title}\n\nAbstract: {abstract}"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        text = self._build_text(row)
        enc = self.tokenizer(
            text,
            max_length=self.tokenization.max_length,
            padding=self.tokenization.padding,
            truncation=self.tokenization.truncation,
            return_tensors="pt",
        )

        item: dict[str, Any] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sample_id": str(row[self.text_config.id_col]),
            "text": text,
        }
        for col in self.metadata_cols:
            value = row[col]
            item[col] = "" if pd.isna(value) else value
        return item
