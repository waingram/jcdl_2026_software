from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .data_contract import (
    TargetMode,
    TEXT_COLUMN,
    sdg_target_columns,
    target_columns_for_mode,
    validate_student_ready_df,
)


@dataclass(frozen=True)
class TokenizationConfig:
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True


class DistillationDataset(Dataset[dict[str, Any]]):
    """
    Dataset for student distillation from teacher-produced SDG targets.

    Each item returns:
      - input_ids: LongTensor[max_length]
      - attention_mask: LongTensor[max_length]
      - targets: FloatTensor[17] for target_mode in {"p1","teacher_logit","hard_label"}
      - p1_reference: FloatTensor[17] when canonical p1 reference columns are available
      - sample_id: str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        tokenizer: PreTrainedTokenizerBase,
        target_mode: TargetMode = "p1",
        tokenization: TokenizationConfig | None = None,
    ) -> None:
        if target_mode == "both":
            raise ValueError(
                "DistillationDataset expects a single target mode ('p1', 'teacher_logit', or 'hard_label'), got 'both'."
            )

        validate_student_ready_df(df, target_mode=target_mode)

        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.target_mode = target_mode
        self.tokenization = tokenization or TokenizationConfig()
        self.target_cols = target_columns_for_mode(target_mode)
        self.p1_reference_cols = sdg_target_columns("p1")
        if not all(col in self.df.columns for col in self.p1_reference_cols):
            self.p1_reference_cols = None

        # Normalize dtypes once up-front.
        self.df[TEXT_COLUMN] = self.df[TEXT_COLUMN].astype(str)
        self.df["sample_id"] = self.df["sample_id"].astype(str)
        self.df[self.target_cols] = self.df[self.target_cols].apply(
            pd.to_numeric, errors="raise"
        )
        if self.p1_reference_cols is not None:
            self.df[self.p1_reference_cols] = self.df[self.p1_reference_cols].apply(
                pd.to_numeric, errors="raise"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        text = row[TEXT_COLUMN]

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
            "targets": torch.tensor(
                row[self.target_cols].to_numpy(dtype="float32"),
                dtype=torch.float32,
            ),
            "sample_id": row["sample_id"],
        }

        if self.p1_reference_cols is not None:
            item["p1_reference"] = torch.tensor(
                row[self.p1_reference_cols].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

        return item
