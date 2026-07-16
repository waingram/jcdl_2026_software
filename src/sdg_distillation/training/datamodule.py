from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader
try:
    import lightning as L
except ImportError:  # pragma: no cover - compatibility fallback
    try:
        import pytorch_lightning as L  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DistillationDataModule requires `lightning` (or `pytorch-lightning`). "
            "Install with: uv add lightning"
        ) from exc

from sdg_distillation.hf_local import load_local_tokenizer
from .data_contract import TargetMode, sdg_target_columns, validate_student_ready_df
from .dataset import DistillationDataset, TokenizationConfig
from .splits import split_train_val_df


class DistillationDataModule(L.LightningDataModule):
    """
    LightningDataModule for distillation from student_ready parquet/csv artifacts.
    """

    def __init__(
        self,
        *,
        train_path: Path,
        test_path: Path,
        val_path: Path | None = None,
        reference_train_path: Path | None = None,
        reference_val_path: Path | None = None,
        reference_test_path: Path | None = None,
        tokenizer_name: str,
        target_mode: TargetMode = "p1",
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.val_path = Path(val_path) if val_path is not None else None
        self.reference_train_path = Path(reference_train_path) if reference_train_path is not None else None
        self.reference_val_path = Path(reference_val_path) if reference_val_path is not None else None
        self.reference_test_path = Path(reference_test_path) if reference_test_path is not None else None
        self.tokenizer_name = tokenizer_name
        self.target_mode = target_mode
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.persistent_workers = bool(persistent_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)

        self.tokenizer = None
        self.train_dataset: Optional[DistillationDataset] = None
        self.val_dataset: Optional[DistillationDataset] = None
        self.test_dataset: Optional[DistillationDataset] = None

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

    @staticmethod
    def _attach_p1_reference(df: pd.DataFrame, reference_path: Path | None) -> pd.DataFrame:
        p1_cols = sdg_target_columns("p1")
        if all(col in df.columns for col in p1_cols):
            return df
        if reference_path is None:
            return df

        ref_df = DistillationDataModule._read_table(reference_path).reset_index(drop=True)
        required_cols = ["sample_id", *p1_cols]
        missing = [c for c in required_cols if c not in ref_df.columns]
        if missing:
            raise ValueError(f"Reference file missing required columns: {missing}")
        if ref_df["sample_id"].duplicated().any():
            n_dup = int(ref_df["sample_id"].duplicated().sum())
            raise ValueError(f"Reference file has duplicated sample_id rows: {n_dup}")

        base_df = df.reset_index(drop=True).copy()
        base_df["sample_id"] = base_df["sample_id"].astype(str)
        ref_df = ref_df.loc[:, required_cols].copy()
        ref_df["sample_id"] = ref_df["sample_id"].astype(str)

        merged = base_df.merge(ref_df, on="sample_id", how="left", validate="1:1")
        missing_mask = merged[p1_cols].isna().any(axis=1)
        if missing_mask.any():
            missing_ids = merged.loc[missing_mask, "sample_id"].astype(str).tolist()
            raise ValueError(
                "Reference file is missing sample_ids for p1 alignment: "
                f"{missing_ids[:5]}"
            )
        return merged

    def setup(self, stage: str | None = None) -> None:
        if self.tokenizer is None:
            self.tokenizer = load_local_tokenizer(self.tokenizer_name)

        tok_cfg = TokenizationConfig(max_length=self.max_length)

        if stage in (None, "fit"):
            train_df = self._read_table(self.train_path)
            validate_student_ready_df(
                train_df,
                target_mode=self.target_mode,
                expected_split="train",
            )

            train_df = self._attach_p1_reference(
                train_df.reset_index(drop=True),
                self.reference_train_path,
            )
            if self.val_path is not None:
                val_df = self._read_table(self.val_path).reset_index(drop=True)
                val_ref_path = self.reference_val_path
                if val_ref_path is None:
                    val_ref_path = self.reference_train_path
                val_df = self._attach_p1_reference(val_df, val_ref_path)
                # Prefer explicit val split when provided.
            else:
                train_df, val_df = split_train_val_df(
                    train_df,
                    val_fraction=self.val_fraction,
                    seed=self.seed,
                )

            # For explicit val_path, allow either split labeling convention.
            if self.val_path is not None:
                validate_student_ready_df(
                    val_df,
                    target_mode=self.target_mode,
                    expected_split=None,
                )

            self.train_dataset = DistillationDataset(
                train_df,
                tokenizer=self.tokenizer,
                target_mode=self.target_mode,
                tokenization=tok_cfg,
            )

            if len(val_df) > 0:
                # Note: split remains "train" in metadata because val is carved from train.
                self.val_dataset = DistillationDataset(
                    val_df,
                    tokenizer=self.tokenizer,
                    target_mode=self.target_mode,
                    tokenization=tok_cfg,
                )
            else:
                self.val_dataset = None

        if stage in (None, "test"):
            test_df = self._read_table(self.test_path)
            validate_student_ready_df(
                test_df,
                target_mode=self.target_mode,
                expected_split="test",
            )
            test_df = self._attach_p1_reference(
                test_df.reset_index(drop=True),
                self.reference_test_path,
            )
            self.test_dataset = DistillationDataset(
                test_df,
                tokenizer=self.tokenizer,
                target_mode=self.target_mode,
                tokenization=tok_cfg,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must be called before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("setup('test') must be called before test_dataloader().")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            pin_memory=self.pin_memory,
        )
