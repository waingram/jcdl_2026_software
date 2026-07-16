from __future__ import annotations

import pandas as pd
import torch
from transformers import PretrainedConfig

from sdg_distillation.training.dataset import DistillationDataset
from sdg_distillation.training.model import DistillationLitModule


class _DummyTokenizer:
    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del text, padding, truncation
        assert return_tensors == "pt"
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def _sample_df_p1(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "sample_id": f"s{i}",
            "retrieval_sdg": "1",
            "split": "train",
            "row_id": i,
            "text": f"abstract {i}",
        }
        for j in range(1, 18):
            row[f"p1_sdg{j:02d}"] = 0.1 + (j % 3) * 0.2
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_df_hard(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "sample_id": f"h{i}",
            "retrieval_sdg": "1",
            "split": "train",
            "row_id": i,
            "text": f"abstract {i}",
        }
        for j in range(1, 18):
            row[f"hard_label_sdg{j:02d}"] = int((i + j) % 2 == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_distillation_dataset_shapes_and_types() -> None:
    ds = DistillationDataset(
        _sample_df_p1(),
        tokenizer=_DummyTokenizer(),  # type: ignore[arg-type]
        target_mode="p1",
    )
    item = ds[0]
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["targets"].dtype == torch.float32
    assert item["p1_reference"].dtype == torch.float32
    assert item["input_ids"].shape == (512,)
    assert item["attention_mask"].shape == (512,)
    assert item["targets"].shape == (17,)
    assert item["p1_reference"].shape == (17,)
    assert torch.allclose(item["targets"], item["p1_reference"])


def test_hard_label_dataset_shapes_and_types() -> None:
    ds = DistillationDataset(
        _sample_df_hard(),
        tokenizer=_DummyTokenizer(),  # type: ignore[arg-type]
        target_mode="hard_label",
    )
    item = ds[0]
    assert item["targets"].dtype == torch.float32
    assert item["targets"].shape == (17,)
    assert "p1_reference" not in item
    assert set(item["targets"].tolist()).issubset({0.0, 1.0})


class _DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        bsz, seq_len = input_ids.shape
        out = torch.randn((bsz, seq_len, self.hidden_size), dtype=torch.float32)
        return type("DummyOut", (), {"last_hidden_state": out})()


def test_training_step_returns_finite_loss(monkeypatch) -> None:
    import sdg_distillation.training.model as model_mod

    hidden_size = 32

    def _fake_config_from_pretrained(model_name: str, **kwargs: object) -> PretrainedConfig:
        del kwargs
        del model_name
        cfg = PretrainedConfig()
        cfg.hidden_size = hidden_size
        return cfg

    def _fake_model_from_pretrained(
        model_name: str, *, config: PretrainedConfig, **kwargs: object
    ) -> _DummyEncoder:
        del kwargs
        del model_name
        return _DummyEncoder(hidden_size=int(config.hidden_size))

    monkeypatch.setattr(
        model_mod.AutoConfig,
        "from_pretrained",
        staticmethod(_fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        model_mod.AutoModel,
        "from_pretrained",
        staticmethod(_fake_model_from_pretrained),
    )
    monkeypatch.setattr(model_mod, "resolve_local_hf_path", lambda model_name: model_name)

    model = DistillationLitModule(
        model_name="dummy/model",
        target_mode="p1",
        learning_rate=2.0e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=0.1,
        distillation_loss_mode="plain_bce",
        confidence_threshold=0.0,
        confidence_gamma=1.0,
        teacher_temperature=1.0,
        num_labels=17,
    )

    batch = {
        "input_ids": torch.ones((4, 16), dtype=torch.long),
        "attention_mask": torch.ones((4, 16), dtype=torch.long),
        "targets": torch.full((4, 17), 0.3, dtype=torch.float32),
    }
    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss).item()


def test_training_step_returns_finite_loss_for_hard_labels(monkeypatch) -> None:
    import sdg_distillation.training.model as model_mod

    hidden_size = 32

    def _fake_config_from_pretrained(model_name: str, **kwargs: object) -> PretrainedConfig:
        del kwargs
        del model_name
        cfg = PretrainedConfig()
        cfg.hidden_size = hidden_size
        return cfg

    def _fake_model_from_pretrained(
        model_name: str, *, config: PretrainedConfig, **kwargs: object
    ) -> _DummyEncoder:
        del kwargs
        del model_name
        return _DummyEncoder(hidden_size=int(config.hidden_size))

    monkeypatch.setattr(
        model_mod.AutoConfig,
        "from_pretrained",
        staticmethod(_fake_config_from_pretrained),
    )
    monkeypatch.setattr(
        model_mod.AutoModel,
        "from_pretrained",
        staticmethod(_fake_model_from_pretrained),
    )
    monkeypatch.setattr(model_mod, "resolve_local_hf_path", lambda model_name: model_name)

    model = DistillationLitModule(
        model_name="dummy/model",
        target_mode="hard_label",
        learning_rate=2.0e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=0.1,
        distillation_loss_mode="plain_bce",
        confidence_threshold=0.0,
        confidence_gamma=1.0,
        teacher_temperature=1.0,
        num_labels=17,
    )

    batch = {
        "input_ids": torch.ones((4, 16), dtype=torch.long),
        "attention_mask": torch.ones((4, 16), dtype=torch.long),
        "targets": torch.randint(0, 2, (4, 17), dtype=torch.int64).to(torch.float32),
    }
    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss).item()
