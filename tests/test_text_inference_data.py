from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from sdg_distillation.inference.dataset import InferenceTextConfig, TextInferenceDataset
from sdg_distillation.inference.datamodule import TextInferenceDataModule


class DummyTokenizer:
    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        token_value = min(len(text), max_length)
        input_ids = torch.full((1, max_length), token_value, dtype=torch.long)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_text_inference_dataset_builds_title_and_metadata() -> None:
    df = pd.DataFrame(
        [
            {
                "uri": "http://example.org/1",
                "title": "A title",
                "abstract": "An abstract",
                "degree": "PhD",
            }
        ]
    )
    dataset = TextInferenceDataset(
        df,
        tokenizer=DummyTokenizer(),
        text_config=InferenceTextConfig(
            id_col="uri",
            text_col="abstract",
            title_col="title",
            include_title=True,
        ),
        metadata_cols=["degree"],
    )

    item = dataset[0]
    assert item["sample_id"] == "http://example.org/1"
    assert item["degree"] == "PhD"
    assert item["text"] == "Title: A title\n\nAbstract: An abstract"
    assert tuple(item["input_ids"].shape) == (512,)
    assert tuple(item["attention_mask"].shape) == (512,)


def test_text_inference_datamodule_reads_csv_and_exposes_predict_loader(
    tmp_path: Path, monkeypatch
) -> None:
    predict_path = tmp_path / "vt_etds_sample.csv"
    pd.DataFrame(
        [
            {"uri": "u1", "abstract": "a1", "title": "t1", "degree": "PhD"},
            {"uri": "u2", "abstract": "a2", "title": "t2", "degree": "MS"},
        ]
    ).to_csv(predict_path, index=False)

    from sdg_distillation.inference import datamodule as dm_module

    monkeypatch.setattr(dm_module, "load_local_tokenizer", lambda _: DummyTokenizer())

    dm = TextInferenceDataModule(
        predict_path=predict_path,
        tokenizer_name="dummy",
        id_col="uri",
        text_col="abstract",
        title_col="title",
        include_title=False,
        metadata_cols=["degree"],
        batch_size=2,
        num_workers=0,
    )
    dm.setup("predict")
    loader = dm.predict_dataloader()
    batch = next(iter(loader))

    assert batch["sample_id"] == ["u1", "u2"]
    assert batch["degree"] == ["PhD", "MS"]
    assert tuple(batch["input_ids"].shape) == (2, 512)
