from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def resolve_local_hf_path(model_name_or_path: str) -> str:
    """Resolve a Hugging Face repo ID to a cached local snapshot path.

    Passing a local directory to `from_pretrained` avoids online metadata checks
    in offline environments. If `model_name_or_path` is already a local path,
    it is returned unchanged.
    """

    path = Path(model_name_or_path)
    if path.exists():
        return str(path)
    return snapshot_download(repo_id=model_name_or_path, local_files_only=True)


def _is_false_positive_mistral_warning(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False
    model_type = config.get("model_type")
    return model_type not in {"mistral", "mistral3", "voxtral", "ministral", "pixtral"}


@contextmanager
def _silence_false_positive_regex_warning(model_path: str):
    logger = logging.getLogger("transformers.tokenization_utils_tokenizers")
    previous_level = logger.level
    should_silence = _is_false_positive_mistral_warning(model_path)
    if should_silence:
        logger.setLevel(max(previous_level, logging.ERROR))
    try:
        yield
    finally:
        if should_silence:
            logger.setLevel(previous_level)


def load_local_tokenizer(model_name_or_path: str):
    model_path = resolve_local_hf_path(model_name_or_path)
    with _silence_false_positive_regex_warning(model_path):
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
