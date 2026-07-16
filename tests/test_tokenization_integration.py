# tests/test_tokenization_integration.py
from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_zero_one_are_single_tokens_under_qwen_tokenizer():
    """
    Integration test: requires model/tokenizer availability in the environment.
    This is intentionally opt-in because it may hit the network if not cached.
    """
    if os.environ.get("RUN_TOKENIZER_INTEGRATION_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_TOKENIZER_INTEGRATION_TESTS=1 to enable")

    from transformers import AutoTokenizer

    model_name = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    ids0 = tok.encode("0", add_special_tokens=False)
    ids1 = tok.encode("1", add_special_tokens=False)

    assert len(ids0) == 1, f"'0' must be single token; got {ids0}"
    assert len(ids1) == 1, f"'1' must be single token; got {ids1}"