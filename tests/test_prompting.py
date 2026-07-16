# tests/test_prompting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sdg_relevance_labeling.prompting import QwenPromptBuilder, SDGSpec


class DummyTokenizer:
    """
    Minimal stub that behaves like a HF tokenizer for the one thing we need:
    apply_chat_template(..., tokenize=False, add_generation_prompt=True) -> str
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        # Make sure we embed the sentinel exactly once inside "user" content.
        # The builder itself is responsible for placing the sentinel in user_msg.
        # We return a templated string that preserves it.
        system = messages[0]["content"]
        user = messages[1]["content"]
        return f"<SYS>\n{system}\n</SYS>\n<USR>\n{user}\n</USR>\n<ASSISTANT>\n"


@pytest.fixture()
def sdg1() -> SDGSpec:
    return SDGSpec(
        number="1",
        title="End poverty in all its forms everywhere",
        description="poverty eradication, social protection, and access to resources",
        targets="**Target 1.1**: ...",
    )


@pytest.mark.parametrize(
    "variant, expected_mode, expects_logits, store_probs",
    [
        ("justification", "justification", False, False),
        ("binary_label", "label", False, False),
        ("binary_bit", "bit", False, False),
        ("binary_bit_with_probs", "bit", True, True),
    ],
)
def test_prompt_contract_and_sentinel_split(variant, expected_mode, expects_logits, store_probs, sdg1):
    tok = DummyTokenizer()
    pb = QwenPromptBuilder(
        tokenizer=tok,
        sdg=sdg1,
        variant=variant,
        include_contribution_types=True,
        include_indirect_clause=True,
        sentinel="<<__ABSTRACT_SENTINEL__>>",
    )

    # Contract sanity
    assert pb.contract.variant == variant
    assert pb.contract.response_mode == expected_mode
    assert pb.contract.expects_logits is expects_logits
    assert pb.contract.store_probs is store_probs

    # The template call should happen exactly once in __init__.
    assert len(tok.calls) == 1
    assert tok.calls[0]["tokenize"] is False
    assert tok.calls[0]["add_generation_prompt"] is True

    # build() should interpolate abstract and never leave the sentinel behind.
    prompt = pb.build("hello abstract")
    assert "<<__ABSTRACT_SENTINEL__>>" not in prompt
    assert "hello abstract" in prompt

    # Minimal content invariants: SDG and targets appear (guardrails against accidental prompt regression).
    assert "SDG 1" in prompt
    assert "Targets" in prompt
    assert "**Target 1.1**" in prompt


@pytest.mark.parametrize("include_types", [True, False])
@pytest.mark.parametrize("include_indirect", [True, False])
def test_prompt_optional_blocks_toggle(include_types, include_indirect, sdg1):
    tok = DummyTokenizer()
    pb = QwenPromptBuilder(
        tokenizer=tok,
        sdg=sdg1,
        variant="binary_label",
        include_contribution_types=include_types,
        include_indirect_clause=include_indirect,
        sentinel="<<__ABSTRACT_SENTINEL__>>",
    )
    prompt = pb.build("x")
    print(prompt)

    if include_types:
        assert "specify the type of contribution" in prompt.lower()
    else:
        assert "specify the type of contribution" not in prompt.lower()

    if include_indirect:
        assert "not strict criteria" in prompt.lower()


def test_unknown_variant_raises(sdg1):
    tok = DummyTokenizer()
    with pytest.raises(ValueError):
        QwenPromptBuilder(
            tokenizer=tok,
            sdg=sdg1,
            variant="nope",  # type: ignore[arg-type]
        )