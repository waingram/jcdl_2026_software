# tests/test_parsing.py
from __future__ import annotations

import pytest

from sdg_relevance_labeling.inference import QwenGenerateRunner


def _runner_without_init() -> QwenGenerateRunner:
    # parse_label/parse_bit do not depend on instance state.
    return object.__new__(QwenGenerateRunner)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Relevant", "Relevant"),
        (" relevant ", "Relevant"),
        ("Relevant.", "Relevant"),
        ("NON-RELEVANT", "Non-Relevant"),
        (" non-relevant. ", "Non-Relevant"),
    ],
)
def test_parse_label_accepts_expected_forms(text, expected):
    r = _runner_without_init()
    assert r.parse_label(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Rel",  # incomplete
        "Relevant because ...",  # extra tokens
        "Non Relevant",  # wrong punctuation
        "Classification: Relevant",  # justification-shaped output
    ],
)
def test_parse_label_rejects_unexpected_forms(text):
    r = _runner_without_init()
    with pytest.raises(ValueError):
        r.parse_label(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", "Non-Relevant"),
        (" 1 ", "Relevant"),
        ("\n0\n", "Non-Relevant"),
    ],
)
def test_parse_bit_accepts_expected_forms(text, expected):
    r = _runner_without_init()
    assert r.parse_bit(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2",
        "01",
        "1.",
        "bit=1",
    ],
)
def test_parse_bit_rejects_unexpected_forms(text):
    r = _runner_without_init()
    with pytest.raises(ValueError):
        r.parse_bit(text)