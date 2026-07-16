# src/prompting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple


@dataclass(frozen=True)
class SDGSpec:
    number: str
    title: str
    description: str
    targets: str


Variant = Literal[
    "justification",
    "binary_label",
    "binary_bit",
    "binary_bit_with_probs",
]


@dataclass(frozen=True)
class PromptContract:
    variant: Variant
    response_mode: Literal["justification", "label", "bit"]
    expects_logits: bool
    store_probs: bool


class QwenPromptBuilder:
    """
    Build a single chat-templated prompt string using apply_chat_template once.

    The variant controls:
      - response contract in the prompt
      - which parsing rule applies downstream
      - whether 0/1 logits are semantically well-defined
    """

    def __init__(
        self,
        tokenizer: Any,
        sdg: SDGSpec,
        variant: Variant,
        include_contribution_types: bool = True,
        include_indirect_clause: bool = True,
        sentinel: str = "<<__ABSTRACT_SENTINEL__>>",
    ) -> None:
        self.tokenizer = tokenizer
        self.sdg = sdg
        self.variant = variant
        self.include_contribution_types = include_contribution_types
        self.include_indirect_clause = include_indirect_clause
        self.sentinel = sentinel

        self.contract = self._contract_for(variant)
        self._prefix, self._suffix = self._build_prefix_suffix()

    @staticmethod
    def _contract_for(variant: Variant) -> PromptContract:
        if variant == "justification":
            return PromptContract(
                variant=variant,
                response_mode="justification",
                expects_logits=False,
                store_probs=False,
            )
        if variant == "binary_label":
            return PromptContract(
                variant=variant,
                response_mode="label",
                expects_logits=False,
                store_probs=False,
            )
        if variant == "binary_bit":
            return PromptContract(
                variant=variant,
                response_mode="bit",
                expects_logits=False,
                store_probs=False,
            )
        if variant == "binary_bit_with_probs":
            return PromptContract(
                variant=variant,
                response_mode="bit",
                expects_logits=True,
                store_probs=True,
            )
        raise ValueError(f"Unknown variant {variant!r}")

    def build(self, abstract_text: str) -> str:
        return self._prefix + (abstract_text or "") + self._suffix

    def _build_prefix_suffix(self) -> Tuple[str, str]:
        sdg_number = self.sdg.number
        sdg_title = self.sdg.title
        sdg_targets = self.sdg.targets

        system_msg = (
            f"You are evaluating whether a research abstract makes a substantive contribution to "
            f"Sustainable Development Goal {sdg_number} (SDG {sdg_number}): {sdg_title}. "
            f"Contributions may include direct interventions, theoretical insights, policy discussions, "
            f"empirical studies, or other ways in which the research advances knowledge or action related "
            f"to SDG {sdg_number}."
        )

        include_types_block = """
If applicable, specify the type of contribution, such as:
- Theoretical analysis
- Empirical study
- Policy framework
- Technical innovation
- Other (briefly describe)
""".strip()

        indirect_clause = """
These targets provide guidance but are not strict criteria. A document may still be relevant even if its contribution is indirect.
""".strip()

        if self.contract.response_mode == "label":
            task_lines = f"""
Determine whether the following abstract makes a substantive contribution to SDG {sdg_number}.
- If yes, classify it as Relevant.
- If no, classify it as Non-Relevant.
""".strip()

            response_block = """
### Response Format
Respond with exactly one of the following words, and nothing else:

Relevant
Non-Relevant
""".strip()

        elif self.contract.response_mode == "bit":
            # Define the mapping explicitly in the same units as the response.
            task_lines = f"""
Determine whether the following abstract makes a substantive contribution to SDG {sdg_number}.
- If yes, classify it as Relevant.
- If no, classify it as Non-Relevant.
""".strip()

            response_block = """
### Response Format
Respond with a binary digit: 0 for Non-Relevant, 1 for Relevant.
Respond with exactly one digit, and nothing else:

0
1
""".strip()

        else:
            # justification
            task_lines = f"""
Determine whether the following abstract makes a substantive contribution to SDG {sdg_number}.
- If yes, classify it as Relevant.
- If no, classify it as Non-Relevant.
""".strip()

            response_block = """
### Response Format
1. Classification: (Relevant / Non-Relevant)
2. Justification: Briefly explain why.
3. (Optional) Contribution Type: If clear, name one type.
""".strip()

        user_msg = f"""
### Task
{task_lines}

{include_types_block if self.include_contribution_types else ""}
---

SDG {sdg_number} Targets (Reference Only)
{sdg_targets}

{indirect_clause if self.include_indirect_clause else ""}
---

Document Abstract:
{self.sentinel}

{response_block}
""".strip()

        messages: list[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        templated = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        count = templated.count(self.sentinel)
        if count != 1:
            raise ValueError(
                f"Sentinel split failed: sentinel={self.sentinel!r} count={count}"
            )

        prefix, suffix = templated.split(self.sentinel, 1)
        return prefix, suffix