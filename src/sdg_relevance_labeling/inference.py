# src/inference.py
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationArgs:
    max_new_tokens: int = 3
    return_full_text: bool = False  # kept for parity; we control decoding ourselves
    temperature: float = 0.0
    do_sample: bool = False


_LABEL_RE = re.compile(r"^\s*(Relevant|Non-Relevant)\s*\.?\s*$", re.IGNORECASE)
_BIT_RE = re.compile(r"^\s*([01])\s*$")


def ensure_padding(
    tokenizer: Any,
    padding_side: str = "left",
    ensure_pad_token: bool = True,
) -> int:
    """
    Ensure tokenizer has pad_token_id and desired padding_side.
    Returns the number of tokens added to the tokenizer vocab (0 in the common case).
    """
    added = 0

    if ensure_pad_token and getattr(tokenizer, "pad_token_id", None) is None:
        # Prefer reusing eos as pad to avoid resizing embeddings.
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Only do this if you truly have no EOS; otherwise avoid introducing a new token.
            added = tokenizer.add_special_tokens({"pad_token": "<pad>"})

        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if padding_side not in {"left", "right"}:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    tokenizer.padding_side = padding_side

    return int(added)


class QwenGenerateRunner:
    """
    Batched inference using model.generate so we can capture per-step logits.

    Prompts are strings (already chat-templated). For logits variants, we interpret logits
    for the FIRST generated token only, and only with a 0/1 response contract.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        seed: int = 0,
        padding_side: str = "left",
        ensure_pad_token: bool = True,
    ) -> None:
        torch.random.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        added = ensure_padding(
            self.tokenizer,
            padding_side=padding_side,
            ensure_pad_token=ensure_pad_token,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # If we actually added a new token, we must resize embeddings.
        # This should be 0 for Qwen because we reuse eos as pad.
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Cache token ids for "0" and "1" and enforce single-token encoding.
        self.tok0_id = self._single_token_id("0")
        self.tok1_id = self._single_token_id("1")

    def _single_token_id(self, s: str) -> int:
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Expected {s!r} to be a single token under this tokenizer; got ids={ids}"
            )
        return int(ids[0])

    def _input_device(self) -> torch.device:
        # With sharded device_map, parameters live on multiple devices; the first parameter’s
        # device is typically the correct place for input_ids (often cuda:0).
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def generate_batched(
        self,
        prompts: Sequence[str],
        batch_size: int,
        gen: GenerationArgs,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> List[str]:
        """
        Fast path: generate text only (no logits or probabilities).

        This method:
        - Accepts a list of already-templated prompt strings
        - Runs batched text generation using model.generate()
        - Returns exactly one generated string per input prompt

        It is designed for speed and simplicity when scores are not needed.
        """

        # Normalize prompts to a concrete list so we can index and slice safely.
        prompts_list = list(prompts)

        # Accumulates generated outputs in the same order as inputs.
        out: List[str] = []

        n = len(prompts_list)
        if n == 0:
            return out

        # Determine the device where input tensors should live.
        # With sharded models, this is typically the device of the first parameter.
        device = self._input_device()

        # Process prompts in batches to control memory usage.
        for start in range(0, n, batch_size):
            batch_prompts = prompts_list[start : start + batch_size]

            # Tokenize the batch:
            # - padding=True ensures all sequences in the batch have the same length
            # - truncation=False preserves the full prompt (important for classification prompts)
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            # Move all tensor values (input_ids, attention_mask, etc.) to the model device.
            for k in list(enc.keys()):
                if isinstance(enc[k], torch.Tensor):
                    enc[k] = enc[k].to(device)

            # Ensure at least one token is generated.
            max_new = max(1, int(gen.max_new_tokens))

            # Whether to use sampling (e.g., for justification text) or greedy decoding.
            do_sample = bool(gen.do_sample)

            # Base arguments for model.generate().
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new,
                "do_sample": do_sample,
            }

            # Only pass temperature when sampling is enabled.
            # Passing temperature with do_sample=False can trigger warnings.
            if do_sample:
                gen_kwargs["temperature"] = float(gen.temperature)

            # Retry loop to handle transient CUDA or generation failures.
            last_err: Optional[BaseException] = None
            for attempt in range(max_retries):
                try:
                    sequences = self.model.generate(
                        **enc,
                        **gen_kwargs,
                    )
                    last_err = None
                    break
                except BaseException as e:
                    last_err = e
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)

            if last_err is not None:
                raise RuntimeError("generate() failed after retries") from last_err

            # IMPORTANT:
            # We slice generated tokens starting at the *padded input length*,
            # not the per-row attention_mask sum.
            #
            # With left padding, attention_mask.sum() differs across rows,
            # and slicing at that point will accidentally include prompt tokens
            # in the decoded output.
            input_len = int(enc["input_ids"].shape[1])

            # Decode only the generated tokens for each sequence.
            for i in range(sequences.size(0)):
                gen_ids = sequences[i, input_len:]
                generated_text = self.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                )
                out.append(generated_text)

        # Safety check: ensure we produced exactly one output per input prompt.
        if len(out) != n:
            raise RuntimeError(f"Expected {n} outputs, got {len(out)}")

        return out

    @torch.inference_mode()
    def generate_batched_with_logits(
        self,
        prompts: Sequence[str],
        batch_size: int,
        gen: GenerationArgs,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate text and also capture logits for the first generated token.

        This method returns one dict per input prompt with:
        - generated_text: the decoded generated suffix (excluding the prompt)
        - logit_0: the model's pre-softmax score for token "0" at the first generated step
        - logit_1: the model's pre-softmax score for token "1" at the first generated step
        - teacher_logit: here defined as (logit_1 - logit_0), i.e., a logit margin
        - p1: softmax probability of choosing "1" when restricting the choice set to {"0","1"}

        Key semantic constraint:
        - These logits are only meaningful if your prompt contract forces the model’s *first*
        generated token to be the decision token (strict "0" or "1").
        - If the model sometimes emits whitespace, a newline, or "assistant" first, these logits
        still exist, but they no longer correspond to the decision you think they do.
        """
        # Normalize prompts to a concrete list for stable slicing and indexing.
        prompts_list = list(prompts)

        # Accumulates per-example dict outputs in the same order as inputs.
        out: List[Dict[str, Any]] = []

        n = len(prompts_list)
        if n == 0:
            return out

        # Determine the device to place input tensors on.
        # For sharded models this is typically the "first parameter" device.
        device = self._input_device()

        # Process prompts in batches to control GPU memory usage.
        for start in range(0, n, batch_size):
            batch_prompts = prompts_list[start : start + batch_size]

            # Tokenize the batch.
            # - padding=True pads all prompts to the same length (crucial for batching)
            # - truncation=False preserves full prompts (important for SDG prompting)
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            # Move tokenized tensors to the model device.
            for k, v in list(enc.items()):
                if isinstance(v, torch.Tensor):
                    enc[k] = v.to(device)

            # Ensure we generate at least one token so we can read "first-step" logits.
            max_new = max(1, int(gen.max_new_tokens))

            # Whether we sample or decode greedily.
            do_sample = bool(gen.do_sample)

            # generation kwargs:
            # - return_dict_in_generate=True gives us a structured output (sequences + scores)
            # - output_scores=True makes generate() store per-step scores (logits-like tensors)
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Only pass temperature when sampling is enabled.
            # Passing it with do_sample=False can trigger warnings and is semantically irrelevant.
            if do_sample:
                gen_kwargs["temperature"] = float(gen.temperature)

            # Retry loop for transient GPU / HF generation errors.
            last_err: Optional[BaseException] = None
            outputs = None
            for attempt in range(max_retries):
                try:
                    outputs = self.model.generate(**enc, **gen_kwargs)
                    last_err = None
                    break
                except BaseException as e:
                    last_err = e
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)

            if last_err is not None:
                raise RuntimeError("generate() failed after retries") from last_err

            # outputs.sequences has shape [B, input_len + new_tokens].
            # It includes the full prompt tokens followed by generated tokens.
            sequences = outputs.sequences

            # outputs.scores is a tuple of length = number of generated tokens.
            # Each element is shaped [B, vocab], one row per batch item.
            if not outputs.scores:
                raise RuntimeError("output_scores=True but no scores returned")

            # We only use the first generated step scores (step 0).
            # This corresponds to the distribution over the very first generated token.
            scores0 = outputs.scores[0]  # [B, vocab]

            if scores0.dim() != 2:
                raise RuntimeError(f"Unexpected scores[0] shape: {tuple(scores0.shape)}")

            # IMPORTANT:
            # Use padded input length for slicing generated tokens, not per-row prompt length.
            # The tokenizer padded all inputs to the same length, so input_len is consistent.
            input_len = int(enc["input_ids"].shape[1])

            # For each example in the batch:
            # - decode only the generated suffix
            # - extract logits for token ids corresponding to "0" and "1"
            # - compute a 2-class softmax probability for "1"
            for i in range(sequences.size(0)):
                # Generated portion only (exclude prompt).
                gen_ids = sequences[i, input_len:]
                generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Logits for the "0" and "1" decision tokens at the first generation step.
                # These are pre-softmax scores over the full vocabulary.
                l0 = float(scores0[i, self.tok0_id].item())
                l1 = float(scores0[i, self.tok1_id].item())

                # Logit margin (often used directly as a distillation target).
                # Positive margin => model prefers "1" over "0".
                margin = l1 - l0

                # Convert {l0, l1} into a probability p1 over the restricted choice set {"0","1"}.
                # This is not the same as softmax over the entire vocabulary.
                #
                # We use a numerically stable 2-logit softmax:
                #   p1 = exp(l1) / (exp(l0) + exp(l1))
                # implemented as exp(li - max) to avoid overflow.
                m = max(l0, l1)
                e0 = math.exp(l0 - m)
                e1 = math.exp(l1 - m)
                p1 = e1 / (e0 + e1)

                out.append(
                    {
                        "generated_text": generated_text,
                        "logit_0": l0,
                        "logit_1": l1,
                        "teacher_logit": margin,
                        "p1": p1,
                    }
                )

        # Sanity check: one output dict per input prompt.
        if len(out) != n:
            raise RuntimeError(f"Expected {n} outputs, got {len(out)}")

        return out

    def parse_label(self, generated_text: str) -> str:
        s = (generated_text or "").strip()
        m = _LABEL_RE.match(s)
        if not m:
            raise ValueError(
                f"Unexpected model output (expected Relevant/Non-Relevant): {s!r}"
            )

        label = m.group(1).lower()
        if label == "relevant":
            return "Relevant"
        if label == "non-relevant":
            return "Non-Relevant"

        # Defensive: should be unreachable given the regex
        raise ValueError(f"Invalid label value after match: {label!r}")

    def parse_bit(self, generated_text: str) -> str:
        s = (generated_text or "").strip()
        m = _BIT_RE.match(s)
        if not m:
            raise ValueError(f"Unexpected model output (expected 0/1): {s!r}")

        bit = m.group(1)
        if bit == "1":
            return "Relevant"
        if bit == "0":
            return "Non-Relevant"

        # Defensive: unreachable
        raise ValueError(f"Invalid bit value after match: {bit!r}")