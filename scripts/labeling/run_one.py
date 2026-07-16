#!/usr/bin/env python3

# Description: Run one evaluation SDG prompt over a single input CSV and
# write unified labeling outputs, including optional logits/probabilities
# when the prompt contract requests them.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/labeling/run_one.py [--run-config PATH --limit N --dry-run]

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from sdg_relevance_labeling.config import load_run_config, load_sdgs
from sdg_relevance_labeling.data import iter_batches, load_scopus_csv
from sdg_relevance_labeling.inference import GenerationArgs, QwenGenerateRunner
from sdg_relevance_labeling.prompting import QwenPromptBuilder

_JUSTIFICATION_CLASS_RE = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?\s*classification\s*(?:\*\*)?\s*:\s*(relevant|non-relevant)\s*\.?\s*$"
)
_ANY_LABEL_RE = re.compile(r"(?i)\b(non-relevant|relevant)\b")


def _parse_justification_label(generated_text: str) -> str:
    """
    Strict parse for justification: require an explicit 'Classification: ...' line.
    Fallback: if missing, allow exactly one label occurrence anywhere (rare, but keeps runs moving).
    """
    s = (generated_text or "").strip()
    m = _JUSTIFICATION_CLASS_RE.search(s)
    if m:
        label = m.group(1).lower()
        return "Non-Relevant" if label.startswith("non") else "Relevant"

    # Fallback: only if exactly one label appears anywhere.
    hits = _ANY_LABEL_RE.findall(s)
    hits_norm = [h.lower() for h in hits]
    # Normalize and dedupe while preserving count logic.
    if len(hits_norm) == 1:
        return "Non-Relevant" if hits_norm[0].startswith("non") else "Relevant"

    raise ValueError(
        "Unexpected justification output: could not extract a single Classification label."
    )

def _csv_one_line(s: str) -> str:
    # Force 1 physical line per record. Reversible.
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a single-file labeling run."""

    p = argparse.ArgumentParser(
        description="Run one SDG prompt over one input CSV and write unified outputs for flip/logit analysis."
    )
    p.add_argument(
        "--run-config", required=True, type=Path, help="Path to run config YAML."
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Override max_rows for quick tests."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and parsing but do not write CSV.",
    )
    return p.parse_args()


def main() -> None:
    """Run one configured labeling job end to end for a single input CSV."""

    args = parse_args()
    run_cfg = load_run_config(args.run_config)

    sdgs = load_sdgs(run_cfg.sdgs_path)
    sdg_num = run_cfg.run.sdg_number
    if sdg_num not in sdgs:
        known = ", ".join(sorted(sdgs.keys(), key=lambda x: int(x)))
        raise ValueError(f"Unknown SDG {sdg_num!r}. Known SDGs: {known}")
    sdg_spec = sdgs[sdg_num]

    runner = QwenGenerateRunner(
        model_name=run_cfg.model.name,
        trust_remote_code=run_cfg.model.trust_remote_code,
        device_map=run_cfg.model.device_map,
        torch_dtype=run_cfg.model.torch_dtype,
        seed=run_cfg.model.seed,
        padding_side=run_cfg.model.padding_side,
        ensure_pad_token=run_cfg.model.ensure_pad_token,
    )

    prompt_builder = QwenPromptBuilder(
        tokenizer=runner.tokenizer,
        sdg=sdg_spec,
        variant=run_cfg.prompt.variant,
        include_contribution_types=run_cfg.prompt.include_contribution_types,
        include_indirect_clause=run_cfg.prompt.include_indirect_clause,
    )
    contract = prompt_builder.contract

    gen = GenerationArgs(
        max_new_tokens=run_cfg.generation.max_new_tokens,
        return_full_text=run_cfg.generation.return_full_text,
        temperature=run_cfg.generation.temperature,
        do_sample=run_cfg.generation.do_sample,
    )

    df = load_scopus_csv(
        run_cfg.input.path,
        doi_col=run_cfg.input.doi_col,
        abstract_col=run_cfg.input.text_col,
    )

    limit = args.limit if args.limit is not None else run_cfg.run.max_rows
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        df = df.head(limit)

    # Dry-run: print a single interpolated prompt to stdout (first row after any limiting).
    if args.dry_run and len(df) > 0:
        sample_abs = str(df.iloc[0][run_cfg.input.text_col] or "")
        print("----- BEGIN SAMPLE PROMPT (interpolated) -----")
        print(prompt_builder.build(sample_abs))
        print("----- END SAMPLE PROMPT (interpolated) -----")

    n_batches = math.ceil(len(df) / run_cfg.run.batch_size) if len(df) else 0

    rows: List[Dict[str, Any]] = []

    batch_iter = iter_batches(
        df,
        batch_size=run_cfg.run.batch_size,
        doi_col=run_cfg.input.doi_col,
        abstract_col=run_cfg.input.text_col,
    )

    for batch in tqdm(
        batch_iter, total=n_batches, desc=f"SDG {sdg_spec.number}", unit="batch"
    ):
        prompts = [prompt_builder.build(r.abstract) for r in batch]

        if contract.store_probs:
            # Only variant that requests logits/probs
            gen_out = runner.generate_batched_with_logits(
                prompts=prompts,
                batch_size=run_cfg.run.batch_size,
                gen=gen,
            )
        else:
            gen_texts = runner.generate_batched(
                prompts=prompts,
                batch_size=run_cfg.run.batch_size,
                gen=gen,
            )
            # Normalize to dicts so downstream row construction stays uniform
            gen_out = [{"generated_text": t} for t in gen_texts]

        if len(gen_out) != len(batch):
            raise RuntimeError(
                f"Got {len(gen_out)} outputs for batch size {len(batch)}"
            )

        for r, o in zip(batch, gen_out):
            generated_text = o["generated_text"]

            # Parse according to response_mode
            if contract.response_mode == "bit":
                parsed_label = str(runner.parse_bit(generated_text))
            elif contract.response_mode == "label":
                parsed_label = runner.parse_label(generated_text)
            else:
                parsed_label = _parse_justification_label(generated_text)

            row: Dict[str, Any] = {
                "row_id": r.row_id,
                "DOI": r.doi,
                "SDG": sdg_spec.number,
                "generated_text": _csv_one_line(generated_text),
                "parsed_label": parsed_label,
                "logit_0": None,
                "logit_1": None,
                "teacher_logit": None,
                "p1": None,
            }

            if contract.store_probs:
                row["logit_0"] = float(o["logit_0"])
                row["logit_1"] = float(o["logit_1"])
                row["teacher_logit"] = float(o["teacher_logit"])
                row["p1"] = float(o["p1"])

            rows.append(row)

    if args.dry_run:
        print(f"DRY RUN: processed {len(rows)} rows; no output written.")
        return

    out_csv = run_cfg.output_labels_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    print(f"Processed {len(rows)} rows; output written to {out_csv}")

if __name__ == "__main__":
    main()
