#!/usr/bin/env python3

# Description: Run one evaluation SDG prompt across many input CSV files
# while reusing a single loaded model, and write one output CSV per input
# with optional dry-run prompt inspection.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/labeling/run_all.py [--run-config PATH --inputs PATH [PATH ...] | --inputs-dir DIR --inputs-glob GLOB --output-dir DIR --run-tag NAME --overwrite --limit N --dry-run --print-sample-prompt-once]

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from sdg_relevance_labeling.config import RunConfig, load_run_config, load_sdgs
from sdg_relevance_labeling.data import iter_batches, load_scopus_csv
from sdg_relevance_labeling.inference import GenerationArgs, QwenGenerateRunner
from sdg_relevance_labeling.prompting import QwenPromptBuilder

_JUSTIFICATION_CLASS_RE = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?\s*classification\s*(?:\*\*)?\s*:\s*(relevant|non-relevant)\s*\.?\s*$"
)
_ANY_LABEL_RE = re.compile(r"(?i)\b(non-relevant|relevant)\b")

# For naming outputs from retrieval-set filenames like sdg7_2023_train.csv
_RETRIEVAL_SDG_RE = re.compile(r"(?i)^sdg(\d+)[_-](.+)$")


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
    if len(hits_norm) == 1:
        return "Non-Relevant" if hits_norm[0].startswith("non") else "Relevant"

    raise ValueError(
        "Unexpected justification output: could not extract a single Classification label."
    )


def _csv_one_line(s: str) -> str:
    """Force 1 physical line per record. Reversible."""
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def _infer_output_basename(
    input_csv: Path,
    eval_sdg_number: str,
    run_tag: str,
) -> str:
    """
    Derive a stable, provenance-rich basename from the input filename.

    If input stem looks like: sdg7_2023_train
      -> sdg7xsdg{eval}_2023_train__{run_tag}

    Else:
      -> {stem}__EVAL_sdg{eval}__{run_tag}
    """
    stem = input_csv.stem
    m = _RETRIEVAL_SDG_RE.match(stem)
    if m:
        retrieval_sdg = m.group(1)
        rest = m.group(2)
        return f"sdg{retrieval_sdg}xsdg{eval_sdg_number}_{rest}__{run_tag}"
    return f"{stem}__EVAL_sdg{eval_sdg_number}__{run_tag}"


def _discover_inputs(
    inputs_dir: Optional[Path],
    inputs_glob: Optional[str],
    explicit_inputs: Sequence[Path],
) -> List[Path]:
    if explicit_inputs:
        out = [p for p in explicit_inputs]
        for p in out:
            if not p.exists():
                raise FileNotFoundError(p)
        return out

    if inputs_dir is None:
        raise ValueError("Provide either --inputs, or --inputs-dir (+ optional --inputs-glob).")

    pattern = inputs_glob or "*.csv"
    out = sorted(inputs_dir.glob(pattern))
    if not out:
        raise FileNotFoundError(f"No inputs matched: dir={inputs_dir} glob={pattern!r}")
    return out


def _run_single_csv(
    *,
    run_cfg: RunConfig,
    input_csv: Path,
    output_csv: Path,
    runner: QwenGenerateRunner,
    prompt_builder: QwenPromptBuilder,
    gen: GenerationArgs,
    limit: Optional[int],
    dry_run: bool,
) -> int:
    """
    Process exactly one CSV end-to-end (load -> batch -> generate -> parse -> write).
    Returns the number of processed rows.
    """
    contract = prompt_builder.contract

    df = load_scopus_csv(
        input_csv,
        doi_col=run_cfg.input.doi_col,
        abstract_col=run_cfg.input.text_col,
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        df = df.head(limit)

    # Dry-run: print one interpolated prompt for this file (first row after limiting).
    if dry_run and len(df) > 0:
        sample_abs = str(df.iloc[0][run_cfg.input.text_col] or "")
        print(f"\n=== DRY-RUN SAMPLE PROMPT for {input_csv.name} ===")
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

    desc = f"{input_csv.name} | eval SDG {run_cfg.run.sdg_number}"
    for batch in tqdm(batch_iter, total=n_batches, desc=desc, unit="batch"):
        prompts = [prompt_builder.build(r.abstract) for r in batch]

        if contract.store_probs:
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
            gen_out = [{"generated_text": t} for t in gen_texts]

        if len(gen_out) != len(batch):
            raise RuntimeError(f"Got {len(gen_out)} outputs for batch size {len(batch)}")

        for r, o in zip(batch, gen_out):
            generated_text = o["generated_text"]

            # Parse according to response_mode
            if contract.response_mode == "bit":
                parsed_label = runner.parse_bit(generated_text)  # returns "Relevant"/"Non-Relevant"
            elif contract.response_mode == "label":
                parsed_label = runner.parse_label(generated_text)
            else:
                parsed_label = _parse_justification_label(generated_text)

            row: Dict[str, Any] = {
                "row_id": r.row_id,
                "DOI": r.doi,
                "SDG": run_cfg.run.sdg_number,
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

    if dry_run:
        print(f"DRY RUN: processed {len(rows)} rows from {input_csv.name}; no output written.")
        return len(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Processed {len(rows)} rows; output written to {output_csv}")
    return len(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for multi-file labeling runs."""

    p = argparse.ArgumentParser(
        description=(
            "Run one SDG prompt over MANY input CSVs, reusing the same loaded model, "
            "and write one output CSV per input."
        )
    )
    p.add_argument("--run-config", required=True, type=Path, help="Path to run config YAML.")

    # Input discovery
    p.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[],
        help="Explicit list of input CSV paths. If provided, ignores --inputs-dir/--inputs-glob.",
    )
    p.add_argument(
        "--inputs-dir",
        type=Path,
        default=None,
        help="Directory containing input CSV files (used if --inputs is not provided).",
    )
    p.add_argument(
        "--inputs-glob",
        type=str,
        default="*.csv",
        help="Glob pattern under --inputs-dir (default: *.csv).",
    )

    # Output control
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output.dir from YAML (outputs still go to per-input basenames).",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Tag appended to basenames (default: output.basename from YAML). "
            "Example: qwen2.5_7b_binary_bit_with_probs_v1"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist (default: skip existing outputs).",
    )

    # Execution controls
    p.add_argument("--limit", type=int, default=None, help="Override max_rows for quick tests.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and parsing but do not write CSVs (prints one sample prompt per file).",
    )
    p.add_argument(
        "--print-sample-prompt-once",
        action="store_true",
        help="In dry-run, print a sample prompt for only the first file (instead of every file).",
    )
    return p.parse_args()


def main() -> None:
    """Load the model once, label each discovered input CSV, and write outputs."""

    args = parse_args()
    run_cfg = load_run_config(args.run_config)

    # Load SDG spec and build the prompt ONCE for this session.
    sdgs = load_sdgs(run_cfg.sdgs_path)
    eval_sdg_num = run_cfg.run.sdg_number
    if eval_sdg_num not in sdgs:
        known = ", ".join(sorted(sdgs.keys(), key=lambda x: int(x)))
        raise ValueError(f"Unknown SDG {eval_sdg_num!r}. Known SDGs: {known}")
    sdg_spec = sdgs[eval_sdg_num]

    # Discover input files.
    inputs = _discover_inputs(args.inputs_dir, args.inputs_glob, args.inputs)
    print(f"Discovered {len(inputs)} input CSVs.")

    # Load model ONCE.
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

    gen = GenerationArgs(
        max_new_tokens=run_cfg.generation.max_new_tokens,
        return_full_text=run_cfg.generation.return_full_text,
        temperature=run_cfg.generation.temperature,
        do_sample=run_cfg.generation.do_sample,
    )

    output_dir = args.output_dir if args.output_dir is not None else run_cfg.output.dir
    run_tag = args.run_tag if args.run_tag is not None else run_cfg.output.basename

    printed_once = False
    total_rows = 0
    n_done = 0
    n_skipped = 0

    for input_csv in inputs:
        out_base = _infer_output_basename(
            input_csv=input_csv,
            eval_sdg_number=eval_sdg_num,
            run_tag=run_tag,
        )
        output_csv = output_dir / f"{out_base}.csv"

        if output_csv.exists() and not args.overwrite and not args.dry_run:
            print(f"SKIP (exists): {output_csv}")
            n_skipped += 1
            continue

        dry_run_for_this_file = args.dry_run
        if args.dry_run and args.print_sample_prompt_once:
            dry_run_for_this_file = not printed_once

        n_rows = _run_single_csv(
            run_cfg=run_cfg,
            input_csv=input_csv,
            output_csv=output_csv,
            runner=runner,
            prompt_builder=prompt_builder,
            gen=gen,
            limit=args.limit,
            dry_run=dry_run_for_this_file,
        )
        if dry_run_for_this_file:
            printed_once = True

        total_rows += n_rows
        n_done += 1

    print(
        f"\nDONE. files_processed={n_done} files_skipped={n_skipped} total_rows_processed={total_rows}"
        + (" (dry-run; no files written)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
