#!/usr/bin/env python3

# Description: Sweep evaluation SDGs by repeatedly invoking
# scripts/labeling/run_all.py with temporary run-config files that vary
# only run.sdg_number.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Mon Mar  9 08:09:44 EDT 2026

# Usage: python scripts/labeling/run_sweep.py [--run-config PATH --sdgs SPEC --inputs PATH [PATH ...] | --inputs-dir DIR --inputs-glob GLOB --output-dir DIR --run-tag NAME --overwrite --limit N --dry-run --print-sample-prompt-once --continue-on-error]

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml


def _parse_sdg_spec(spec: str) -> list[int]:
    """
    Parse SDG selections like:
      - "1-17"
      - "1,3,5"
      - "1-3,7,9-10"
    """
    out: set[int] = set()
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if start > end:
                raise ValueError(f"Invalid range {p!r}: start > end")
            out.update(range(start, end + 1))
        else:
            out.add(int(p))
    if not out:
        raise ValueError("No SDGs parsed from --sdgs")
    return sorted(out)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the eval-SDG sweep wrapper."""

    p = argparse.ArgumentParser(
        description=(
            "Run scripts/labeling/run_all.py repeatedly across eval SDGs without "
            "manually editing run.sdg_number in YAML."
        )
    )
    p.add_argument(
        "--run-config",
        required=True,
        type=Path,
        help="Base run config YAML. This file is never modified in place.",
    )
    p.add_argument(
        "--sdgs",
        type=str,
        default="1-17",
        help="Eval SDGs to sweep (e.g., '1-17' or '1,3,5'). Default: 1-17.",
    )

    # Mirror run_all input discovery.
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

    # Mirror run_all output control.
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output.dir from YAML.",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Override output basename tag passed through to run_all.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to each run_all invocation.",
    )

    # Mirror run_all execution controls.
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override max_rows for quick tests.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to run_all.",
    )
    p.add_argument(
        "--print-sample-prompt-once",
        action="store_true",
        help="Pass --print-sample-prompt-once to run_all.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining SDGs if one run fails.",
    )
    return p.parse_args()


def _build_forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []

    if args.inputs:
        forwarded.extend(["--inputs", *[str(p) for p in args.inputs]])
    elif args.inputs_dir is not None:
        forwarded.extend(["--inputs-dir", str(args.inputs_dir)])
        if args.inputs_glob is not None:
            forwarded.extend(["--inputs-glob", args.inputs_glob])

    if args.output_dir is not None:
        forwarded.extend(["--output-dir", str(args.output_dir)])
    if args.run_tag is not None:
        forwarded.extend(["--run-tag", args.run_tag])
    if args.overwrite:
        forwarded.append("--overwrite")
    if args.limit is not None:
        forwarded.extend(["--limit", str(args.limit)])
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.print_sample_prompt_once:
        forwarded.append("--print-sample-prompt-once")
    return forwarded


def _load_base_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError("run config must be a mapping")
    if "run" not in doc or not isinstance(doc["run"], dict):
        raise ValueError("run config is missing mapping key 'run'")
    return doc


def _validate_sdgs(sdgs: Iterable[int], base_cfg: dict) -> list[int]:
    # Prefer validating against the SDG YAML referenced by the run config.
    sdgs_path_raw = base_cfg.get("sdgs_path")
    if not isinstance(sdgs_path_raw, str) or not sdgs_path_raw.strip():
        return list(sdgs)

    sdgs_path = Path(sdgs_path_raw)
    if not sdgs_path.exists():
        return list(sdgs)

    with sdgs_path.open("r", encoding="utf-8") as f:
        sdg_doc = yaml.safe_load(f)
    if not isinstance(sdg_doc, dict):
        return list(sdgs)
    sdg_map = sdg_doc.get("sdgs")
    if not isinstance(sdg_map, dict):
        return list(sdgs)
    valid = {int(str(k)) for k in sdg_map.keys()}

    bad = [s for s in sdgs if s not in valid]
    if bad:
        raise ValueError(f"Requested SDGs not in {sdgs_path}: {bad}")
    return list(sdgs)


def main() -> None:
    """Generate temporary configs per eval SDG and invoke run_all for each one."""

    args = parse_args()
    run_all_path = Path(__file__).with_name("run_all.py")
    if not run_all_path.exists():
        raise FileNotFoundError(f"Could not find run_all.py at {run_all_path}")

    base_cfg = _load_base_config(args.run_config)
    sdgs = _validate_sdgs(_parse_sdg_spec(args.sdgs), base_cfg)
    forwarded = _build_forwarded_args(args)

    failures: list[tuple[int, int]] = []

    for sdg in sdgs:
        cfg = deepcopy(base_cfg)
        cfg["run"]["sdg_number"] = str(sdg)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f".eval_sdg{sdg}.yaml",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = Path(tmp.name)
            yaml.safe_dump(cfg, tmp, sort_keys=False)

        cmd = [
            sys.executable,
            str(run_all_path),
            "--run-config",
            str(tmp_path),
            *forwarded,
        ]

        print(f"\n=== SWEEP eval_sdg={sdg} ===")
        print(" ".join(cmd))

        rc = 0
        try:
            proc = subprocess.run(cmd, check=False)
            rc = int(proc.returncode)
        finally:
            tmp_path.unlink(missing_ok=True)

        if rc != 0:
            failures.append((sdg, rc))
            msg = f"eval_sdg={sdg} failed with exit_code={rc}"
            if args.continue_on_error:
                print(f"WARNING: {msg}; continuing.")
                continue
            raise RuntimeError(msg)

    if failures:
        joined = ", ".join([f"sdg{s}(rc={rc})" for s, rc in failures])
        print(f"\nDONE with failures: {joined}")
        raise SystemExit(1)

    print(f"\nDONE. Completed sweep for eval SDGs: {sdgs}")


if __name__ == "__main__":
    main()
