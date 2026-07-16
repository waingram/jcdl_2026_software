#!/usr/bin/env python3
"""Run the publication loss conditions over the paper's three training seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys


CONDITIONS = {
    "plain": ("plain_bce_t1_v1", "sbce_split_plain_bce_t1_v1__s"),
    "weighted-t2": ("weighted_bce_t20_v1", "sbce_ext_weighted_bce_t20_v1__s"),
    "weighted-t1.25": (
        "weighted_bce_t125_v1",
        "sbce_split_weighted_bce_t125_v1__s",
    ),
    "masked": (
        "masked_bce_t1_tau50_v1",
        "sbce_ext_masked_bce_t1_tau50_v1__s",
    ),
    "hard": ("hard_label_v1", "sbce_split_hard_label_v1__s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=list(CONDITIONS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides applied to every run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for condition in args.conditions:
        config_name, run_prefix = CONDITIONS[condition]
        for seed in args.seeds:
            command = [
                sys.executable,
                "-m",
                "sdg_distillation.training.train",
                f"--config-name=factorial/{config_name}",
                f"seed={seed}",
                f"logging.name={run_prefix}{seed}",
                *args.overrides,
            ]
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, check=False)
            if result.returncode != 0 and not args.continue_on_error:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
