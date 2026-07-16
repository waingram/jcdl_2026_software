# tests/test_config_loading.py
from __future__ import annotations

from pathlib import Path

import pytest

from sdg_relevance_labeling.config import load_run_config, load_sdgs


def test_load_sdgs_mapping_format(tmp_path: Path):
    sdgs_yaml = tmp_path / "sdgs.yaml"
    sdgs_yaml.write_text(
        """
version: 1
sdgs:
  "1":
    title: "End poverty in all its forms everywhere"
    description: "poverty eradication, social protection, and access to resources"
    targets_md: |-
      **Target 1.1**: By 2030, eradicate extreme poverty...
  "2":
    title: "End hunger"
    description: "food security"
    targets: "Target 2.1..."
""".strip(),
        encoding="utf-8",
    )

    sdgs = load_sdgs(sdgs_yaml)
    assert set(sdgs.keys()) == {"1", "2"}
    assert sdgs["1"].number == "1"
    assert "Target 1.1" in sdgs["1"].targets
    assert sdgs["2"].targets.startswith("Target 2.1")


def test_load_run_config_minimal(tmp_path: Path):
    # Create files we reference in the config.
    sdgs_yaml = tmp_path / "sdgs.yaml"
    sdgs_yaml.write_text("version: 1\nsdgs: {}\n", encoding="utf-8")

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        f"""
version: 1
sdgs_path: "{sdgs_yaml.as_posix()}"

model:
  name: "Qwen/Qwen2.5-7B-Instruct"

input:
  path: "{(tmp_path / 'input.csv').as_posix()}"
  doi_col: "DOI"
  text_col: "Abstract"

output:
  dir: "{(tmp_path / 'outputs').as_posix()}"
  basename: "scopus_sdg1_qwen_binary_labels_v1"

run:
  sdg_number: "1"
  batch_size: 8

prompt:
  variant: "binary_bit"
  include_contribution_types: true
  include_indirect_clause: true

generation:
  max_new_tokens: 3
  do_sample: false
  temperature: 0.0
""".strip(),
        encoding="utf-8",
    )

    cfg = load_run_config(run_yaml)
    assert cfg.version == 1
    assert cfg.sdgs_path == sdgs_yaml
    assert cfg.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.run.batch_size == 8
    assert cfg.prompt.variant == "binary_bit"
    assert cfg.generation.max_new_tokens == 3
