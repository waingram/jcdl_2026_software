# JCDL 2026 paper software

Software for **Beyond Topic Matching: Goal-Conditioned Evaluation of Research Contributions in ETD Collections**.

This directory is a paper-specific extraction of the dissertation codebase. It contains only source code, experiment configurations, tests, and code-only notebooks needed for the analyses reported in the JCDL manuscript.

## Important: no data or model artifacts

This package intentionally contains no:

- Scopus or ETD records
- abstracts, titles, DOIs, or repository metadata
- teacher labels, logits, or probabilities
- student predictions
- W&B run exports
- model checkpoints
- generated result tables or figures

Scopus inputs must be obtained under an appropriate institutional license. VTechWorks ETD metadata can be harvested with the included software. Model weights are downloaded separately from Hugging Face, and trained checkpoints must be regenerated or supplied by the authors.

The repository ignores `data/`, `outputs/`, `checkpoints/`, and `wandb/` so those artifacts cannot be added accidentally.

## Package layout

- `src/sdg_relevance_labeling/`: constrained Qwen teacher prompting and token scoring
- `src/sdg_distillation/`: DeBERTa student training, loss functions, evaluation, inference, and reporting
- `src/sdg_shared/`: shared plotting style
- `configs/runs/`: teacher prompt variants used in the paper
- `configs/train/`: publication loss conditions and ETD transfer conditions
- `configs/baselines/`: table/report definitions
- `scripts/`: dataset construction, experiment launchers, analyses, and paper table/figure builders
- `notebooks/`: code-only corpus preparation notebooks with outputs removed
- `tests/`: unit and smoke tests for the retained paper pipeline
- `MANIFEST.md`: paper claim and artifact to software mapping

## Environment

Python 3.12 is expected. GPU experiments were run with CUDA-enabled PyTorch; PyTorch is supplied by the NVIDIA base image and is deliberately excluded from the uv dependency resolution.

Recommended container setup:

```bash
docker build \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  --build-arg USER_NAME="$(id -un)" \
  -t jcdl-sdg-distillation .
```

For an existing CUDA/PyTorch environment:

```bash
uv sync --frozen
uv run pytest -q
```

If not using the container, install a PyTorch build appropriate for the local CUDA or CPU platform before `uv sync`.

## Expected local inputs

Paths below are conventions, not bundled content.

```text
data/
  processed/
    sdg1_2023_train.csv ... sdg17_2023_test.csv
    publication/
      student_ready_train.parquet
      student_ready_test.parquet
      student_ready_factorial_train.parquet
      student_ready_factorial_val.parquet
      student_ready_hard_factorial_train.parquet
      student_ready_hard_factorial_val.parquet
      student_ready_hard_test.parquet
    etd/
      vt_etds_train.csv
      vt_etds_test.csv
      student_ready_train.parquet
      student_ready_test.parquet
      student_ready_etd_adapt_train.parquet
      student_ready_etd_adapt_val.parquet
outputs/
  teacher/
    publication/
    etd/
  evaluation/
  inference/
  analysis/
  wandb/
```

All commands below are run from this directory.

## Reproduction workflow

### 1. Prepare corpora

The publication processing sequence is documented in:

- `notebooks/create_scopus_data_per_sdg.ipynb`
- `notebooks/create_new_scopus_dataset.ipynb`

The ETD collection can be harvested and normalized with:

```bash
uv run python scripts/create_etd_dataset.py --help
uv run python scripts/find_etd_department_candidates.py --help
```

### 2. Generate teacher supervision

Run the 17 goal-conditioned teacher passes:

```bash
uv run python scripts/labeling/run_sweep.py \
  --run-config configs/runs/qwen2.5_7b_binary_bit_with_probs_v1.yaml \
  --inputs-dir data/processed \
  --output-dir outputs/teacher/publication
```

The SDG 1 response-format experiment uses the justification, binary-label, binary-bit, and binary-bit-with-probabilities configs in `configs/runs/`.

Combine publication teacher outputs and build student-ready artifacts:

```bash
uv run python scripts/combine_teacher_grid.py --strict
uv run python scripts/build_student_ready_dataset.py --target-set both --strict
uv run python scripts/build_student_ready_dataset.py \
  --target-set hard_label \
  --output-stem student_ready_hard \
  --strict
uv run python scripts/build_factorial_split_dataset.py
```

For ETDs, use `combine_vtechworks_teacher_grid.py` and `build_vtechworks_student_ready_dataset.py`. The fixed ETD adaptation split is produced by the same split builder:

```bash
uv run python scripts/build_factorial_split_dataset.py \
  --soft-train-path data/processed/etd/student_ready_train.parquet \
  --hard-train-path data/processed/etd/student_ready_hard_train.parquet \
  --out-dir data/processed/etd \
  --seed 20260317 \
  --soft-stem student_ready_etd_adapt \
  --hard-stem student_ready_hard_etd_adapt \
  --summary-name etd_adapt_split_summary.json \
  --strata-name etd_adapt_split_strata.csv
```

### 3. Train publication students

The launcher reproduces the five paper conditions over seeds 17, 23, and 41:

```bash
uv run python scripts/run_publication_loss_seeds.py
```

Use `--dry-run` to inspect the generated Hydra commands. Additional Hydra overrides can be appended, for example `logging.offline=true`.

Build the seed-level summaries and compact paper table:

```bash
uv run python scripts/build_factorial_ablation_report.py \
  --config configs/baselines/factorial_ablation_validation_v1.yaml
uv run python scripts/build_factorial_ablation_report.py \
  --config configs/baselines/factorial_ablation_extension_v1.yaml
uv run python scripts/build_publication_loss_validation_table.py
uv run python scripts/analyze_intervention_coverage.py
```

### 4. Evaluate boundary-signal behavior

Evaluate each selected checkpoint with `mode=evaluate`, `evaluation.save_predictions=true`, and the report names expected by `configs/baselines/heldout_confidence_bins_v1.yaml`. Then run:

```bash
uv run python scripts/build_confidence_binned_report.py
uv run python scripts/analyze_publication_teacher_entropy.py
```

### 5. Reproduce teacher-signal diagnostics

Teacher margin distribution and ACM figure:

```bash
uv run python scripts/analyze_sdg_logit_distributions.py \
  --teacher-paths \
    outputs/teacher/publication/teacher_grid_wide_train.csv \
    outputs/teacher/publication/teacher_grid_wide_test.csv \
  --report-name teacher_signal
uv run python scripts/build_publication_teacher_logit_distribution_figure.py \
  --summary-path outputs/analysis/teacher_signal/teacher_logit_distribution_by_sdg.csv \
  --figure-profile acm
```

Duplicate-evaluation repeatability:

```bash
uv run python scripts/analyze_duplicate_teacher_repeatability.py --no-text
uv run python scripts/build_duplicate_teacher_logit_delta_summary.py
```

SDG 1 response-format overlap and margin categories:

```bash
uv run python scripts/flip_overlap_analysis.py --help
uv run python scripts/build_flip_category_margin_figure.py --help
```

The qualitative audit sample is generated by `build_publication_teacher_qa_sample.py`.

### 6. Reproduce ETD transfer and profiles

Train the adapted and scratch conditions with the configs in `configs/train/domain_transfer/`. For adaptation, set `initialization.checkpoint_path` to the publication plain-BCE checkpoint or retain the recorded W&B run-ID convention.

Run student inference for zero-shot, adapted, and scratch checkpoints, then build the transfer reports:

```bash
uv run python scripts/run_student_inference.py --help
uv run python scripts/build_etd_transfer_report.py
uv run python scripts/build_etd_transfer_performance_paper_table.py
uv run python scripts/analyze_etd_bootstrap_transfer.py
uv run python scripts/build_etd_transfer_bootstrap_paper_table.py
uv run python scripts/build_etd_institutional_profile_report.py
```

## Reproducibility boundaries

The code, hyperparameters, split seeds, prompt variants, and analysis definitions are included. Exact numerical reproduction additionally requires the licensed Scopus exports, the harvested ETD snapshot, model versions available at execution time, selected checkpoints, and compatible GPU software.

Runtime statements in the paper were reconstructed from archived file times and W&B logs. They are provenance observations, not outputs of a dedicated timing benchmark in this package. The training module retains a `mode=throughput_probe` option for new measurements.
