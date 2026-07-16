# Paper-to-software manifest

This manifest maps the active JCDL manuscript to the retained software. Static editorial diagrams are part of the manuscript repository and are not duplicated here.

| Paper component | Primary software |
|---|---|
| Scopus corpus preparation and fixed splits | `notebooks/create_scopus_data_per_sdg.ipynb`, `notebooks/create_new_scopus_dataset.ipynb`, `scripts/build_factorial_split_dataset.py` |
| VTechWorks ETD harvesting and metadata normalization | `scripts/create_etd_dataset.py`, `scripts/find_etd_department_candidates.py`, `notebooks/create_etd_dataset.ipynb` |
| Goal definitions and official target text used by prompts | `configs/sdgs.yaml` |
| Goal-conditioned Qwen teacher | `src/sdg_relevance_labeling/`, `scripts/labeling/`, `configs/runs/qwen2.5_7b_binary_bit_with_probs_v1.yaml` |
| SDG 1 response-format experiment | four publication configs in `configs/runs/`, `scripts/flip_overlap_analysis.py`, `scripts/build_flip_category_margin_figure.py` |
| Teacher-grid assembly | `scripts/combine_teacher_grid.py`, `scripts/combine_vtechworks_teacher_grid.py` |
| Student-ready supervision matrices | `scripts/build_student_ready_dataset.py`, `scripts/build_vtechworks_student_ready_dataset.py` |
| Teacher-margin distribution figure | `scripts/analyze_sdg_logit_distributions.py`, `scripts/build_publication_teacher_logit_distribution_figure.py` |
| Duplicate-evaluation repeatability table | `scripts/analyze_duplicate_teacher_repeatability.py`, `scripts/build_duplicate_teacher_logit_delta_summary.py` |
| Qualitative 50-abstract audit sampling | `scripts/build_publication_teacher_qa_sample.py` |
| Plain, hard, weighted, and masked losses | `src/sdg_distillation/training/losses.py`, `configs/train/factorial/` |
| Three-seed publication experiments | `scripts/run_publication_loss_seeds.py` |
| Validation soft-BCE table | `scripts/build_factorial_ablation_report.py`, `scripts/build_publication_loss_validation_table.py` |
| Intervention coverage fractions | `scripts/analyze_intervention_coverage.py` |
| Confidence-bin evaluation | `src/sdg_distillation/reporting/confidence_bins.py`, `scripts/build_confidence_binned_report.py` |
| Teacher entropy table | `scripts/analyze_publication_teacher_entropy.py` |
| Student training and checkpoint evaluation | `src/sdg_distillation/training/` |
| ETD zero-shot, adaptation, and scratch inference | `scripts/run_student_inference.py`, `configs/train/domain_transfer/` |
| ETD transfer point estimates | `src/sdg_distillation/reporting/etd_transfer_summary.py`, `scripts/build_etd_transfer_report.py`, `scripts/build_etd_transfer_performance_paper_table.py` |
| ETD paired bootstrap | `scripts/analyze_etd_bootstrap_transfer.py`, `scripts/build_etd_transfer_bootstrap_paper_table.py` |
| Department-level ETD profiles | `src/sdg_distillation/reporting/etd_institutional_profile.py`, `scripts/build_etd_institutional_profile_report.py` |
| Runtime/throughput instrumentation | `src/sdg_distillation/training/train.py` (`mode=throughput_probe`); historical paper times require external logs |
