# Notebooks

This directory contains the exploratory/provenance notebooks used to build and inspect Scopus SDG datasets and the Virginia Tech ETD dataset before the scripted LLM labeling and distillation pipeline.

## Retrieval Provenance

The raw Scopus files were downloaded using the Elsevier SDG queries, with this suffix appended to each query to constrain corpus scope:

```text
 AND PUBYEAR > 2019 AND PUBYEAR < 2025 AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) ) AND ( LIMIT-TO ( AFFILCOUNTRY , "United States" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) ) AND ( LIMIT-TO ( SRCTYPE , "j" ) OR LIMIT-TO ( SRCTYPE , "p" ) )
```

This corresponds to:
- publication years 2020-2024,
- document types article (`ar`) or conference paper (`cp`),
- affiliation country United States,
- language English,
- source types journal (`j`) or conference proceeding (`p`).

Additional retrieval constraints:
- Scopus export limit of 20,000 records per query.
- Results were ranked by citation count before export to prioritize higher-impact papers and reduce noisy/less-relevant retrievals.

The SDG query set used for Scopus corresponds to:
- Jayabalasingham B, Boverhof R, Agnew K, Klein L. *Identifying Research Supporting the United Nations Sustainable Development Goals* (Elsevier Data Repository, V1, 2019). DOI: `10.17632/87txkw7khs.1`.

BibTeX:

```bibtex
@misc{jayabalasingham2019identifying,
  title        = {Identifying Research Supporting the United Nations Sustainable Development Goals},
  author       = {Bamini Jayabalasingham and Roy Boverhof and Kevin Agnew and Lisette Klein},
  year         = 2019,
  howpublished = {Elsevier Data Repository, V1},
  doi          = {10.17632/87txkw7khs.1},
  note         = {V1},
}
```

Provenance boundaries:
- The exact 17 base SDG query strings are Elsevier intellectual property and are referenced via the citation/DOI above (not reproduced here).
- Exact export timestamps were not retained.
- No post-export fixes were applied outside this repository; the only cleaning/transformation steps are those shown in these notebooks.

## Notebook Inventory

## `create_etd_dataset.ipynb`
- Purpose: Document and run the Virginia Tech ETD dataset build from VTechWorks OAI-PMH DIM feeds.
- Inputs:
  - Doctoral ETD OAI-PMH DIM endpoint:
    - `https://vtechworks.lib.vt.edu/server/oai/request?verb=ListRecords&metadataPrefix=dim&set=col_10919_11041`
  - Master's ETD OAI-PMH DIM endpoint:
    - `https://vtechworks.lib.vt.edu/server/oai/request?verb=ListRecords&metadataPrefix=dim&set=col_10919_9291`
  - Cached raw XML pages under:
    - `data/raw/vtechworks_oai/`
- Key steps:
  - Shows the exact `scripts/create_etd_dataset.py` invocation and the parameter values used for the build.
  - Reuses cached XML with `--skip-harvest` by default, but documents the live harvest endpoints explicitly.
  - Parses the requested DIM metadata fields into CSV, including `date_issued` from `dc.date.issued`.
  - Filters out rows with empty abstracts unless `--keep-empty-abstracts` is set.
  - Normalizes department labels into `department_normalized` using deterministic cleanup, parenthetical stripping, approved candidate merges, and a conservative approved-override map.
  - Performs an 80/20 department-stratified split using `seed=200`.
  - Keeps singleton departments in the training split.
  - Writes a raw-to-canonical department mapping audit table, including VT-major reference matches for the canonicalized label.
  - Writes a residual fuzzy-match candidate table for manual review of possible remaining department variants; this file is rewritten on each run and may be empty after approved merges are applied.
  - Computes ETD abstract length summary statistics in words and BERT tokens for comparison with the Scopus corpus.
  - Summarizes output row counts and department coverage after the run.
- Outputs:
  - `data/processed/vtechworks/vt_etds_all.csv`
  - `data/processed/vtechworks/vt_etds_train.csv`
  - `data/processed/vtechworks/vt_etds_test.csv`
  - `data/processed/vtechworks/vt_etds_department_summary.csv`
  - `data/processed/vtechworks/vt_etds_department_mapping_audit.csv`
  - `data/processed/vtechworks/vt_etds_department_similarity_candidates.csv`

## `create_scopus_data_per_sdg.ipynb`
- Purpose: Build per-SDG train/test CSVs used by the labeling pipeline.
- Inputs:
  - `data/raw/scopus/SDG01.csv` ... `data/raw/scopus/SDG17.csv`
- Key steps:
  - Cleans `Abstract` text (regex-based boilerplate/copyright cleanup).
  - Drops records with missing/empty abstract/title.
  - Keeps `["DOI", "Abstract", "SDG"]`.
  - Performs per-SDG stratified split using `StratifiedShuffleSplit(test_size=0.2, random_state=200)`.
- Outputs:
  - `data/processed/sdg1_2023_train.csv` ... `sdg17_2023_train.csv`
  - `data/processed/sdg1_2023_test.csv` ... `sdg17_2023_test.csv`

## `create_new_scopus_dataset.ipynb`
- Purpose: Build a single deduplicated multi-label dataset across all 17 SDGs.
- Inputs:
  - `data/raw/scopus/SDG01.csv` ... `data/raw/scopus/SDG17.csv`
- Key steps:
  - Reads all SDG CSVs and applies abstract cleaning.
  - Drops rows with invalid title/abstract.
  - Deduplicates by grouping on `DOI` + `Title`.
  - Uses Levenshtein consistency checks for grouped text fields.
  - Builds 17D multi-label target vector in `SDG Labels`.
  - Sets `Article Text` (current notebook version uses `Abstract`).
  - Splits with `MultilabelStratifiedShuffleSplit(test_size=0.2, random_state=200)`:
    - first train/test,
    - then train/val from the train split.
- Outputs:
  - CSV:
    - `data/processed/scopus17_2023_train.csv`
    - `data/processed/scopus17_2023_val.csv`
    - `data/processed/scopus17_2023_test.csv`
  - Parquet:
    - `data/processed/scopus17_2023_train.parquet`
    - `data/processed/scopus17_2023_val.parquet`
    - `data/processed/scopus17_2023_test.parquet`

## `explore_scopus_dataset.ipynb`
- Purpose: EDA for the multi-label dataset.
- Inputs:
  - `data/processed/scopus17_2023_train.parquet`
  - `data/processed/scopus17_2023_val.parquet`
  - `data/processed/scopus17_2023_test.parquet`
- Typical analyses:
  - Label prevalence per SDG.
  - SDGs-per-paper distribution.
  - Co-occurrence, correlation, clustering, and network plots.

## Dependencies Used in Notebooks

Some dependencies used directly in notebook cells (beyond core pandas/numpy/matplotlib):
- `seaborn`
- `python-Levenshtein`
- `iterative-stratification` (for `MultilabelStratifiedShuffleSplit`)
- `scikit-learn` (for `StratifiedShuffleSplit`)
- `transformers` (token length checks)
- `python-dotenv`

## Notes and Caveats

- These notebooks are best treated as provenance and exploration artifacts, not production pipelines.
- The notebook outputs include traces of historical environment issues (for example, missing `Levenshtein` in one run).
- `explore_scopus_dataset.ipynb` contains at least one cell assuming 16 SDGs while data uses 17 labels; this is exploratory and should be corrected if reused.
- Potential future work for the ETD normalization pipeline is to use embedding-based retrieval against the VT major list as a candidate-generation aid, while keeping final canonicalization decisions explicit and auditable.
