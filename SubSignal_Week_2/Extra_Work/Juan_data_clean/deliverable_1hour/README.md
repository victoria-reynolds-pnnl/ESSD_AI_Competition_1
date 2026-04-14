# Deliverable: 1-Hour FTES Data

## Contents
- `data/original/FTES-Full_Test_1hour_avg.csv`
  - Original hourly FTES source dataset
- `data/cleaned/ftes_1hour_cleaned.csv`
  - Audit-preserving cleaned output with normalized schema and QA flags
- `data/cleaned/ftes_1hour_ml_ready.csv`
  - ML-ready cleaned output with numeric-compatible sensor fields, encoded boolean/triplex-style state fields where applicable, and QA flags
- `qa/ftes_1hour_summary.json`
  - Full-run QA summary for the hourly dataset
- `scripts/cleaning.py`
  - Cleaning script used to generate the outputs
- `scripts/requirements.txt`
  - Environment file for the cleaning workflow
- `docs/data_dictionary.md`
  - Submission-facing data dictionary
- `docs/data_preparation_paragraph.md`
  - Submission-ready data preparation paragraph

## Scope
This deliverable is specific to the 1-hour FTES dataset.
It does not include the 1-second source file or any 1-second cleaned outputs.

## Recommended File For Downstream ML Work
Use `data/cleaned/ftes_1hour_ml_ready.csv` as the normalized starting table for downstream modeling preparation.
It is suitable for feature engineering and chronological train/validation splitting, but duplicate timestamps and time-gap anomalies should still be resolved before fitting a model.

## Important QA Note
The hourly source dataset contains duplicate timestamps with materially different sensor values. These rows were retained and flagged as source-data anomalies rather than silently deduplicated.
