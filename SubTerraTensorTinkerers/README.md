# Sub‑Terra Tensor Tinkerers - Week 3 Submission

**Project Title:** Signal‑to‑Storage: Early Warning Forecasts for FTES Charging Operations

**Team:** Scott Unger, Julian Chesnutt, Theresa Pham, Xi Tan, Ashton Kirol, Mike Parker

## Test/Training Split Strategy
- **Dataset(s):**
  - Split-indexed: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_split_indexed.csv
  - Train only: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_train.csv
  - Validation only: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_val.csv
  - Test only: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_test.csv
- **Split method:** Time-based, chronological 
- **Split ratios:** 
  - 70% Train (1191 rows, 2024-12-13 20:00:00 -> 2025-02-01 10:00:00)
  - 15% Validation (255 rows, 2025-02-01 11:00:00 -> 2025-02-12 01:00:00)
  - 15% Test (256 rows, 2025-02-12 02:00:00 -> 2025-02-22 17:00:00)
- **Leakage prevention:** 
  - Aligned validation and test sets to the training feature schema
  - Missing values in the training set were imputed with medians calculated from the training set only (cleaning script from Week 2 was modified as well)
- **Rationale:** Since we are working with time-series data, randomization during the split could lead to data leakage from future timestamps. We simply kept the dataset in its original, chronological order and split into train/validation/test sequentially. 

## Model Performance Summary
- **Model/version:**
- **Evaluation dataset:**
- **Primary metric(s):**
- **Secondary metric(s):**
- **Overall results:** (table or bullets)
  - Metric: value
- **Per-slice / subgroup results:** (if applicable)
- **Calibration / thresholding:** (if applicable)
- **Comparison to baseline(s):** (if applicable)

## Failures and Limitations
- **Known failure cases:** (examples, patterns)
- **Out-of-scope scenarios:**
- **Data limitations:** (coverage, label noise, imbalance, missingness)
- **Model limitations:** (assumptions, brittleness, generalization concerns)
- **Operational constraints:** (latency, compute, dependencies)

## HITL Review
### Review Questions
- [ ] Does the model behavior make sense given domain expectations?
- [ ] Any concerning spurious correlations?
- [ ] Any high-risk failure modes?

### Findings
- **Summary of review:**
- **Examples reviewed:** (IDs/links, counts, date range)
- **Issues identified:**  
  - Issue:
  - Impact:
  - Evidence:
- **Mitigations/changes made:** (data, training, post-processing, UX)
- **Open questions / follow-ups:**
- **Reviewer(s) & date:**

## Reproducibility Steps
- **Repo/commit:**
- **Environment:** (OS, Python version, CUDA, etc.)
- **Dependencies:** (requirements file / lockfile)
- **Data access:** (location, version, checksum, permissions)
- **Setup:**
  1. 
  2. 
- **Train:**
  ```bash
  # command(s)