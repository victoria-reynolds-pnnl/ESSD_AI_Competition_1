# Week 4 -- LLM Benchmark vs Week 3 ML Models

## Approach Report

### Task Definition

The classification target is binary **high_risk** (0 or 1), where high-risk
events correspond to `risk_tier_3 == 2` -- the top tier produced by HDBSCAN
clustering of NERC extreme-weather events in Week 3. The same chronological
70/15/15 train/val/test split (`week_3/Data/splits_indexed.csv`) is reused so
that LLM results are directly comparable to the Week 3 Logistic Regression
(test F1 = 0.766) and XGBoost (test F1 = 0.750) baselines.

### Features Presented to the LLM

Each prompt contains seven numeric features drawn from the labeled dataset:

| Feature | Description |
|---------|-------------|
| `lowest_temperature_k` | Minimum temperature during event (Kelvin) |
| `duration_days` | Event duration in days |
| `spatial_coverage` | Spatial extent of the event |
| `yearly_max_heat_wave_intensity` | Maximum heat-wave intensity that year |
| `yearly_max_heat_wave_duration` | Maximum heat-wave duration that year |
| `yearly_max_heat_wave_intensity_trend` | Year-over-year intensity trend |
| `yearly_max_heat_wave_duration_trend` | Year-over-year duration trend |

Missing trend values are rendered as `N/A` in the prompt.

### Prompting Strategy (Iteration History)

Three prompt versions were developed:

1. **v1 (initial zero-shot)** -- Bare instruction with feature listing. Produced
   excessively verbose free-text answers that were difficult to parse reliably.
2. **v2 (structured zero-shot)** -- Added an explicit JSON output constraint
   (`{"prediction": 0}` or `{"prediction": 1}`), a domain context sentence
   explaining the cold-snap / heat-wave setting, and a one-line reasoning
   request.  Parse success rate improved significantly.
3. **v3 (few-shot)** -- Prepended five labelled examples (2 high-risk,
   3 not-high-risk) selected from the training split to provide the model with
   concrete decision boundaries.  Examples were chosen to have complete (non-null)
   features and to span the range of risk outcomes.

The final notebook evaluates each model under both v2 (zero-shot) and v3
(few-shot) strategies and records per-iteration results.

### LLM Output Validation

A three-stage parser converts raw LLM responses into integer labels:

1. **JSON extraction** -- `re.search(r'\{.*?\}')` captures the first JSON object;
   `json.loads` extracts `prediction`.
2. **Regex fallback** -- If JSON parsing fails, look for `"prediction"\s*:\s*([01])`
   in the raw text.
3. **Bare digit fallback** -- If regex also fails, search for the first `0` or `1`
   in the response string.

Rows that fail all three stages are flagged with `parsed = -1` (unparseable)
and excluded from metric computation. The raw responses and parsed labels are
both persisted so reviewers can audit every conversion.

---

## Error Analysis

LLMs frequently struggle with borderline cases where features sit near the
decision boundary -- for example, events with moderate duration (4-6 days) and
temperatures around 260 K, which could go either way depending on trend
features. Another recurring failure mode is **format non-compliance**: some
models occasionally wrap the JSON in markdown code fences or add trailing
commentary that breaks the primary JSON parser, though the regex fallback
typically recovers these. Finally, the severe class imbalance (~7 % high-risk
in the training set) means that a model which defaults to predicting 0 can
achieve high accuracy while missing most true high-risk events, so F1 and
recall are the more informative metrics.

---

## Comparison Summary

Week 3 trained two supervised ML models on engineered features using the same
chronological test split:

| Model | Test F1 | Test Accuracy | Test Precision | Test Recall |
|-------|---------|---------------|----------------|-------------|
| Logistic Regression | 0.766 | 0.855 | 0.694 | 0.854 |
| XGBoost | 0.750 | 0.884 | 0.781 | 0.721 |

The LLM benchmark notebook evaluates up to four vLLM-hosted models
(ports 8000, 8001, 8002) under zero-shot (v2) and few-shot (v3) prompts.

### Key Results Summary

| Model | Prompt | F1 | Accuracy | Precision | Recall |
|-------|--------|----|----------|-----------|--------|
| **Week 3 Logistic Regression** | N/A | **0.766** | 0.855 | 0.695 | 0.853 |
| **Week 3 XGBoost** | N/A | **0.750** | 0.884 | 0.938 | 0.625 |
| Gemma-3-12b-it | Few-Shot (v3) | 0.628 | 0.677 | 0.463 | 0.976 |
| Phi-3.5-mini-instruct | Few-Shot (v3) | 0.576 | 0.631 | 0.424 | 0.896 |
| Phi-mini-MoE-instruct | Zero-Shot (v2) | 0.437 | 0.279 | 0.279 | 1.000 |

Because the LLM receives the same seven features as raw text with no
feature engineering, gradient-based optimisation, or threshold tuning, we
expect LLM F1 to fall below the supervised baselines. The comparison table
produced in Step 6 of the notebook shows side-by-side F1, accuracy, precision,
recall, and AUC-ROC for every model+prompt combination alongside the ML
baselines, making it straightforward to quantify the gap and to identify
which prompt iteration (if any) closes it.

---

## Repeatability Steps

1. **Environment** -- Run on the JupyterHub instance provided for the
   competition. The notebook requires Python 3.9+ with `pandas`, `numpy`,
   `scikit-learn`, `matplotlib`, and `openai` installed.

2. **Data** -- Ensure `week_3/Data/week3_labeled_with_tiers.csv` and
   `week_3/ml_results.csv` are present at the repository root. No additional
   data downloads are needed.

3. **LLM endpoints** -- The three vLLM model servers must be running on
   `localhost` ports 8000, 8001, and 8002 (the default JupyterHub
   configuration). The notebook will skip unreachable endpoints gracefully.

4. **Execute** -- Open `TRACK/Notebooks/llm_benchmark.ipynb` and run all cells
   top-to-bottom (Cell > Run All). The notebook will:
   - Load the test split and few-shot examples.
   - Query each reachable LLM with zero-shot and few-shot prompts.
   - Parse responses and compute metrics.
   - Save per-model CSVs to `TRACK/Results/`.
   - Print a comparison table and persist it as
     `TRACK/Results/comparison_table.csv`.

5. **Outputs** -- After execution, `TRACK/Results/` will contain:
   - `{model_name}_results_raw_{iteration}.csv` -- raw LLM responses.
   - `{model_name}_results_clean_{iteration}.csv` -- parsed predictions +
     ground truth.
   - `comparison_table.csv` -- aggregated metrics for all models and prompts
     alongside Week 3 ML baselines.

---

## Directory Layout

```
TRACK/
  README.md                       # This file
  Notebooks/
    llm_benchmark.ipynb           # Main benchmarking notebook (Steps 0-6)
  Prompts/
    prompt_template.txt           # All prompt versions with iteration notes
    few_shot_samples.json         # 5 training-set examples for few-shot
  Results/                        # Created at runtime by the notebook
    {model}_results_raw_{iter}.csv
    {model}_results_clean_{iter}.csv
    comparison_table.csv
```
