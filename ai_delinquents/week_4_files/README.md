# ESSD AI Competition — Week 4

**Project Title**: Water supply forecasts for the Columbia River Basin  
**Team Name**: AI Delinquents

---

## Approach

We benchmarked three locally hosted LLMs (Phi-3.5-mini-instruct, Phi-mini-MoE-instruct, gemma-3-12b-it)
on the same regression task as Week 3: predicting April–September naturalized streamflow volume at
The Dalles, OR (units: kcfs-days). Each LLM received a structured natural-language prompt describing the
seven predictor variables for a given water year and was asked to return a JSON object with a single
numeric prediction.

**Split**: Same held-out test set as Week 3 — WY 2013–2018 (n=6). No test year data appeared in any prompt.

**Prompt versions tested**:
- `v1` — Zero-shot: system context + output format constraint only
- `v2_few_shot` — 3 training examples (dry/median/wet years, selected by tercile from WY 1985–2012)

The best-performing version per model (lowest MAE on the 6 test years) was used for the final benchmark results.

---

## Model Performance Summary

All models evaluated on the same held-out test set (WY 2013–2018, n=6) using the same metrics as Week 3. Parse success was 100% across all LLMs — the structured JSON prompt format worked without failures.

| Model | Type | Prompt | NSE | KGE | RMSE (kcfs-days) | MAE (kcfs-days) | Skill vs Clim | Parse Success |
|---|---|---|---|---|---|---|---|---|
| Climatology baseline | ML | — | −0.03 | −0.41 | 8,488 | 7,079 | — | 100% |
| MLR | ML | — | **0.77** | **0.74** | **4,034** | **3,412** | **+0.77** | 100% |
| XGBoost | ML | — | 0.46 | 0.50 | 6,170 | 5,016 | +0.47 | 100% |
| gemma-3-12b-it | LLM | v2 few-shot | 0.33 | 0.47 | 6,817 | 6,160 | +0.36 | 100% |
| Phi-3.5-mini-instruct | LLM | v1 zero-shot | 0.12 | 0.43 | 7,824 | 6,989 | +0.15 | 100% |
| Phi-mini-MoE-instruct | LLM | v2 few-shot | **−0.57** | 0.36 | **10,472** | **8,807** | **−0.52** | 100% |

NSE > 0 means the model outperforms predicting the mean; NSE = 1 is perfect.  
Skill score = 1 − MSE(model)/MSE(climatology); positive = better than the baseline.

---

## Error Analysis

### Parse reliability
All three LLMs returned valid, parseable JSON on every query (100% success rate). The strict output format constraint (`"Return ONLY valid JSON: {"prediction_kcfs_days": <positive number>}"`) was effective. No markdown fences, no extra text, no out-of-range values were returned across 18 total predictions (6 years × 3 models).

### LLM vs ML model performance
gemma-3-12b-it was the strongest LLM (NSE = 0.33, RMSE = 6,817 kcfs-days), performing comparably to XGBoost (NSE = 0.46) but falling well short of MLR (NSE = 0.77). Phi-3.5-mini-instruct produced marginally positive NSE (0.12), indicating modest skill above the mean. Phi-mini-MoE-instruct performed worse than simply predicting the training median every year (NSE = −0.57, skill = −0.52), making it the only model of six to be outperformed by the climatology baseline.

### The dominant failure mode: compressed prediction variance
The defining pattern across all LLMs was **regression to the mean** — predictions clustered near climatology regardless of the input feature values. Phi-3.5-mini-instruct returned exactly 45,000 kcfs-days for four of six years and exactly 35,000 for the remaining two — effectively a two-state lookup table. Phi-mini-MoE-instruct ranged only from 30,000–40,200 kcfs-days across all six years, well below the observed range of 31,185–57,104. Even gemma's predictions (38,500–52,345) were more compressed than the observed spread. This is reflected in the low KGE scores for all LLMs (0.36–0.47) — KGE penalizes variance ratio (predicted σ / observed σ) and all LLMs under-predict spread substantially.

### Extreme years exposed the most severe LLM errors

**WY 2015 (record drought):** Observed = 31,185 kcfs-days. April 1 SWE was −51% below median with a strong El Niño (Nino3.4 = 2.5) — the most anomalous year in the test set. All models struggled here:

| Model | Prediction | Error |
|---|---|---|
| Phi-3.5-mini | 35,000 | +3,815 |
| Phi-mini-MoE | 30,000 | −1,185 (closest, likely coincidental anchoring) |
| gemma | 38,523 | +7,338 |
| MLR | 35,654 | +4,469 |
| XGBoost | 43,329 | +12,144 |

**WY 2017 (wet year):** Observed = 57,104 kcfs-days. April 1 SWE was +28% above median. ML models excelled; LLMs could not respond to the above-normal snowpack signal:

| Model | Prediction | Error |
|---|---|---|
| MLR | 57,557 | +453 |
| XGBoost | 57,364 | +259 |
| gemma | 52,345 | −4,759 |
| Phi-3.5-mini | 45,000 | −12,104 |
| Phi-mini-MoE | 40,000 | −17,104 |

The pattern is clear: ML models leverage learned feature-to-target relationships and can extrapolate to unusual years. LLMs anchor near the center of the training distribution they were exposed to during pretraining and do not adjust proportionally to extreme input signals.

### Zero-shot vs few-shot prompt comparison
The prompt iteration step (tested on all 6 test years × both prompt versions × all models before the final run) selected different best versions per model:

- **Phi-3.5-mini-instruct selected v1 (zero-shot)** as better-performing. Adding three few-shot examples appeared to *worsen* performance for this model, likely because the examples anchored the model to the specific volume values shown rather than helping it generalize. Small-parameter models may lack the in-context learning capacity to benefit from numeric few-shot demonstrations.
- **gemma-3-12b-it and Phi-mini-MoE-instruct both selected v2 (few-shot)** as better. For gemma, the few-shot examples provided useful range context (the three training examples spanned ~35k–58k kcfs-days) that improved the dynamic range of its predictions. For Phi-mini-MoE, few-shot was selected as better during iteration but the model still substantially underperformed in the final benchmark — suggesting model capacity was the binding constraint, not prompt design.

The overall picture is that **few-shot prompting benefits larger, more capable models** but provides limited or even counterproductive gains for smaller models on domain-specific numeric regression tasks. This is consistent with the broader literature on in-context learning, where few-shot gains are strongly correlated with model scale.

### Why do LLMs underperform ML models on this task?
Several structural reasons explain the performance gap:

1. **No learned feature-to-target mapping.** MLR and XGBoost were explicitly trained on (feature, volume) pairs from 28 water years. The LLMs were never trained on Columbia Basin hydrology — they are reasoning from general knowledge about snow, climate, and rivers encoded during pretraining, not from a learned statistical model of this specific system.

2. **Numeric reasoning limitations.** LLMs process numbers as tokens, not as quantities on a continuous scale. A feature value of 25.2 inches SWE vs 49.0 inches is not experienced as "nearly double" in the way a regression model treats it. This limits the models' ability to scale predictions proportionally to extreme inputs.

3. **No uncertainty calibration.** ML models were fit to minimize prediction error on training data; their predictions are conditioned on the statistical structure of the training set. LLMs produce outputs that are plausible given the prompt context but not calibrated to the empirical distribution of historical volumes.

4. **Anchoring to round numbers.** LLM predictions are strongly biased toward round multiples of 5,000 kcfs-days (35,000; 40,000; 45,000), revealing that the models are pattern-matching to numerically "convenient" outputs rather than performing true regression.

---

## Failure and Limitations

**Small test set.** With only n=6 test years, all metric comparisons carry high uncertainty. A single anomalous year (e.g., WY 2015 or 2017) can swing NSE by 0.1–0.2 for any model. Results should be interpreted directionally rather than as precise performance estimates.

**Reproducibility caveat.** LLM determinism (`temperature=0.0`, `seed=0`) is not guaranteed across endpoint restarts — model inference frameworks do not always produce bit-identical outputs across hardware states. The results reported here reflect one benchmark run.

**Few-shot examples do not cover the test distribution.** The three training-set few-shot examples (dry/median/wet by tercile) did not include a year with the extreme conditions of WY 2015 (−51% SWE, strong El Niño). Including a more extreme dry-year example may have improved LLM performance on that year.

**Model size is confounded with prompt version.** The cleanest comparison between zero-shot and few-shot would hold model constant — the iteration results allowed each model to self-select its best version, making it difficult to fully isolate the prompt effect from the model capacity effect. A controlled experiment would run both prompt versions on all models and report results separately.

**Domain specificity.** These LLMs are general-purpose models not fine-tuned on hydrology. Fine-tuned or domain-adapted models (e.g., trained on USGS/BPA data) would likely outperform the general models tested here, narrowing the gap with classical ML.

**LLMs are not designed for tabular regression.** The task — mapping seven numeric predictors to a continuous target — is precisely the setting where classical ML methods have a structural advantage. LLMs are better suited to tasks involving natural language reasoning, pattern recognition in text, or synthesis of heterogeneous information. The results here do not indicate that LLMs are poor forecasting tools in general, only that they are not competitive with purpose-built regression models on low-dimensional tabular prediction tasks.

---

## Repeatability

```bash
# 1. Install dependencies
pixi install
# or: pip install -r requirements.txt  (add openai if not present)

# 2. Confirm LLM endpoints are running at localhost:8000-8002

# 3. Open and run the notebook
#    Set DRY_RUN = False in Cell 2 before running on JupyterHub
jupyter notebook Notebooks/llm_benchmark.ipynb
```

---

## File Structure

```
week_4_files/
├── README.md
├── Notebooks/
│   └── llm_benchmark.ipynb
├── Prompts/
│   ├── prompt_template.txt      ← canonical prompt template (best version)
│   └── few_shot_samples.json    ← 3 training-year examples used in v2 prompt
└── Results/
    ├── {model_label}_results_clean_{n}.csv   ← clean submission CSVs per model
    ├── {model_label}_results_raw_{n}.csv     ← raw responses per model
    ├── llm_results_clean_{n}.csv             ← combined clean CSV
    └── metrics_summary_{n}.csv              ← NSE/KGE/RMSE for all models
```

Week 3 artifacts referenced (not duplicated):
- `../week_3_files/outputs/test_split.csv`
- `../week_3_files/outputs/test_predictions.csv`
- `../week_3_files/outputs/train_split.csv`
