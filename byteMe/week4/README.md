# ByteMe Week 4: LLM Benchmarking vs ML Model
**Team:** ByteMe | **Competition:** ESSD AI Competition | **Due:** April 16, 2026

## Approach Report

### Task Definition

Predict temperature deltas (°C) at four horizons (+15, +60, +240, +1440 min) from a geothermal injection system. Each prediction is a continuous float returned in a strict JSON response: `{"target_15": <float>, "target_60": <float>, "target_240": <float>, "target_1440": <float>, "confidence": <0-1>}`.

### Input Features (15 total)

Base features: `cumulative_heat_input`, `elapsed_injection_min`, `net_flow_rolling_6h`, `TC_INT_delta`, `T_gradient_INT_TC`, `days_since_injection`, `hour_sin`, `hour_cos`, `delta_T_above_T0_TN`

Lag features: `target_lag_15`, `target_lag_30`, `target_lag_60`, `flow_lag_15`, `flow_lag_30`, `flow_lag_60`

### Models

Three LLMs served via vLLM on JupyterHub (phi-3.5-mini-instruct:8000, phi-mini-moe-instruct:8001, gemma-3-12b-it:8002), queried via the `openai` Python client. ML baseline: multi-output **XGBoost** from Week 3.

### Prompt Engineering Strategy

**9 prompt versions** tested across two dimensions: **format style** (verbose named key=value pairs vs compact numeric arrays) and **few-shot count** (zero-shot, 1, 5, 10, 20 examples). Each prompt has a system message (role, JSON schema, feature order), few-shot input→output pairs, and a user query. All few-shot examples are sampled exclusively from the training split (no label leakage). Templates stored in `Prompts/prompt_template_<model>.txt`; example sets in `Prompts/examples_<format>_<N>.json`.

### LLM Output and Validation

Responses are parsed via regex to extract JSON, then validated: all four target fields must be present, numeric, and within [-10, 10]. Failures are classified as `empty_response`, `invalid_json`, `missing_targets`, `non_numeric`, or `out_of_range`.

### Evaluation Pipeline

1. **Small-sample validation** (5 rows × 3 models × 9 versions = 135 queries): parse success rate + MSE.
2. **Best version selection**: rank by parse rate (primary), MSE (secondary).
3. **Full benchmark** (100 rows × 3 models): MSE and R² per horizon, computed identically to Week 3 (same formulas, same test split from `splits_indexed.csv`).

### Agentic Auto-Design Loop

Each LLM designs its own prompt via a meta-prompt (task description + 10 training examples + prior failures). The LLM returns `{"system_prompt": "...", "user_template": "...", "strategy_notes": "..."}`, which is validated (must contain `{features_json}` placeholder), tested on 5 samples, and iterated for 10 rounds per model. Results in `Prompts/auto_prompt_<model>.json` and `Results/agentic_round_history_<model>.json`.

---

## Error Analysis

The most common failure mode is **invalid JSON formatting**: models wrap output in markdown code fences (`` ```json ... ``` ``) or add explanatory text, breaking the parser. A second major failure is **zero-shot hallucination** — gemma on verbose_zeroshot produced MSE=32.8 vs 0.00015 with 10-shot (200,000× worse) despite 100% parse success, returning syntactically valid JSON with meaningless round-number values. Third, **sensitivity to prompt length**: phi-3.5's parse rate dropped from 100% to 60% simply by increasing from 10 to 20 compact examples, as the model switched from JSON to natural-language output (format drift). All models also report confidence of 0.85–1.0 regardless of accuracy (uncalibrated).

### Likely Causes

- **`invalid_json`**: Instruction-tuning biases models toward markdown code fences despite "respond ONLY with JSON" instructions.
- **Zero-shot hallucination**: No few-shot examples to anchor the numeric scale; models default to heuristic round numbers (5.0, 1.0).
- **Format drift (compact_20)**: Long prompts approach the context window limit, causing the model to lose track of formatting constraints.
- **Confidence miscalibration**: Few-shot examples all use `confidence: 1.0`, teaching the model to always report high confidence.

### Parse Success Rates (5-sample validation)

Two version/model combos had sub-100% parse success:

| Version | Model | Parse Rate | Issue |
|---|---|---|---|
| compact_20 | phi-3.5-mini-instruct | **60%** | Format drift — switched to natural language |
| verbose_zeroshot | phi-mini-moe-instruct | **80%** | Missing target fields |

Zero-shot MSE can be **100,000×** worse than few-shot even when parsing succeeds — the models produce syntactically valid JSON with numerically meaningless predictions.

### Agentic Loop Failures

- Some rounds produced templates missing the `{features_json}` placeholder.
- gemma converged to an identical strategy across all 10 rounds (no exploration).
- Agentic prompts did not outperform manual prompts on any model.

---

## LLM vs ML Comparison Summary

All three LLMs dramatically outperformed the Week 3 XGBoost baseline across all four horizons on the same 100-sample test set. The best LLM achieved 35× lower MSE at 15 min (R²=0.985 vs 0.485) and 3.7× lower at 1440 min (R²=0.895 vs 0.616). Few-shot prompting with ≥5 compact examples consistently delivered R²>0.97 through 240 min, while XGBoost peaked at R²=0.63. However, LLMs carry practical tradeoffs: JSON parse failures produce missing predictions (XGBoost always returns output), prompt sensitivity can collapse performance, inference is orders of magnitude slower, and predictions are opaque (no feature importances). **Note:** the large LLM-vs-ML gap may partly be an artifact of insufficiently cleaned input data in the Week 3 ML pipeline.

### Per-Model Results (100-Sample Test, Best Manual Prompt)

| Model | Avg MSE (LLM) | Avg MSE (ML) | R² Range (LLM) | R² Range (ML) |
|---|---|---|---|---|
| gemma-3-12b-it | 0.000604 | 0.005718 | 0.874–0.986 | 0.485–0.631 |
| phi-3.5-mini-instruct | 0.000546 | 0.005718 | 0.889–0.986 | 0.485–0.631 |
| phi-mini-moe-instruct | 0.000531 | 0.005718 | 0.895–0.983 | 0.485–0.631 |

Manual prompts outperformed auto-designed (agentic) prompts on 11/12 model×horizon combinations. The gap widens at longer horizons — phi-3.5 auto-designed R² collapses to 0.26 at 1440 min vs 0.89 for manual. Full per-horizon breakdowns are in `Results/llm_benchmark_performance.csv` and `Results/agentic_benchmark_performance.csv`.

### Key Takeaways

- **LLMs outperform XGBoost on every horizon**, with the largest advantage at short horizons (25–35× lower MSE at 15 min).
- **phi-mini-moe-instruct is the most consistent model** (best 1440-min R²=0.895, lowest cross-horizon variance).
- **Manual prompt engineering beats agentic auto-design**, especially at longer horizons.
- **≥5 few-shot examples are essential** for prediction accuracy; parse reliability is not guaranteed at high shot counts (compact_20 dropped to 60%).

---

## Repeatability Steps

1. **Clone the repository**: Clone the repo into your working directory.
2. **Download Large Data files**: Download `FTES_cleaned_1sec_1min_resample.csv` from [processed data](https://pnnl.sharepoint.com/:f:/r/teams/ESSDAICompetition/Shared%20Documents/General/Team%2010%20Byte%20Me/processedData?csf=1&web=1&e=LcbazL) folder on SharePoint and put it in `week3_artifacts/`.  
3. **Run notebook**: Execute `Notebooks/fewshot_llm_ml_benchmark.ipynb` end-to-end. It loads data, generates few-shot examples, runs small-sample validation (135 queries), selects best prompt per model, runs full benchmark (300 queries), executes the agentic loop (10 rounds × 3 models), and generates the final comparison.
4. **Outputs**: All CSVs written to `Results/`, prompts to `Prompts/`. Results include `raw_response` and `prompt_file` columns for auditing without re-querying.