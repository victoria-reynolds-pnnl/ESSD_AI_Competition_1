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

The most common failure mode is **invalid JSON formatting**: models wrap output in markdown code fences (`` ```json ... ``` ``) or add explanatory text, breaking the parser. A second major failure is **zero-shot hallucination** — gemma on verbose_zeroshot produced MSE=33.4 vs 0.0001 with 10-shot (300,000× worse) despite 100% parse success, returning syntactically valid JSON with meaningless round-number values. Third, **sensitivity to prompt length**: phi-3.5's parse rate dropped from 100% to 40% simply by increasing from 10 to 20 compact examples, as the model switched from JSON to natural-language output (format drift). All models also report confidence of 0.85–1.0 regardless of accuracy (uncalibrated).

### Likely Causes

- **`invalid_json`**: Instruction-tuning biases models toward markdown code fences despite "respond ONLY with JSON" instructions.
- **Zero-shot hallucination**: No few-shot examples to anchor the numeric scale; models default to heuristic round numbers (5.0, 1.0).
- **Format drift (compact_20)**: Long prompts approach the context window limit, causing the model to lose track of formatting constraints.
- **Confidence miscalibration**: Few-shot examples all use `confidence: 1.0`, teaching the model to always report high confidence.

### Parse Success Rates (5-sample validation)

Three version/model combos had sub-100% parse success:

| Version | Model | Parse Rate | Issue |
|---|---|---|---|
| compact_20 | phi-3.5-mini-instruct | **40%** | Format drift — switched to natural language |
| verbose_zeroshot | phi-mini-moe-instruct | **80%** | Missing target fields |
| compact_zeroshot | phi-mini-moe-instruct | **80%** | Missing target fields |

Zero-shot MSE can be **300,000×** worse than few-shot even when parsing succeeds — the models produce syntactically valid JSON with numerically meaningless predictions.

### Agentic Loop Failures

- Some rounds produced templates missing the `{features_json}` placeholder.
- gemma explored varied strategies across rounds but MSE was highly unstable (0.0014 to 34.3), with several rounds producing catastrophically bad predictions despite 100% parse success.
- Agentic prompts did not outperform manual prompts for gemma or phi-3.5; phi-mini-moe's auto-prompt was competitive but still did not beat its manual prompt.

---

## LLM vs ML vs Persistence Baseline Comparison

All three LLMs, the XGBoost ML baseline, and auto-designed (agentic) LLM prompts were compared against a **persistence baseline** (predict that the future temperature delta equals the current value, i.e., y_t+h = y_t). **Neither the LLMs nor the ML model consistently beat the persistence baseline.** Persistence won 7 of 12 model×horizon matchups on MSE. Manual LLM prompts beat persistence only at short horizons for gemma-3-12b-it (+15m, +60m) and across three horizons for phi-mini-moe-instruct. All methods underperformed persistence at +1440 min. Auto-designed prompts catastrophically failed for gemma (MSE >1, R² < −72), while phi-3.5's auto-prompt was poor but not catastrophic, and phi-mini-moe's auto-prompt was competitive with its manual prompt.

### Full Results (100-Sample Test)

| Model | Horizon | MSE Manual | MSE Auto | MSE ML | MSE Persist | Best |
|---|---|---|---|---|---|---|
| gemma-3-12b-it | +15m | **0.000255** | 1.025 | 0.00151 | 0.000338 | Manual LLM |
| gemma-3-12b-it | +60m | **0.000194** | 1.026 | 0.00111 | 0.000281 | Manual LLM |
| gemma-3-12b-it | +240m | 0.000443 | 1.026 | 0.00157 | **0.000313** | Persistence |
| gemma-3-12b-it | +1440m | 0.001345 | 1.034 | 0.00111 | **0.001090** | Persistence |
| phi-3.5-mini-instruct | +15m | 0.000390 | 0.00269 | 0.00151 | **0.000338** | Persistence |
| phi-3.5-mini-instruct | +60m | 0.000346 | 0.00331 | 0.00111 | **0.000281** | Persistence |
| phi-3.5-mini-instruct | +240m | 0.000611 | 0.00532 | 0.00157 | **0.000313** | Persistence |
| phi-3.5-mini-instruct | +1440m | 0.001527 | 0.01103 | 0.00111 | **0.001090** | Persistence |
| phi-mini-moe-instruct | +15m | **0.000250** | 0.000282 | 0.00151 | 0.000338 | Manual LLM |
| phi-mini-moe-instruct | +60m | **0.000225** | 0.000252 | 0.00111 | 0.000281 | Manual LLM |
| phi-mini-moe-instruct | +240m | **0.000297** | 0.000614 | 0.00157 | 0.000313 | Manual LLM |
| phi-mini-moe-instruct | +1440m | 0.001471 | 0.001695 | 0.00111 | **0.001090** | Persistence |

Full per-horizon breakdowns are in `Results/llm_benchmark_performance.csv` and `Results/agentic_benchmark_performance.csv`.

### Key Takeaways

- **Neither LLMs nor XGBoost consistently beat the persistence baseline.** Persistence wins 7/12 matchups and is the best method at every +1440 min comparison.
- **Auto-designed prompts catastrophically failed for gemma** (MSE >1, R² as low as −78). phi-3.5's auto-prompt was poor but functional (MSE 0.003–0.011, R² 0.29–0.84). phi-mini-moe's auto-prompt was competitive (MSE 0.000282–0.001695, R² 0.87–0.98).
- **Manual LLMs outperform XGBoost on every horizon except 1440m** (up to 6× lower MSE at 15 min), but this advantage narrows or disappears against persistence.
- **phi-mini-moe-instruct is the most consistent LLM** — the only model to beat persistence at three horizons (+15m, +60m, +240m) with manual prompts, and its auto-prompt also remained competitive.
- **gemma-3-12b-it beats persistence only at short horizons** (+15m, +60m) with manual prompts, with the best overall R² of 0.986 at +60m.
- **≥5 few-shot examples are essential** for prediction accuracy; parse reliability drops at high shot counts (compact_20 fell to 40% for phi-3.5).
- **The persistence baseline's strong performance** likely reflects high temporal autocorrelation in geothermal signals — temperature changes slowly, making "predict no change" hard to beat.

## Repeatability Steps

1. **Clone the repository**: Clone the repo into your working directory.
2. **Download Large Data files**: Download `FTES_cleaned_1sec_1min_resample.csv` from [processed data](https://pnnl.sharepoint.com/:f:/r/teams/ESSDAICompetition/Shared%20Documents/General/Team%2010%20Byte%20Me/processedData?csf=1&web=1&e=LcbazL) folder on SharePoint and put it in `week3_artifacts/`.  
3. **Run notebook**: Execute `Notebooks/fewshot_llm_ml_benchmark.ipynb` end-to-end. It loads data, generates few-shot examples, runs small-sample validation (135 queries), selects best prompt per model, runs full benchmark (300 queries), executes the agentic loop (10 rounds × 3 models), and generates the final comparison.
4. **Outputs**: All CSVs written to `Results/`, prompts to `Prompts/`. Results include `raw_response` and `prompt_file` columns for auditing without re-querying.