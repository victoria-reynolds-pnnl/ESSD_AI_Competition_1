# Zero Shot Hopefuls - Compound Hazard Forecasting

## Week 4: LLM Benchmarking vs ML Model

### Approach Report

We benchmarked three locally-hosted LLMs (Phi-3.5-mini-instruct, Phi-mini-MoE-instruct, gemma-3-12b-it) on the same 1-week-ahead compound heatwave+drought prediction task from Week 3. The core challenge was adapting a tabular numerical prediction task — 26 climate features per grid cell — for small language models (7-12B parameters) that lack native numerical reasoning capabilities. We designed prompts that present each feature with its human-readable name, units, and domain context (e.g., "Palmer Drought Severity Index (pdsi): -2.45, range -5=extreme drought to +5=extreme wet"), grouped by category (temperature, drought, fire, meteorological, vegetation, seasonal). Output was constrained to strict JSON format: `{"prediction": 0 or 1, "confidence": 0.0-1.0}`. We tested four prompt versions: v1 (zero-shot, all 26 features), v2 (few-shot with 5 labeled training examples), v3 (zero-shot, top 8 SHAP-important features only), and v3_few (few-shot, top 8 features). The best-performing version per model was selected based on parse reliability and accuracy on a 20-row iteration subset, then run against the full 1,000-row stratified test sample. Few-shot examples were drawn exclusively from the training set (2001-2015) to prevent data leakage. LLM responses were parsed defensively: a regex extracts JSON objects from potentially markdown-wrapped or verbose responses, then validates prediction values and confidence ranges.

### Error Analysis

All three models achieved a 100% parse success rate — every response was valid JSON with the correct schema, so output formatting was not a failure mode. The primary errors were prediction errors, particularly around the class-imbalanced positive class. Phi-3.5-mini-instruct was the most conservative predictor (29 positive predictions vs 27 actual), producing 14 false negatives (missed 52% of compound events) and 16 false positives. In contrast, Phi-mini-MoE-instruct and gemma-3-12b-it over-predicted positive events (142 and 109 respectively), catching 70% of real events but at the cost of many false alarms (123 and 90 false positives). Confidence calibration was problematic across all models: gemma-3-12b-it reported near-identical confidence (mean 0.92, range 0.75-0.95) regardless of correctness, making its confidence scores useless for ranking. Phi-mini-MoE-instruct showed an inverted calibration pattern — higher confidence on incorrect predictions (0.81) than correct ones (0.02). Phi-3.5-mini-instruct similarly reported higher confidence when wrong (0.74) than when right (0.31), suggesting these small models cannot meaningfully self-assess prediction quality on numerical tabular data.

### Comparison Summary

| Model                        | F1         | AUC-ROC    | Precision  | Recall     | Brier Score | Parse Rate |
| ---------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- | ---------- |
| LightGBM (Week 3)            | **0.6441** | **0.9870** | **0.5938** | 0.7037     | **0.0200**  | —          |
| Logistic Regression (Week 3) | 0.5283     | 0.9812     | 0.5385     | 0.5185     | 0.0415      | —          |
| Phi-3.5-mini-instruct (LLM)  | 0.4643     | 0.8323     | 0.4483     | 0.4815     | 0.1494      | 100%       |
| Phi-mini-MoE-instruct (LLM)  | 0.2249     | 0.7935     | 0.1338     | **0.7037** | 0.1005      | 100%       |
| gemma-3-12b-it (LLM)         | 0.2794     | 0.3879     | 0.1743     | **0.7037** | 0.8313      | 100%       |

LightGBM decisively outperforms all three LLMs on this task. The best-performing LLM (Phi-3.5-mini-instruct) achieved F1=0.4643 versus LightGBM's 0.6441 — a 28% relative gap. The performance difference is most visible in discriminative ability: LightGBM's AUC of 0.987 means it nearly perfectly ranks compound-event weeks above non-event weeks, while even the best LLM (Phi-3.5-mini, AUC=0.832) struggles to distinguish between them. Phi-mini-MoE and gemma-3-12b-it achieved high recall (0.70, matching LightGBM) but at the cost of precision collapsing to 0.13-0.17, meaning ~85% of their positive predictions were false alarms. Gemma's AUC of 0.39 (below random chance at 0.50) indicates its confidence scores are inversely correlated with actual risk — actively misleading. The Brier scores tell a similar story: LightGBM's well-calibrated probabilities (Brier=0.020) contrast sharply with gemma's 0.831 (near-maximum miscalibration). These results confirm that for structured numerical prediction tasks with clear decision boundaries, purpose-built ML models trained on domain data substantially outperform zero-shot LLM reasoning. The LLMs' strength lies in their ability to attempt the task with no training data at all — Phi-3.5-mini achieved a reasonable F1 of 0.46 purely from prompt-based reasoning about feature descriptions, which would be a useful starting point when labeled data is unavailable. However, once labeled data exists, ML models are the clear choice for operational compound hazard forecasting.

### Repeatability Steps

```bash
# 1. Prepare the test subset (run from repo root, requires Week 2/3 data):
source .venv/Scripts/activate
python zero-shot-hopefuls/Scripts/prepare_llm_subset.py

# 2. Run the benchmark notebook on JupyterHub:
#    - Log in to https://rcjh.pnl.gov/ (username: first_name.last_name)
#    - Activate the ai_comp conda environment
#    - Open zero-shot-hopefuls/Notebooks/llm_benchmark.ipynb
#    - Run all cells (Steps 1-4 work offline; Steps 5-10 require LLM endpoints)
```

**Role of AI:** Claude (Anthropic) was used to generate the benchmark notebook, data preparation script, prompt templates, and README sections. All AI-generated code is cited in comments. Claude also assisted with prompt design strategy, feature presentation approach for tabular data, and error analysis framework. All AI-generated code was reviewed by the team.

---

**Role of AI:** Claude (Anthropic) was used to generate the training, evaluation, and interpretability scripts, with each AI-generated section cited in code comments. Claude also assisted with split strategy design, feature selection rationale (leakage analysis), metric selection justification, and HITL review framing. All AI-generated code was reviewed by the team.

## File Structure

```
zero-shot-hopefuls/
  Scripts/
    prepare_llm_subset.py                      # Week 4: Sample test subset and generate ML baselines
  Notebooks/
    llm_benchmark.ipynb                        # Week 4: LLM benchmarking notebook (run on JupyterHub)
  Prompts/
    prompt_template.txt                        # Week 4: Final prompt template for LLM queries
    few_shot_samples.json                      # Week 4: 5 labeled examples from training set
  Results/
    {model_name}_results_clean_{iter}.csv      # Week 4: Per-model clean results
    {model_name}_results_raw_{iter}.csv        # Week 4: Per-model raw results with LLM responses
  requirements.txt                             # Python environment dependencies
  README.md                                    # This file
```
