[README_GNN_LLM_PIPELINE.md](https://github.com/user-attachments/files/26921174/README_GNN_LLM_PIPELINE.md)
# Week 4 GNN-to-LLM Pipeline

## Overview
This pipeline connects Week 3 GNN outputs to Week 4 LLM evaluation:
1. Extract network connectivity insights from GNN edge weights
2. Generate narrative text summaries
3. Query LLMs to interpret the narratives
4. Compare LLM classifications with Week 3 ML predictions

---

## Files Created

### **gnn_insights_to_text.py**
**Purpose:** Extract edge weights and generate input data for LLM notebooks

**What it does:**
- Loads `edge_weights_over_time.npy` (GNN output)
- Loads `ftes_scaled_for_GNN.csv` (sensor data)
- Extracts connectivity statistics per time window
- Generates mixed technical+domain narrative text (~1500 records)
- Assigns risk labels (low/moderate/high_risk) based on connectivity patterns
- Outputs: `gnn_generated_data/gnn_insights_for_llm.csv`

**Run before the notebooks:**
```bash
cd Week4
python gnn_insights_to_text.py
```

**Output structure:**
```
sample_id, text, label, confidence, pattern, edge_weight_mean, edge_weight_std, active_edges
1, "Network snapshot 1: Edge weight analysis...", high_risk, 0.92, high_stability, 0.45, 0.12, 8
...
```

---

### **week4_gnn_llm_benchmark.ipynb**
**Purpose:** Compare LLM performance vs Week 3 ML baseline on GNN narratives

**Workflow:**
1. Load GNN insights and Week 3 ML predictions
2. Define strict JSON classification prompt
3. Smoke test: Query one example on each LLM endpoint
4. Full benchmark: Classify all narratives on all models
5. Evaluate: Compare LLM accuracy with ML baseline
6. Export: Save raw and clean CSV results

**Output files:**
- `Results/gnn_llm_results_raw_1.csv` — Full responses with raw LLM output
- `Results/gnn_llm_results_clean_1.csv` — Parsed predictions only

**Key metrics computed:**
- LLM accuracy per model
- Parse success rate (valid JSON %)
- Invalid output rate (formatting errors)
- Side-by-side comparison with ML

---

### **week4_gnn_prompt_iteration.ipynb**
**Purpose:** Test multiple prompt versions and document failure modes

**Workflow:**
1. Load GNN narratives
2. Define 3 prompt versions:
   - `v1_minimal` — Strict, minimal instructions
   - `v2_contextual` — More guidance and context
   - `v3_few_shot` — Examples-based prompting
3. Test each version on small subset (8 examples)
4. Compare parse reliability and accuracy
5. Analyze failures (parse errors vs wrong predictions)
6. Export best prompts and error analysis

**Output files:**
- `Prompts/gnn_prompt_v1_minimal.txt` — Best template per version
- `Prompts/gnn_few_shot_examples.json` — Few-shot examples
- `Results/gnn_prompt_iteration_results.csv` — Detailed results

**Error analysis includes:**
- Breakdown of parse error types
- Sample misclassifications
- Recommendations for report

---

## Workflow: Step by Step

### **Step 1: Generate GNN Insights**
```bash
cd d:\AI_Week3_and_4\Week4
python gnn_insights_to_text.py
```
Expected output:
```
gnn_generated_data/gnn_insights_for_llm.csv (up to 1500 records)
```

### **Step 2: Run Benchmark (on Jupyter Hub)**
Open `week4_gnn_llm_benchmark.ipynb` and run cells in order:
- Cell 1-3: Load imports and config
- Cell 4: Load data (will create gnn_insights_for_llm.csv)
- Cell 5-6: Define prompt and parser
- Cell 7: **Smoke test** — validates LLM endpoints are working
- Cell 8: **Full benchmark** — this is slow (queries all models on all records)
- Cell 9-10: Evaluate and export results

⏱️ **Time estimate:** 10 min for smoke test, 1-2 hours for full benchmark (depends on number of records and model response times)

### **Step 3: Run Prompt Iteration (on Jupyter Hub)**
Open `week4_gnn_prompt_iteration.ipynb` and run cells in order:
- Cell 1-3: Load imports and config
- Cell 4: Load data
- Cell 5-6: Define 3 prompt versions
- Cell 7: **Test prompts on subset** — small, fast test (8 examples × 3 versions × 3 models)
- Cell 8-9: Analyze failures and best version
- Cell 10: Export prompts and report

⏱️ **Time estimate:** 5-10 minutes

---

## Important Notes

### Environment Requirements
- **Python packages:**
  ```
  numpy, pandas, openai
  ```
- **LLM endpoints:** Configure MODEL_ENDPOINTS in notebooks to match your Jupyter Hub setup
  - Default: localhost:8000, 8001, 8002 (Phi models + Gemma)
  - Update `host` and `port` as needed

### Data Requirements
- ✅ `Week3/edge_weights_over_time.npy` — from Week 3 GNN training
- ✅ `Week3/ftes_scaled_for_GNN.csv` — sensor data
- ⚠️ `Week4/week3_artifacts/ml_predictions.csv` — Week 3 ML baseline (optional for comparison)

If any file is missing, `gnn_insights_to_text.py` will raise an error.

### Running on Jupyter Hub
1. Upload `gnn_insights_to_text.py` to your Jupyter environment
2. Run it first to generate `gnn_generated_data/gnn_insights_for_llm.csv`
3. Then open the `.ipynb` notebooks and execute cells
4. For full benchmark: Start the smoke test first to verify endpoints are reachable

---

## Output Structure

After running both notebooks, you'll have:

```
Results/
├── gnn_llm_results_raw_1.csv          # Full benchmark (all columns)
├── gnn_llm_results_clean_1.csv        # Clean benchmark (key columns only)
├── gnn_phi-3.5-mini-instruct_raw_1.csv   # Per-model raw
├── gnn_phi-3.5-mini-instruct_clean_1.csv # Per-model clean
├── ...more per-model files...
└── gnn_prompt_iteration_results.csv   # Prompt comparison (small subset)

Prompts/
├── gnn_prompt_v1_minimal.txt         # Best template (minimal version)
├── gnn_prompt_v2_contextual.txt      # Best template (contextual version)
├── gnn_prompt_v3_few_shot.txt        # Best template (few-shot version)
└── gnn_few_shot_examples.json        # Few-shot examples
```

---

## Example Analysis

After running the pipeline, you can answer:

1. **Benchmark:** "Which LLM performs best at interpreting GNN insights?" 
   - See `Results/gnn_llm_results_clean_1.csv` → accuracy per model

2. **Comparison:** "How do LLMs compare to Week 3 ML on the same task?"
   - See summary table in benchmark notebook output

3. **Reliability:** "What % of LLM outputs parse as valid JSON?"
   - See `parse_success_rate` in prompt iteration results

4. **Failure modes:** "Why do LLMs fail?"
   - See error breakdown in prompt iteration notebook (Step 5)

---

## Customization

### Modify edge weight → risk classification logic
Edit `assess_risk_level_and_confidence()` in `gnn_insights_to_text.py`:
```python
if mean_w > 0.5:
    risk = 'high_risk'  # Adjust threshold here
elif mean_w > 0.25:
    risk = 'moderate_risk'
else:
    risk = 'low_risk'
```

### Add more prompt versions
In `week4_gnn_prompt_iteration.ipynb`, add a new version to `build_prompt()`:
```python
elif version == 'v4_custom':
    return (
        'Your custom prompt here...'
    )
```

### Change sample size
In `gnn_insights_to_text.py`:
```python
MAX_RECORDS = 500  # Instead of 1500
```

In `week4_gnn_prompt_iteration.ipynb`:
```python
subset = df.sample(n=min(20, len(df)), random_state=42)  # More examples to test
```

---

## Troubleshooting

**Q: `FileNotFoundError: edge_weights_over_time.npy not found`**
- A: Run `gnn_insights_to_text.py` first to generate the data

**Q: `ConnectionError: Cannot connect to localhost:8000`**
- A: Verify LLM endpoints are running on Jupyter Hub; update `MODEL_ENDPOINTS` in notebooks

**Q: Notebooks running very slowly**
- A: Normal if querying many models on many records. Consider reducing `MAX_RECORDS` or testing on a subset first.

**Q: Parse success rate is 0% (all JSON invalid)**
- A: Try the contextual or few-shot prompt versions in `week4_gnn_prompt_iteration.ipynb` before full benchmark

---

## For Week 4 Submission

Required deliverables:
1. ✅ **Approach**: Describe how you used LLMs to interpret GNN insights
   - Use text from benchmark notebook (Step 9) as template
2. ✅ **Error Analysis**: Document LLM failure modes
   - Use text from prompt iteration notebook (Step 7) as template
3. ✅ **Predictions CSV**: `Results/gnn_llm_results_clean_1.csv`
4. ✅ **Prompts**: `Prompts/gnn_prompt_*.txt` (your best versions)

---

Questions? Check the docstrings in `gnn_insights_to_text.py` or the markdown cells in the notebooks for detailed explanations of each step.
