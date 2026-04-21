[README.md](https://github.com/user-attachments/files/26921044/README.md)
# SubSignal Week 4 Deliverables

## Contents
- `Notebooks/llm_benchmark.ipynb`
- `Prompts/prompt_template.txt`
- `Results/microsoft_phi_3_5_mini_instruct_results_clean_1.py`

## Approach (Prompting + Validation)
The notebook benchmarks one LLM (`microsoft/Phi-3.5-mini-instruct`) against a ridge-regression baseline on the Week 3 pressure-forecasting framing (12-hour window -> next-hour pressure at 10 nodes). The prompt is intentionally simple and constrained to reduce formatting drift: it asks for exactly 10 comma-separated numbers in a fixed node order. Validation is parser-first: responses are converted into a numeric vector by regex extraction; rows that fail parsing are marked invalid (`parse_ok=False`) and excluded from metric aggregation, while still retained in row-level outputs for auditability.

## Error Analysis
Most failures were output-format failures rather than numeric instability: e.g., model responses that included explanatory text before values, or fewer than 10 numeric values. Another common failure mode was partial compliance where values were present but mixed with extra tokens (headers or prose), which lowered parse reliability. These cases are preserved in the `raw_output` and `parse_ok` columns so reviewers can inspect exactly why a row was rejected.

## Repeatability Steps
1. Open `Notebooks/llm_benchmark.ipynb` in the `ai_comp` environment.
2. Ensure `ftes_scaled_for_GNN.csv` is in the working directory or in `Week 3 Deliverables/Data/`.
3. Run cells top-to-bottom with fixed seed (`GLOBAL_SEED = 42`).
4. Confirm benchmark completion message and parse success rate.
5. Confirm metrics are written to the run folder under `Results/week4_llm_simple/<timestamp>/`.
6. Run `Results/microsoft_phi_3_5_mini_instruct_results_clean_1.py` and point it at the generated `row_level_results.csv` to produce the clean submission CSV.

## Result Files Required by Rubric
The cleaned per-LLM file includes:
- prompt text (`prompt`)
- raw model output (`raw_output`)
- structured parsed prediction (`parsed_prediction`)
- parse status (`parse_ok`)
- latency and per-node true/ML/LLM values
