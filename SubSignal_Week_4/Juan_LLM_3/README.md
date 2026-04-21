# Week 4 LLM Connectivity Benchmark

This package benchmarks a local LLM against the Week 3 GNN connectivity workflow in `ESSD_AI_Competition_1/SubSignal_Week_3`. The LLM sees the same 12-hour pressure windows and predicts the strongest connected node pairs. Benchmark labels are derived from the saved Week 3 GNN edge weights, so this is an agreement study against the GNN's inferred connectivity rather than physical ground truth.

## Approach

The benchmark uses the Week 3 scaled FTES dataset and the saved `edge_weights_over_time.npy` tensor. For each saved GNN window, the notebook:

- builds a compact text summary of 12 hours of pressure behavior across the 10 pressure nodes
- includes `tc_injecting`, `tc_producing`, and `delta_P_delta_Q` context
- converts the Week 3 directed edge weights into undirected pair scores using mean absolute weight
- keeps the top 3 undirected pairs as the target connectivity label
- prompts `gemma3:12b-it-qat` through a local Ollama OpenAI-compatible endpoint

## Prompting And Validation Strategy

The final prompt is stored in [Prompts/prompt_template.txt](/Users/bara407/projects/GenAICompetition/ESSD_AI_Competition_1/SubSignal_Week_4_LLM_Gemma/team_folder/Prompts/prompt_template.txt). It instructs the model to infer the strongest connected undirected node pairs from the 12-hour window summary and return strict JSON only. Pair ids are constrained to an allowed list, and the parser rejects malformed JSON, invalid pair ids, duplicates, and wrong list lengths.

Few-shot prompting is supported through [Prompts/few_shot_samples.json](/Users/bara407/projects/GenAICompetition/ESSD_AI_Competition_1/SubSignal_Week_4_LLM_Gemma/team_folder/Prompts/few_shot_samples.json). The notebook uses only prompt-development examples and keeps the benchmark test split separate with a 12-window purge gap to reduce overlap leakage from sliding windows.

## Evaluation

Primary metric:

- `top3_overlap_rate`

Secondary metrics:

- `jaccard_at_3`
- `top1_match_rate`
- `parse_success_rate`

The notebook also reports performance by operating regime (`idle`, `injecting`, `producing`, `mixed`) so you can compare where the LLM tracks the GNN and where it drifts.

## Error Analysis

The main failure mode is likely conflating common-mode movement with true pairwise coupling. When many wells move together during operational shifts, the LLM may over-select broad field-response pairs instead of the same concentrated top pairs the GNN emphasizes.

Another likely failure mode is windows with weak separation among the top ranked pairs. In those windows the GNN's top 3 can be numerically close, so small reasoning differences in the LLM can change the predicted set even when the qualitative story is similar.

## Repeatability

1. Start Ollama and make sure `gemma3:12b-it-qat` is installed.
2. Open [Notebooks/llm_benchmark.ipynb](/Users/bara407/projects/GenAICompetition/ESSD_AI_Competition_1/SubSignal_Week_4_LLM_Gemma/team_folder/Notebooks/llm_benchmark.ipynb).
3. Run the cells in order using the `SubSignalAiCompetition/.venv` Python environment.
4. Confirm that the raw and clean CSVs in `Results/` are rewritten.

## Notes

- The Week 3 edge tensor has shape `(2, 2456, 90)`, so the benchmark uses the exact `2456` saved windows represented in the tensor rather than all possible 12-hour slices from the CSV.
- A live smoke test against local `gemma3:12b-it-qat` succeeded with parseable JSON output. The checked-in CSVs are still schema-complete placeholders until the full benchmark cell is run across the test split.
