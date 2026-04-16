# Prompting approach

Give exact example to LLM of expected output
Track iterations for every prompt


# Error analysis (2-3 sentences and include failure examples)



# Repeatability steps
To reproduce results using JupyterHub:

1. Upload data files (`data_cleaned_split.csv` and `arima_arimax.csv`) located in `evergreen_analyzers/Notebooks/week3_artifacts` to jupyterhub root
2. Run all cells in order starting at Step 1
3. Change `ITERATION_NUMBER` in Step 2 to track different runs
4. Results will be saved to `ROOT / Results` and Prompts will be saved to `ROOT / Prompts`