# Prompting approach

To create a fair comparison between our Week 3 ARIMAX model and an LLM-based approach, we selected the forecasting task as our benchmarking target because it produces numerical predictions of daily flow rate in cubic feet per second that can be directly evaluated against the same held-out test set using RMSE (Week 3 metrics). Rather than forecasting monthly means, we structured the prompt to match the daily granularity of our ARIMAX output: phi-3.5-mini-instruct, phi-mini-moe-instruct, and gemma-3-12b-it were given historical daily flow rate data from two monitoring location as a comma-separated list of values, then instructed to predict the next 960 days of daily flow rates (corresponding to our test data) while paying attention to seasonality and other patterns in the provided data. The prompt enforced strict output formatting — a comma-separated list of floating-point numbers with no extra text, brackets, or other characters — so that the LLM's predictions could be programmatically parsed and aligned date-for-date against the ARIMAX test set for a direct RMSE comparison. We reduced the input to a single site's one-year window (fewer than 1,500 values) to stay within the context window, a necessary constraint given our Week 1 finding that the full 1.2M-point dataset far exceeded feasible LLM processing capacity

# LLM output and validation approach

Forcing an LLM to produce dense numerical time series required significant workarounds that themselves revealed inherent limitations. The full daily time series per site exceeded the models’ context window, so we could only feed a truncated portion of each site's training history which is a fundamental disadvantage compared to ARIMAX. Even when the LLM accepted the input, its output length constraints meant it returned predictions for less than half of the requested future dates, producing a sparse forecast with substantial gaps. To validate outputs, we programmatically parsed the results by splitting at commas and casting the entries to floating point numbers. As discussed in our previous deliverables, LLMs are not designed to crunch dense numerical time series, however our results provide a concrete, apples-to-apples benchmark against our ML pipeline.

# Error analysis (2-3 sentences and include failure examples)
The most common failure was incomplete output: the LLM consistently stopped generating predictions well before covering the full test window, leaving the majority of forecast dates as nulls that could not be scored. Missing values were as follows, indicating that gemma-3-12b-it provided the most complete results, but still missed the majority at both sites:
phi-3.5-mini-instruct: 774 (80.62%) missing on site 1, 813 (84.69%) missing on site 2.
phi-mini-moe-instruct: 767 (79.90%) missing on site 1, 822 (85.62%) missing on site 2.
gemma-3-12b-it: 594 (61.88%) missing on site 1, 594 (61.88%) missing on site 2.


# Repeatability steps
To reproduce results using JupyterHub:

1. Upload data files (`data_cleaned_split.csv` and `arima_arimax.csv`) located in `evergreen_analyzers/Notebooks/week3_artifacts` to jupyterhub root
2. Run all cells in order starting at Step 1
3. Change `ITERATION_NUMBER` in Step 2 to track different runs
4. Results will be saved to `ROOT / Results`