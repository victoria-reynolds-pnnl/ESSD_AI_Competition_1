# Train/Test Split Strategy

We adopted a sequential chronological 80/10/10 split per monitoring location: within each site's daily time series, the first 80% of observations form the training set, the next 10% the validation set, and the final 10% the test set. This temporal ordering ensures the model is always forecasting forward in time, which is appropriate for the time-series data generating process. A year-based temporal split was considered but risked concentrating climatically distinct periods entirely within one partition, potentially biasing a model. To prevent data leakage, all engineered features (lags, rolling windows, seasonal decomposition) use only backward-looking calculations, duplicate records were removed during cleaning, and the sequential split guarantees no future data appears in training. Hyperparameters (p, q) were tuned via grid search over p ∈ {0, 1, 2} and q ∈ {0, 1, 2} with d = 1 fixed, selecting the order that minimized RMSE on the validation set; the final model was then refit on train + validation before evaluating on the held-out test set.

# Model Performance Summary

We evaluated both models at each of the 50 monitoring locations using RMSE, MAE, and R². RMSE and MAE are appropriate for this regression task because they are directly interpretable in the target units (cubic feet per second); RMSE penalizes large errors more heavily, which matters for extreme-flow events, while MAE reflects typical prediction offset. R² provides a scale-independent measure of variance explained, enabling comparison across stations with widely varying flow magnitudes. Across stations, ARIMA achieved a median validation RMSE of 702.74 (MAE 411.86, R² −0.11) and median test RMSE of 475.37 (MAE 308.35, R² −0.43), with zero stations reaching a positive test R². ARIMAX achieved a median validation RMSE of 742.63 (MAE 558.25, R² −0.19) and median test RMSE of 644.09 (MAE 481.64, R² −0.69), but produced positive test R² at 11 of 50 stations, with the best reaching 0.93 (USGS-12505450). Non-seasonal ARIMA systematically underfits the strong seasonal cycles in streamflow, producing near-linear forecasts, while ARIMAX's engineered seasonal features capture that structure at nearly half the stations on validation. We report medians rather than means because a few high-flow outlier stations heavily skew the averages. Overall, ARIMAX justifies the added feature-engineering complexity as a meaningful improvement over the ARIMA baseline, though negative median R² values across both models indicate that further approaches (e.g., SARIMA, ensemble methods) would be needed for robust site-general performance. These aggregate statistics are reproduced by running `evaluate.py`, which prints them after writing the per-station results CSV.


# HITL Review

- **Does the model behavior make sense given domain expectations?** Both ARIMA and ARIMAX are run independently per monitoring location (~50 after data cleaning). ARIMA provided a clear site-level baseline and seemed reasonable during Week 2 planning, but Week 3 testing showed it was not a strong fit for our flow-rate forecasts. The non-seasonal ARIMA model failed to capture the pronounced multi-year cycles, producing near-linear predictions that missed seasonal peaks and troughs, which is consistent with domain expectations — streamflow is inherently seasonal. ARIMAX with engineered seasonal features performed noticeably better, capturing the sinusoidal seasonal pattern, though amplitude and timing mismatches remain. This behavior aligns with what we would expect: seasonal features help but cannot fully account for interannual variability driven by snowpack, precipitation, and upstream regulation.

- **Any concerning spurious correlations?** Some locations appear to fail to achieve stationarity, which could be due to geographic variability (e.g., topography, external changes in water usage or inflow) that are not considered as part of the cleaned dataset (which only includes outflow). The rolling window features (7-, 14-, 21-day) are computed from the target variable itself, which risks learning autocorrelation artifacts rather than genuinely predictive patterns — this may explain why ARIMAX overfits at some stations (performing worse than ARIMA at 31/50 test stations despite having more features).

- **Any high-risk failure modes?** ARIMA underfit short-term variability (the "nugget"-like noise), limiting its ability to capture day-to-day dynamics. Dam-regulated stations (e.g., USGS-12472800, Columbia River below Priest Rapids Dam) produce extreme errors because dam releases create abrupt flow changes no historical pattern can predict. Stations with near-zero flow periods generate misleading MRE values. Given these limitations, we kept ARIMA as a baseline and shifted our Week 3 focus toward seasonal feature-related approaches such as ARIMAX.


# Failures and Limitations

The most prominent failure condition is systematic val-to-test degradation: R² dropped from validation to test at 46/50 stations for ARIMA and 39/50 for ARIMAX, consistent with the sequential split placing the most recent (and potentially non-stationary) data in the test partition. ARIMAX actually performed worse than ARIMA at 31 of 50 stations on test R², with several stations showing severe overfitting to the engineered seasonal features (e.g., USGS-12451000 dropped from ARIMA R² −0.12 to ARIMAX R² −4.81). Dam-regulated stations such as USGS-12472800 (Columbia River below Priest Rapids Dam, drainage area 96,000 sq mi) produced extreme RMSE outliers because dam operations introduce abrupt flow changes that neither model can anticipate from historical patterns alone. A related edge case is stations with near-zero flow periods (e.g., USGS-12447383, USGS-12465400), where the mean-relative-error metric exceeded 10^9 due to division by near-zero actuals — a bias in our MRE calculation rather than a true model failure. Neither model captures interannual climate variability (drought years, El Nino shifts) since the feature set includes only intra-annual seasonal signals and short rolling windows. To address these limitations in future work, we would incorporate explicit seasonal differencing (SARIMA), add exogenous climate indices (e.g., ENSO, PDO) and dam-release schedules as features, use site-clustered cross-validation instead of a single sequential split, and replace MRE with symmetric MAPE or log-scale metrics for low-flow stations.

# Reproducibility Steps

1. **Prerequisites**: Python 3.10+ installed.
2. **Install dependencies** from the `Evergreen_Analyzers/` directory:
   ```
   pip install -r Requirements.txt
   ```
3. **Run the full pipeline** from the `Scripts/` directory:
   ```
   cd Scripts
   python train_evaluate_interpret.py
   ```
   This single script performs all steps end-to-end and produces all required outputs:
   - `Data/data_cleaned_split.csv` — preprocessed data with split labels
   - `Data/arima_arimax_results.csv` — validation and test metrics for all locations
   - `Models/*.pkl` — fitted ARIMA and ARIMAX models
   - `Visualizations/arima_arimax_{id}.png` — per-location forecast plots

   **Alternatively**, run the three modular scripts in sequence for step-by-step control:
   ```
   python train.py          # preprocess, split, grid search, save models
   python evaluate.py       # load models, compute val/test metrics, write results CSV
   python interpretability.py  # generate forecast visualizations
   ```
   Each script reads the outputs of the previous step, so they must be run in order.

# AI Use and Disclosure

Training and evaluation scripts were developed with assistance from the PNNL AI Incubator chat (linked in script headers) and GitHub Copilot. The `evaluate.py` aggregate summary function and README writeups for Model Performance Summary and Failures and Limitations were drafted with GitHub Copilot assistance. All AI-assisted code sections are cited in comments at the top of each script file.