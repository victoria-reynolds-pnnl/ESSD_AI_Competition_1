# Sub‑Terra Tensor Tinkerers - Week 3 Submission

**Project Title:** Signal‑to‑Storage: Early Warning Forecasts for FTES Charging Operations

**Team:** Scott Unger, Julian Chesnutt, Theresa Pham, Xi Tan, Ashton Kirol, Mike Parker

## Test/Training Split Strategy
- **Dataset(s):**
  - Split-indexed: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_split_indexed.csv
  - Train only: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_train.csv
  - Validation only: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_val.csv
  - Test only: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Data/FTES_1hour_test.csv
- **Split method:** Time-based, chronological 
- **Split ratios:** 
  - 70% Train (1191 rows, 2024-12-13 20:00:00 -> 2025-02-01 10:00:00)
  - 15% Validation (255 rows, 2025-02-01 11:00:00 -> 2025-02-12 01:00:00)
  - 15% Test (256 rows, 2025-02-12 02:00:00 -> 2025-02-22 17:00:00)
- **Leakage prevention:** 
  - Aligned validation and test sets to the training feature schema
  - Missing values in the training set were imputed with medians calculated from the training set only (cleaning script from Week 2 was modified as well)
- **Rationale:** Since we are working with time-series data, randomization during the split could lead to data leakage from future timestamps. We simply kept the dataset in its original, chronological order and split into train/validation/test sequentially. 

## Training Strategy
- **Model**: XGBoost - Gradient boosted decision trees. Supports high-dimensional data.
- **Tuning Strategy**: Walk-forward validation tuning to minimize loss 
- **Feature Count**: 177
  - TC, TN, TL, and overall well measurements at the current timestep
- **Target Count**: 18
  - Change in value in the next hour for Injection pressure, TC pressure, and TN/TL flow and temperature at varying depths

## Model Performance Summary
- **Models/versions:**  
 - *xgb_original* was the result of our initial training approach, in which the model was trained directly on target values.
 - *xgb_delta* is the final version of the model, where the model learned the **change** in target values rather than target values themselves. Evaluation and interpretability predictions were calculated in later steps by adding the previous timestep's value with the predicted change.
- **Comparison to baseline(s):** The baseline was simply a persistence approach, where every target prediction y(t+1) would be carried over from the target value at time t, or y(t). Since the goal was to forecast in hour-level steps, the baseline proved difficult to surpass with a complex model, since values did not drastically change between timestamps.
- **Evaluation datasets:** https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/tree/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Models/xgb_delta/eval_output
  - *xgb_test_metrics* compares the baseline metrics to model performance metrics.
  - *xgb_test_predictions* provides the actual, predicted, and baseline value for the test set
- **Metrics:**
  - **Primary metric(s):** Mean Absolute Error (MAE) - robust, average error
  - **Secondary metric(s):** Root Mean Squared Error (RMSE) - more sensitive to outliers, penalizes heavily on larger errors
  - By balancing both MAE and RMSE, the model can learn nuanced patterns without overfitting to the training data.
- **Model and Parameter calibration:** https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/tree/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Models/xgb_delta/train_output
  - Since there were a total of 18 targets, the training script loops through each target individually to fine-tune parameters according to that target. 
  - Parameters: Many of the parameters below were calibated to minimize overfitting.
    - *n_estimators* - Number of decision trees
    - *max_depth* - Threshold for tree deepness/complexity
    - *learning_rate* - Shrink step size when updating weights
    - *colsample_bytree* - Per-tree subsample ratio of columns
    - *subsample* - Ratio of subsample data
    - *min_child_weight* - Limit splits on nodes with small weights
    - *reg_lambda* - L2 regulariation
    - *reg_alpha* - L2 regulatization
- **Overall results:**  The original XGBoost model that directly predicts target values at the next timestep was performing poorly and evaluation metrics were well under the baseline. Baseline RMSE and MAE remained consistently low, so calculated MAE and RMSE percent improvement from the baseline to the orignal model were large negative percentages for all targets. However, when inspecting the time series visualizations, for example in [Visualizations/xgb_original/timeseries_Injection_Pressure.png](https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_3/SubTerraTensorTinkerers/Visualizations/xgb_original/timeseries_Injection_Pressure.png), the model seemed to be able to follow the general trends of the true data, but was overshooting predictions (large peaks and dips in the time series, despite true values having minor fluctuations between hourly time steps). This observation was a strong sign of overfitting rather than random guessing. To correct for this, the approach was modified for xgb_delta, where the model is trained on the *change* between the next hour value and the current value, to capture the more subtle behavior in the data. As a result, both error metrics were reduced significantly for all targets. However, based on the the percent change in error from the baseline to the xgb_delta model, only 4 out of the 14 valid target values outperformed the baseline.

## Failures and Limitations
- Since XGBoost is known for supporting high-dimensional data, feature selection was kept to a minimum and the same set of 177 features was used to train each individual target. Involvement of domain expertise on the subject, along with target-specific feature analysis prior to model training, could help reduce the noise that was introduced with the current approach, especially when considering the small dataset size and short temporal coverage compared to the number of features.
- The persistence baseline was difficult to surpass at a 1-hour forecast horizon because most sensor readings change subtly between consecutive hours — only 4 of the 14 evaluable targets (TN-TEC-BOT-L, TL Interval Flow, Injection Pressure, TC Interval Pressure) showed positive MAE and RMSE improvement over naive carry-forward. Four TL temperature targets (TL-TEC-INT-U/L, TL-TEC-BOT-U/L) had zero variance in the test period, meaning the baseline was already perfect and any model prediction could only add error; these targets were excluded from the visualizations and will be excluded or flagged in future iterations
- TN Interval Flow exhibited the most severe failure, with MAE degrading by over 200% relative to baseline, suggesting the model latched onto noise in the delta signal for that target.
- Many targets converged to a `final_n_estimators` of 1 during walk-forward tuning, indicating that regularization correctly suppressed overfitting but also that the delta signal was too weak for the model to learn meaningful corrections beyond persistence. In future work, incorporating longer lag horizons, external covariates (e.g., operational schedules) could provide stronger signal-to-noise ratios and more opportunity to outperform persistence.

## HITL Review

### Does the model behavior make sense given domain expectations?
Broadly, yes. In subsurface thermal energy storage, pressure and flow readings at hourly resolution are expected to be highly autocorrelated; the system's thermal inertia means that conditions at time *t* are strongly predictive of conditions at *t+1*. It is physically reasonable that a persistence baseline achieves very low error and is hard to beat, especially during steady-state charging phases where injection rates and pressures are nearly constant. 

### Any concerning spurious correlations?
The feature space includes lag, rolling mean/std, and rate-of-change features derived from the same underlying sensors that appear as targets. While this is standard for time-series feature engineering, it creates a risk that the model relies on autocorrelative shortcuts rather than learning cross-variable physical relationships (e.g., pressure changes driving flow changes). Additionally, with only ~1,700 samples and 177 features, there is an elevated risk that the walk-forward tuning selected hyperparameters that fit noise in the validation folds rather than true signal.

### Any high-risk failure modes?
The most concerning failure mode is TN Interval Flow, where the delta model's MAE is well over the baseline. If used operationally, this target's predictions would be actively misleading and could mask actual flow anomalies. More broadly, because the model only marginally beats persistence on its best targets (~1–2% MAE improvement), any distributional shift — such as transitioning operational phases, equipment changes, or seasonal thermal boundary shifts — could easily push the model below baseline for all targets. 

## Reproducibility Steps
- Install requirements.txt (xgboost is required)
- Run scripts in order:
  - *Scripts\clean.py*
    - Outputs to Data\02_cleaned
  - Set PREDICT_DELTA to choose to run the original model (False) or the improved model (True) before running *Scripts\train.py*
    - Split dataset outputs to Data\03_split
    - Model outputs to Models\xgb_
  - Set PREDICT_DELTA before running *Scripts\evaluate.py*
    - Outputs to Models\xgb_ \eval_output
  - Set PREDICT_DELTA before running *Scripts\interpretability.py*
    - Outputs to Visualizations\xgb_
