# Zero Shot Hopefuls - Compound Hazard Forecasting

## Week 3: ML Model Training and Assessment

### Train/Test Split Strategy

We use a strictly chronological split to respect the temporal structure of the data and prevent data leakage from future climate conditions into training. **Train (2001-2015):** 1,593,527 rows (75%) spanning 15 years that capture a wide range of compound event frequencies (1.25% to 12.02% annual positive rate). **Validation (2016-2017):** 216,192 rows (10%) used exclusively for hyperparameter tuning and classification threshold selection; these years include both a moderate-event year (2016, 1.68%) and a low-event year (2017, 1.15%), testing model robustness under varying conditions. **Test (2018-2020):** 314,154 rows (15%) held out entirely until final evaluation. Data was never shuffled, and each row is assigned to exactly one split based on its year. We excluded `compound_hw_drought` and `fire_during_compound` from the feature set because they are definitionally linked to the target variable (`target_compound_1wk` is the 1-week forward shift of `compound_hw_drought`); including them would inflate metrics through autocorrelation rather than learning from the underlying physical drivers. All 26 retained features (temperature, drought indices, meteorological variables, and engineered temporal features) represent information observable at or before the prediction week. The test set was evaluated exactly once; those numbers are final.

### Model Performance Summary

| Model               | Split      | F1     | AUC-ROC | Precision | Recall | Brier Score | Threshold |
| ------------------- | ---------- | ------ | ------- | --------- | ------ | ----------- | --------- |
| Logistic Regression | Validation | 0.4736 | 0.9792  | 0.4384    | 0.5149 | 0.0350      | 0.86      |
| Logistic Regression | Test       | 0.5790 | 0.9813  | 0.5184    | 0.6557 | 0.0422      | 0.86      |
| LightGBM            | Validation | 0.5435 | 0.9892  | 0.4374    | 0.7177 | 0.0151      | 0.65      |
| LightGBM            | Test       | 0.6432 | 0.9886  | 0.5385    | 0.7984 | 0.0196      | 0.65      |

LightGBM outperforms Logistic Regression across all metrics on both validation and test sets. On the held-out test set, LightGBM achieves an F1 of 0.6432 versus 0.5790 for LR, with notably higher recall (0.80 vs 0.66), meaning it catches 80% of compound events while maintaining 54% precision. Both models show excellent discriminative ability (AUC > 0.98), indicating they rank compound-event weeks much higher than non-event weeks. LightGBM's lower Brier score (0.020 vs 0.042) indicates better-calibrated probability estimates. Classification thresholds were optimized on the validation set to maximize F1 — the default 0.5 threshold would predict nearly all weeks as non-events given the 4% positive rate. The LR threshold (0.86) is notably higher than LightGBM's (0.65) because `class_weight="balanced"` shifts the LR probability distribution upward. We chose F1 as the primary metric because accuracy alone would be 96% by always predicting negative; AUC-ROC for threshold-independent discrimination relevant to operational decision-making; and Brier score because downstream users need well-calibrated probabilities for risk assessment under uncertainty.

### HITL Review

**Does the model behavior make sense given domain expectations?** Yes. SHAP analysis shows PDSI (Palmer Drought Severity Index) as the top feature driving LightGBM predictions, with a clear threshold effect: PDSI values below approximately -1 (drought conditions) produce strongly positive SHAP values, while normal or wet conditions (PDSI > 0) push predictions toward non-events. Seasonal features rank highly, which is physically expected since compound heatwave+drought events are concentrated in summer months. The heatwave indicator HI02 has strong positive SHAP impact when active (value=1). The Logistic Regression coefficients tell a consistent story: seasonal_cos (most negative, capturing summer peak), pdsi (negative — lower PDSI means more drought means higher risk), and tmax (positive — higher temperatures increase risk) are the top three. Both models agree that drought indices and temperature/heatwave variables are the primary drivers, which aligns with the physical definition of compound heatwave+drought events.

**Any concerning spurious correlations?** BCOC (black carbon + organic carbon aerosol concentration) ranks 6th in LightGBM feature importance, which warrants scrutiny. BCOC is an indicator of biomass burning aerosols — it could be detecting co-occurring wildfire smoke during hot, dry conditions rather than causally driving compound events. This is likely a correlational signal (fires happen when it's hot and dry) rather than a causal one. In a deployment scenario, BCOC might not be available as a real-time predictor 1-2 weeks ahead, so this dependency should be monitored. Fire variables (FHS_c9, BA_km2, maxFRP_c9) rank at the bottom, suggesting the model is not using concurrent fire activity as a shortcut — the BCOC signal appears to capture a different, more diffuse aerosol pattern.

**Any high-risk failure modes?** The most concerning failure mode is performance during low-event years. The validation set (2016-2017) has only a 1.41% positive rate compared to the training set's 4.72%, and the test set sits at 2.65%. While the model handles this variation well (test F1 actually exceeds validation F1, likely because 2018 and 2020 have more typical event rates), 2019 specifically had only 0.36% compound events — the lowest in the entire 20-year record. The model likely over-predicts in such anomalous years, generating false alarms. Additionally, the model treats each grid cell independently without explicit spatial modeling, so it cannot learn that adjacent cells tend to experience compound events simultaneously. Clustered false negatives in specific regions (e.g., transition zones between climate regimes) may indicate spatial blind spots.

### Failures and Limitations

The 4% class imbalance means even with class-weight balancing and threshold optimization, the model produces a non-trivial false alarm rate — on the test set, LightGBM generated 5,702 false positives alongside 6,654 true positives (precision 54%). For operational use, this 1:1 TP/FP ratio may require further threshold adjustment depending on the cost asymmetry between missed events and false alarms. Extreme low-event years like 2019 (0.36% positive rate) are poorly represented in training and may exhibit degraded precision. Spatial autocorrelation is not explicitly modeled: adjacent 0.5-degree grid cells share atmospheric conditions but are treated as independent observations, which inflates the effective sample size and may cause overly optimistic uncertainty estimates. The compound event definition uses a fixed PDSI category threshold (<=3, mild drought or worse) that may not be universally appropriate across all climate zones — arid regions in the Southwest may have structurally different drought dynamics than the humid Southeast. Our temporal features look back at most 14 days (drought_trend_14d), but drought builds over months to years; incorporating longer-horizon antecedent conditions (e.g., 90-day or seasonal PDSI trend) could improve lead-time prediction. Finally, the 20-year record (2001-2020) may be insufficient to capture decadal oscillation patterns (e.g., PDO, AMO) that modulate compound event frequency.

### Reproducibility Steps

First create a directory named "dataset" at root, within it have the "ComExDBM_CONUS" dataset.

```bash
# From the repository root:
source .venv/Scripts/activate
uv pip install -r zero-shot-hopefuls/requirements.txt
         # ~5 min: exports CSV, JSON, data dictionary

# Week 3 pipeline (requires Week 2 outputs):
python zero-shot-hopefuls/Scripts/train.py                 # ~5 min: splits data, trains LR + LightGBM, saves models
python zero-shot-hopefuls/Scripts/evaluate.py              # ~1 min: optimizes thresholds, evaluates on test set
python zero-shot-hopefuls/Scripts/interpretability.py      # ~2 min: SHAP analysis, feature importance, visualizations
```

**Role of AI:** Claude (Anthropic) was used to generate the training, evaluation, and interpretability scripts, with each AI-generated section cited in code comments. Claude also assisted with split strategy design, feature selection rationale (leakage analysis), metric selection justification, and HITL review framing. All AI-generated code was reviewed by the team.

## File Structure

```
zero-shot-hopefuls/
  Data/
    original_sample.csv                        # 1,000-row raw daily data sample
    cleaned_compound_hazard_weekly.csv         # Cleaned dataset (2019-2020 subset, CSV format)
    cleaned_compound_hazard_weekly.json        # Cleaned dataset (10K-row sample, JSON format)
    data_dictionary.json                       # Feature dictionary with descriptions and sources
    splits_indexed.csv                         # Train/val/test split assignments for every row
  Models/
    logistic_regression.pkl                    # Trained LR model + StandardScaler
    lightgbm.pkl                               # Trained LightGBM model (best hyperparameters)
  Scripts/
    01_load_and_merge.py                       # Load 4 NetCDF sources, merge, save daily data
    02_clean_and_engineer.py                   # Clean, engineer 7 features, aggregate weekly
    03_export.py                               # Export to CSV, JSON, data dictionary
    train.py                                   # Chronological split, train LR + LightGBM, tune hyperparameters
    evaluate.py                                # Threshold optimization, test-set evaluation, results CSV
    interpretability.py                        # SHAP analysis, feature importance, visualization generation
  Visualizations/
    lr_coefficients.png                        # Logistic Regression standardized coefficients
    lgb_feature_importance.png                 # LightGBM gain-based feature importance
    lgb_shap_summary.png                       # SHAP beeswarm plot for LightGBM
    lgb_shap_dependence_top.png                # SHAP dependence plot for top feature (pdsi)
  Outputs/
    ml_results.csv                             # Model performance metrics (F1, AUC, precision, recall, Brier)
  requirements.txt                             # Python environment dependencies
  README.md                                    # This file
```
