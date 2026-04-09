# Train/Test Split Strategy

We adopted an 80/10/10 random split stratified by monitoring_location_id. A year-based temporal split was considered but risked concentrating climatically distinct periods entirely within one partition, potentially biasing a model. The random split approach ensures each site and the full range of seasonal and interannual variability are represented across all three partitions. To prevent data leakage, all engineered features (lags, rolling windows, STL decomposition) use only backward-looking calculations, duplicate records were removed during cleaning, and target values are verified not to share dates with feature rows in another partition, ensuring the model extrapolates rather than interpolates.

# Model Performance Summary



# HITL Review (3 questions)



# Failures and Limitations

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