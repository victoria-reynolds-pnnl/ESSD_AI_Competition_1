# ByteMe — Week 3 ML Models Evaluation

**Team:** ByteMe | **Competition:** ESSD AI Competition | **Due:** April 9, 2026



## 1\. Train/Test Split Strategy

The temporal fracture dataset was resampled to a 1-minute interval. The dataset was chronologically split into strict `train` and `test` phases using fixed time partitions mapping to the experimental injection timeline at SURF. To prevent data leakage, explicit forward-looking targets (`target\_diff\_h`) were generated, and a rolling sequence length (`seq\_len=30`) was applied to provide temporal context for recurrent models. No Phase 2 (ambient water) target labels were mixed into Phase 1 (hot water) testing intervals.



## 2\. Model

#### 2.1. Model Inputs and Features

All five models are built using the exact same underlying base set of 15 engineered features, but they process them differently due to their architecture:

**Common Features:**

* **Base Physical \& Temporal:** `cumulative\\\_heat\\\_input`, `elapsed\\\_injection\\\_min`, `net\\\_flow\\\_rolling\\\_6h`, `TC\\\_INT\\\_delta`, `T\\\_gradient\\\_INT\\\_TC`, `days\\\_since\\\_injection`, `hour\\\_sin`, `hour\\\_cos`, `delta\\\_T\\\_above\\\_T0\\\_TN`
* **Explicit Lags:** `target\\\_lag\\\_15`, `target\\\_lag\\\_30`, `target\\\_lag\\\_60`, `flow\\\_lag\\\_15`, `flow\\\_lag\\\_30`, `flow\\\_lag\\\_60`

**Differences:**

1. **Structural Dimensionality:** **LSTM \& Transformer** receive a 3D sequential shape (`batch, 30, 15`) covering the uninterrupted history of the past 30 timesteps. In contrast, **Multi-RF, Multi-XGBoost, \& MLP** use only the single most recent tabular timestep, bypassing the full 30-minute block and relying solely on explicit lagged features for context.
2. **Feature Scaling:** **Neural Networks** (LSTM, Transformer, MLP) use features transformed by a `StandardScaler` to ensure safe gradients. The **Decision Trees** (RF, XGBoost) bypass scaling and consume the raw unscaled data structure.



#### 2.2 Model Performance Summary

We consolidated five core architectures into our evaluation script. The quantitative table details both Mean Squared Error (MSE) and R-Squared (R²) metrics spanning all temporal prediction endpoints:

##### Performance Metrics Table

|Model Architecture|MSE (15m)|MSE (60m)|MSE (240m)|MSE (1440m)|R² (15m)|R² (60m)|R² (240m)|R² (1440m)|
|-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**Multi-RF (RF-Lag)**|0.0063|0.0063|0.0077|0.0350|0.6775|0.6799|0.6048|0.0747|
|**Multi-XGBoost**|0.0070|0.0064|**0.0062**|**0.0333**|0.6394|0.6745|0.6793|**0.1183**|
|**MLP**|0.0079|0.0117|0.0154|0.0992|0.5953|0.3986|0.2076|-1.6242|
|**LSTM**|0.0075|0.0058|0.0061|0.0460|0.6176|0.7022|**0.6867**|-0.2162|
|**Transformer**|**0.0057**|**0.0056**|0.0062|0.0517|**0.7093**|**0.7141**|0.6815|-0.3672|
|**Persistent (Baseline)**|0.0075|0.0076|0.0076|0.0352|0.6131|0.6085|0.6074|0.0688|

#### Analysis

* **Transformer**: The absolute best sequence model post-tuning execution, locking fundamentally superior baseline metrics across the immediate short term (`0.71 R²` on 15m and 60m horizons).
* **LSTM**: Upon executing a manual reversion rollback following an aggressive overfitting gradient collapse, the sequence loops beautifully restructured, commanding highly reliable mid-range spans (`0.68 R²` at +240m).
* **Multi-XGBoost**: Remains the undisputed physical saturation navigator, effectively generating the only stable extrapolations past 24 hours (`0.118 R²`).
* **Multi-RF**: Unwavering consistency dynamically, matching XGBoost up to 60m natively.
* **Persistent (Baseline)**: Acts as our naive benchmark, achieving `\~0.61 R²` on short horizons by simply assuming zero change.
* **MLP**: Responded immensely well to Phase 2 iterative limits, skyrocketing from \~0.20 R² early ranges to firmly maintaining `0.59 R²`.

**Conclusion**: Hyperparameter optimization severely shifted sequence-processing ceilings. The `LSTM` model emerged as the undisputed short-term structure, while the decision tree logic (`Multi-XGBoost`) remains distinctly superior for isolated 4-hour phase extrapolation bounds.



## 3\. Failures and Limitations

1. In the original 1 sec data, there were bad measurements (0) in the temperature columns, and they were averaged into the minute data, leading to spikes within the data. Due to time constraints, the data cleaning will be redone and reincorporated for Week 4 submission. 
2. Raw target data trend leads to out of distribution data for the model to predict. This is resolved by predicting the difference between the current time step and the target time step.
3. Deep models (LSTM, Transformer) trains very slowly on CPU. This is resolved by using MPS on Apple Silicon or GPU on other platforms.
4. Choice of best model depends on the application. For short-term forecasting, Transformer is the best model. For long-term forecasting, XGBoost is the best model.
5. Model hyperparameter tuning is very time consuming. Same hyperparameter settings may not be optimal for all time horizons.



## 4\. Data Files

One data file exceeds GitHub's 100 MB file size limit and is not stored in this repository. It is available on the team SharePoint folder:

| File | Size | Description |
|------|------|-------------|
| `Data/week2_data.csv` | ~144 MB | Processed dataset from Week 2 used as input for model training |

**SharePoint link:** [week2\_data.csv](https://pnnl.sharepoint.com/:x:/r/teams/ESSDAICompetition/Shared%20Documents/General/Team%2010%20Byte%20Me/Week%203/ESSD_AI_Competition_1/Data/week2_data.csv?d=wcc4d8bba9caf42e890cf97a0d85d865a&csf=1&web=1&e=fmgP4t)

The following data files are included in this repository (under 100 MB):

| File | Description |
|------|-------------|
| `Data/ml_results.csv` | Model prediction outputs |
| `Data/model_performance_metrics.csv` | Consolidated MSE and R² metrics across all models and horizons |
| `Data/splits_indexed.csv` | Indexed train/test split boundaries |



## 5\. HITL Review



*Does the model behavior make sense given domain expectations?*

The models predict an overall positive correlation between temperature rise at the producer (TN) and increases in both flow rate and temperature at the injector (TC). This behavior aligns with physical expectations of the system, and the raw data confirm that temperature rise at TN is strongly influenced by both flow rate and temperature changes at TC.

&#x20;

Feature importance across different time horizons is also consistent with observed system behavior. Shorter forecast horizons better capture high-frequency oscillations (e.g., fluctuations in injector temperature at TC), while longer horizons emphasize smoothed trends through features such as net flow and cumulative heat input, which effectively de-trend oscillatory behavior in flow and temperature.

&#x20;

For the +1440 extrapolation, model performance degrades at later time stages. This is expected, as the models are trained solely on historical data and must extrapolate beyond the observed range, leading to increased uncertainty and edge effects.

&#x20;

*Any concerning spurious correlations?*

The observed data exhibit large oscillations in temperature; however, the models (1) suppress predictions that appear as extreme outliers and (2) produce de-trended outputs with reduced oscillation magnitude. Despite this smoothing, the AI/ML models capture the overall trend effectively, demonstrating strong predictive capability in the presence of noisy data.



*Any high-risk failure modes?* 

Further review of the cleaned dataset revealed that temperature values equal to zero (invalid measurements) were inadvertently included in the 1-minute resampled averages, artificially amplifying oscillations. The AI-assisted cleaning process did not filter these zero values (see Week 2 Cleaning documentation). The team anticipates that removing these erroneous values will reduce noise and further improve model performance. Due to time constraints, this updated dataset will be incorporated in the Week 4 submission.



## 6\. Reproducibility Steps

Follow these steps to reconstruct our consolidated metrics:

1. Ensure the raw `processedData` is extracted to `ByteMe\_wk3/Data/processedData/`.
2. Execute the setup sequence:

```bash
   cd ByteMe\_wk3
   pip install -r requirements.txt
   ```

3. Train and export the ML/DL weights:

```bash
   python Scripts/train.py
   ```

4. Evaluate trained models on testset and compute error metrics:

```bash
   python Scripts/evaluation.py
   ```

   *(Plots will populate under `Visualizations/`).*

5. Generate SHAP, PDP and other interpretability analysis:

```bash
   python Scripts/interpretability.py
   ```


## AI Use disclosure
We used AI tools to help us with the following tasks:

* Debugging code
* Code cleaning
* Writing documentation

