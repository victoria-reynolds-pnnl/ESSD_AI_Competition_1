User: List the types of data recorded in the ComExDBM_CONUS Input files.

GitHub Copilot: Optimizing tool selection...I'll open the Input folder README to find what data types are recorded.

Read [](file:///c%3A/Users/sasl988/OneDrive%20-%20PNNL/Documents/ESSD%20AI%20Competition%202026/ESSD_AI_Competition_1/ComExDBM_CONUS/01_Input/README.md)

Now adding a short TODO entry to track this task.

Created 1 todo

Progress: I've read the Input README and created a short TODO. Next: here's the concise list of data types recorded in the ComExDBM_CONUS Input files.

- **Fire detections & summaries:** fire mask classes, pixel quality, maximum fire radiative power (MaxFRP), fire pixel position, daily summaries (mean/max MaxFRP, total fire hotspots).
- **Burned-area / fire events:** MODIS MCD64A1 burned area and FIRED-derived fire events/daily burned area.
- **Temperature:** CPC Global Unified daily gridded temperature (daily temperature anomalies) used for heatwave detection.
- **Drought indices:** gridMET-derived indices such as SPI, SPEI, PDSI, and other drought-related metrics at multiple time scales.
- **Meteorological variables:** NARR (e.g., surface temperature, 3‑hr precipitation, soil moisture, relative humidity, specific humidity, latent/sensible heat fluxes, winds at standard pressure levels) and other reanalysis fields.
- **Aerosols & atmospheric composition:** MERRA-2 carbonaceous aerosols (black carbon + organic carbon) and related atmospheric variables (humidity, winds at multiple levels).
- **Vegetation indices:** Leaf Area Index (LAI), Normalized Difference Vegetation Index (NDVI).
- **Fire Weather Index (FWI):** daily fire weather index fields from global FWI datasets.

Would you like these grouped into a one-line CSV or exported to a new file in the repo?

User: How is the data organized and structured?

GitHub Copilot: Summarizing how the project data is organized and structured.

- **Top-level layout:** repository root contains the ComExDBM_CONUS project folder (ComExDBM_CONUS).
- **Raw inputs:** 01_Input (see README) — raw/source datasets organized by data provider/type (MODIS MOD14A1, MODIS MCD64A1 / FIRED, CPC temperature, gridMET drought indices, NARR, MERRA‑2, LAI/NDVI, Global FWI). Also contains spatial masks/shapefiles in `CONUS4326`.
- **Processing code:** 02_Code — Python scripts used to create summaries, label events, match fires/heatwaves, compute drought metrics, and save outputs.
- **Analysis & intermediate products:** 03_Analysis — intermediate NetCDFs and analysis figures (daily/annual aggregates, maps, and plot folders).
- **Final outputs:** 04_OutputData — processed ComExDBM products organized by theme (e.g., `01_HeatWave`, `02_FireData`, `03_DroughtData`, `04_Meteorological_Variables`), typically as per-year NetCDF files named like `ComExDBM_YYYY_HeatWave_V02.nc` and similar.
- **Per-source structure & metadata:** each input source is kept with its native spatial/temporal resolution and documented in the input README (examples: MOD14A1 — daily ~1 km; MCD64A1 — monthly 500 m with FIRED-derived daily burned-area; CPC — 0.5° daily anomalies; gridMET — 4 km drought indices at multi-day aggregations; NARR/MERRA‑2 — 3‑hour reanalysis fields; LAI/NDVI — daily 0.05°; FWI — daily 0.5°×0.625°).
- **Naming & summary convention:** processed outputs are grouped by hazard type and year, with summary variables (e.g., mean/max MaxFRP, FHS counts, drought indices, heatwave summaries) stored in the NetCDFs under `04_OutputData`.

Would you like me to create a short CSV inventory of input sources and resolutions in the repo?

User: How would you use machine learning and this data set to estimate the probability of compound hazard (co‑occurring heatwave and drought, or wildfire during those conditions) 1–2 weeks ahead and identify regional drivers?

GitHub Copilot: I'll add a concise TODO plan for the ML workflow, then outline the modeling approach and steps.

Created 9 todos

Progress: created a 9‑step TODO for the ML workflow. Next I'll summarize the modeling design and concrete steps.

Plan summary — how I'd estimate 1–2 week compound‑hazard probabilities and identify drivers

1. Define targets

- Binary/multi‑label targets per grid cell/region for lead times 7 and 14 days: Heatwave (HW), Drought (DR), Wildfire occurrence (WF).
- Compound events: joint label for co‑occurrence (HW & DR), and WF during HW/DR. Options: (a) direct multi‑label classifier for joint events; (b) model marginals and estimate joint probability via P(A∩B)=P(A)·P(B|A) with a conditional model.

2. Data assembly & preprocessing

- Use inputs from 01_Input (MOD14A1, MCD64A1/FIRED, CPC temperature, gridMET indices, NARR/MERRA‑2, LAI/NDVI, FWI).
- Regrid / resample to a common spatial grid (choose resolution balancing signal vs compute; e.g., 0.05° or 4 km). Align temporal resolution to daily and aggregate where needed (3‑hr → daily). Fill missing with conservative imputation and flag gaps.
- Create target times by labeling each grid cell at day t with event occurrence in window [t+7] and [t+14].

3. Features & engineering

- Lagged predictors: rolling means/sums (1–14, 7–28 days) of temperature anomalies, precipitation, SPI/SPEI, PDSI, soil moisture, MaxFRP, FWI, LAI/NDVI, humidity, winds, aerosol (BC+OC).
- Anomalies & climatology: standardized anomalies relative to seasonal cycles.
- Derived indices: recent trends, day‑of‑year, persistence features, antecedent fire counts.
- Include spatial context: neighborhood aggregates (e.g., 3×3 cell mean), elevation/landcover/eco‑region masks, distance to recent fires.
- Optionally include NWP/forecast features if available (ensemble means) to extend predictive skill for 7–14 days.

4. Modeling approach

- Baselines: persistence, climatology, logistic regression.
- Primary models (probabilistic): Gradient-boosted trees with probabilistic outputs (LightGBM / XGBoost / CatBoost using a logistic objective), and probabilistic neural nets (e.g., MC‑dropout or deep ensembles) for complex nonlinear interactions.
- Multi‑output strategies: (a) single model predicting joint probability directly, (b) multi-task model predicting marginals + conditional, (c) classifier chain.
- Handle class imbalance with class weights, focal loss, or up/down sampling.

5. Validation & CV

- Time‑aware splits: train on past years, validate on later years. Use rolling origin evaluation for lead times.
- Spatial CV: leave‑region‑out folds to test generalization across regions.
- Metrics: Brier score (main for probabilistic), CRPS (if continuous forecasts), ROC AUC, precision‑recall, reliability diagrams, and sharpness. Compare against persistence/climatology.

6. Calibration & uncertainty

- Calibrate probabilities (Platt / isotonic / beta calibration).
- Quantify uncertainty via probabilistic ensembles, Bayesian models, or prediction intervals (for neural nets use ensembles or MC dropout).
- Produce lead‑time dependent reliability diagnostics.

7. Explainability & regional drivers

- Use SHAP (TreeSHAP for boosted trees) to get local and global feature attributions. Aggregate SHAP by region, season, and lead time to identify dominant drivers (e.g., antecedent soil moisture, positive temp anomaly, high FWI, low LAI).
- Map spatial patterns of feature importance and partial dependence plots to show nonlinear effects.
- Use interaction detection (SHAP interaction values) to find combinations (e.g., high temp anomaly + low soil moisture → elevated joint risk).
- Cluster events/regimes (k‑means or spectral clustering on features) to define regional modes and their drivers.

8. Postprocessing & deployment

- Produce gridded probabilistic maps for 7 and 14‑day leads and regional summaries.
- Build an ingestion pipeline to update with newest reanalysis/NWP and retrain periodically. Containerize model + inference.
- Provide user‑facing products: probability maps, reliability diagrams, top‑drivers per region, and alerts when joint probability exceeds thresholds.
- Log performance and drift; set automated revalidation.

9. Tests / case studies

- Backtest against historical compound events (2001–2020) and perform event‑based evaluation (hit rate, false alarms) and prospective case studies (major heat‑drought‑fire years).
- Produce interpretability reports for a small set of high‑impact events to validate driver explanations.

Deliverables I would produce

- Reproducible data prep scripts (`02_Code` style), training notebooks, and saved model artifacts.
- Evaluation dashboards (Brier, reliability, AUC) and SHAP‑based driver maps.
- A short README describing model inputs, versions, and recommended operational thresholds.

If you want, I can:

- (A) scaffold the reproducible pipeline (data‑prep + feature engineering scripts) under 02_Code, or
- (B) prototype a baseline LightGBM model for 7‑day joint HW+DR probability on a small region and produce SHAP driver maps.

Which do you want me to start?
