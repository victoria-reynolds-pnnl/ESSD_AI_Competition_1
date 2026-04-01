Technical Workflow Plan: Idea 2 (Wildfire Regime Shifts and Hydropower Stability)

1. Scope and Unit of Analysis
   Define study domain as Washington hydropower projects with upstream contributing basins.
   Use hydropower asset as the reporting unit, with nested watershed units (HUC-8/HUC-12) and upstream burned-area summaries.
   Set primary analysis window to the most recent 30 years, with decadal summaries (e.g., 1991-2000, 2001-2010, 2011-2020).
2. Data Ingestion and Harmonization
   Load wildfire polygons (USGS combined fire dataset) and standardize year, area, and geometry.
   Compile hydropower dam/project locations and map each to upstream catchments using NHD flow networks.
   Add watershed boundaries (HUC-8/HUC-12), land cover, and slope/terrain rasters.
   Build a common geospatial index and time index so all features aggregate consistently by basin and year.
3. Feature Engineering (Risk Drivers)
   Fire regime metrics per basin-year:
   Burned area fraction
   Fire frequency (count/year)
   Mean and upper-tail fire size
   Reburn interval and time-since-last-fire
   Terrain/erosion susceptibility metrics:
   Percent burned area on steep slopes
   Burned area in high-erosion land-cover classes
   Connectivity of burned hillslopes to stream channels
   Hydropower exposure metrics:
   Upstream burned area weighted by distance/travel path to reservoir/intake
   Fraction of contributing area burned in last 1-3 years
4. Statistical Quantification of Regime Shift
   Trend detection:
   Mann-Kendall test for monotonic trends in key fire metrics
   Sen’s slope for robust trend magnitude
   Shift detection:
   Change-point methods (e.g., Pettitt or Bayesian change-point) on annual burned-area and fire-frequency series
   Extreme-event characterization:
   Peaks-over-threshold or quantile trend analysis for upper-tail fire behavior
   Output:
   For each basin, estimate trend significance, effect size, and shift timing with uncertainty bounds.
5. Risk Modeling and Classification
   Define risk target as potential sediment/water-quality stress to hydropower operations.
   Build an interpretable baseline model first (regularized logistic/ordinal regression), then compare to tree-based ML (Random Forest/XGBoost).
   Generate risk classes (Low/Moderate/High) from predicted probabilities using calibrated thresholds.
   Use model explanation tools (feature importance + SHAP) to identify dominant risk drivers per basin.
6. Validation Against Trusted Data
   Validate intermediate fire metrics against trusted external summaries where available (state/federal fire statistics).
   Validate impact relevance using trusted hydro-environmental observations:
   USGS water quality/suspended sediment/turbidity stations downstream of burned basins
   Reservoir sedimentation records or utility-reported sediment management signals (if available)
   Hydropower generation anomalies from EIA time series as secondary proxy
   Validation design:
   Time-split cross-validation (train on earlier years, test on later years)
   Spatial holdout by basin groups to test transferability
   Report performance:
   Classification: ROC-AUC, PR-AUC, F1, calibration (Brier score, reliability curve)
   Risk ranking: Spearman rank correlation and top-k hit rate for high-risk basins
7. Deliverables
   Basin-level risk scorecard with uncertainty and confidence flags.
   Washington map layers showing regime-shift evidence and hydropower risk class.
   Short technical memo documenting methods, assumptions, and limitations.
   Reproducible workflow (notebooks/scripts + data dictionary + model card).
8. Practical Execution Sequence
   Week 1: Data assembly, geospatial joins, upstream delineation, QA/QC.
   Week 2: Feature engineering and statistical trend/shift analysis.
   Week 3: Risk model training, calibration, explainability analysis.
   Week 4: Validation against trusted observations, finalize maps and competition narrative.
   If you want, I can now turn this into a concrete task board with exact scripts/modules and file structure for implementation.
