# Technical Workflow Plan: Shifting Fire Regimes and Hydropower Stability in Washington

## Context

The AI Delinquents team wants to quantify how wildfire regime shifts in Washington are changing hydropower basin stability and potential sediment and water-quality stress. The workflow should connect wildfire history to upstream basin exposure, then convert that evidence into risk classes that are statistically defensible and validated against trusted observations.

### Scope and Key Decisions

- Geographic scope: Washington hydropower projects and their upstream basins
- Analysis horizon: Core 30-year window with annual series and decadal rollups
- Primary unit: Hydropower project with linked upstream watershed/catchment features
- Risk outputs: Low/Moderate/High tiers plus probability and uncertainty flags
- Method priority: Interpretable statistical evidence first, machine-learning classification second

---

## Phase 1: Study Design and Data Contracts

**Goal**: Lock analysis entities, boundaries, and reproducible data rules before modeling.

### Approach

1. Define project-to-basin entity chain:
   - Hydropower project point
   - Upstream hydrologic network/catchments
   - Basin-year and basin-decade feature tables
2. Finalize boundaries and identifiers:
   - HUC-8/HUC-12 mapping for each project
   - NHD-based upstream tracing rules
3. Build dataset contracts:
   - Required fields, CRS standard, valid date ranges
   - Missing-data policy and QA flags by source

**Deliverable**: Study design memo and schema specification for all downstream phases.

---

## Phase 2: Data Acquisition and Geospatial Integration

**Goal**: Build a unified, quality-checked basin-year geospatial dataset.

### Data Sources to Integrate

1. USGS combined wildfire polygons (historical fire extent)
2. Washington hydropower project/dam locations
3. NHD streams and catchments for flow-path and upstream logic
4. Watershed boundaries (HUC-8/HUC-12)
5. Land-cover layers
6. Slope/terrain layers

### Implementation

1. Ingest and standardize geometry/time fields.
2. Reproject to a common CRS and validate geometry/topology.
3. Delineate upstream contributing areas per hydropower project.
4. Intersect fire polygons with upstream basin units by year.
5. Persist a basin-year feature store with provenance and QA metadata.

**Tools**: Python, geopandas, shapely, rasterio/xarray, DuckDB/Parquet.

**Deliverable**: Unified geospatial feature store with reproducible ingestion scripts.

---

## Phase 3: Fire Regime and Exposure Feature Engineering

**Goal**: Quantify fire-regime changes and basin vulnerability drivers relevant to hydropower impacts.

### Fire Regime Metrics (Basin-Year)

- Burned area fraction
- Fire event frequency
- Mean fire size and upper-tail fire size
- Reburn interval and time-since-last-fire
- Decadal aggregates and rolling-window summaries

### Exposure and Impact Proxies

- Burned steep-slope fraction (erosion potential)
- Burned erodible land-cover fraction
- Connectivity proxy from burned hillslopes to stream channels
- Weighted upstream burned area with distance/path attenuation to intake/reservoir

**Deliverable**: Feature dictionary plus model-ready annual and decadal tables.

---

## Phase 4: Statistical Regime Shift Quantification

**Goal**: Produce defensible evidence that a basin's fire regime has shifted.

### Proposed Statistical Methods

1. Trend significance:
   - Mann-Kendall test for monotonic trend detection
2. Trend magnitude:
   - Sen's slope for robust effect size
3. Structural changes:
   - Change-point detection (Pettitt or Bayesian change-point)
4. Extreme behavior:
   - Upper-quantile trend or threshold exceedance frequency analysis

### Outputs per Basin

- Trend direction and significance
- Effect size estimates
- Shift timing indicators
- Confidence/uncertainty flags

**Deliverable**: Regime-shift evidence table supporting transparent risk decisions.

---

## Phase 5: Risk Modeling and Classification

**Goal**: Convert statistical and exposure evidence into practical hydropower risk tiers.

### Modeling Strategy

1. Baseline model:
   - Regularized logistic or ordinal regression for interpretability
2. Comparator model:
   - Random Forest or XGBoost for non-linear effects
3. Tiering:
   - Calibrated thresholds map probabilities to Low/Moderate/High
4. Explainability:
   - Global importance plus local explanations (for basin-specific drivers)

### Outputs

- Basin-level risk probability
- Risk class and confidence indicator
- Top contributing factors by basin

**Deliverable**: Trained, validated classifier outputs and interpretation package.

---

## Phase 6: Validation Against Trusted Data

**Goal**: Demonstrate that computed risk aligns with independent, trusted observations.

### Validation Streams

1. Metric sanity checks:
   - Compare computed fire summaries against trusted agency summaries
2. Environmental response checks:
   - USGS water-quality/sediment/turbidity signals where available
   - Reservoir sedimentation or operations records if accessible
3. Operational proxy checks:
   - EIA hydropower generation anomalies as a secondary stress proxy

### Validation Design

- Temporal holdout: train on earlier years, test on later years
- Spatial holdout: hold out basin groups to test transferability
- Performance metrics: ROC-AUC, PR-AUC, F1, calibration/Brier

**Deliverable**: Validation report with model performance, limitations, and confidence tiers.

---

## Phase 7: Delivery Artifacts and Competition Narrative

**Goal**: Package outputs for decision use and ESSD competition communication.

### Deliverables

1. Basin scorecards:
   - Risk tier, probability, trend/shift evidence, uncertainty
2. Ranked risk table:
   - Priority basins for investigation or mitigation planning
3. Map products:
   - Basin risk classes, shift evidence overlays, hydropower context
4. Methods package:
   - Data dictionary, runbook, assumptions, and model card

**Implementation Option**: Quarto report with interactive map/chart components.

---

## Technical Stack Summary

| Component            | Technology                                                 |
| -------------------- | ---------------------------------------------------------- |
| Data retrieval       | Python requests, API clients, file-based geodata ingestion |
| Data processing      | pandas, geopandas, DuckDB, Parquet                         |
| Spatial analysis     | geopandas, shapely, rasterio/xarray                        |
| Statistical analysis | scipy, statsmodels, pymannkendall (or equivalent)          |
| Modeling             | scikit-learn, xgboost (optional)                           |
| Visualization        | plotly, folium/leaflet, matplotlib                         |
| Documentation        | Quarto, Markdown                                           |
| Version control      | Git                                                        |

---

## Verification and Testing

1. Unit-test critical data transforms and feature calculations.
2. Spot-check upstream delineations for representative dams/basins.
3. Validate trend/shift stability under alternate windows (20-year vs 30-year).
4. Evaluate calibration and ranking stability across holdout strategies.
5. Re-run full pipeline end-to-end to confirm reproducibility.

---

## Suggested Timeline (4 Weeks)

| Week | Phase Focus | Deliverable                                            |
| ---- | ----------- | ------------------------------------------------------ |
| 1    | Phases 1-2  | Study design, ingestion scripts, unified feature store |
| 2    | Phases 3-4  | Feature tables and regime-shift evidence outputs       |
| 3    | Phases 5-6  | Risk model outputs and validation report               |
| 4    | Phase 7     | Final map/report package and competition narrative     |

---

## Notes and Risk Controls

- Fire severity fields may be inconsistent across historical records; include fallback metric set and confidence labels.
- Validation coverage will vary by basin due to station density and data availability; avoid overclaiming where observations are sparse.
- If timeline compresses, prioritize a minimum viable release: Phases 1-4 plus baseline classifier and one trusted validation stream.
