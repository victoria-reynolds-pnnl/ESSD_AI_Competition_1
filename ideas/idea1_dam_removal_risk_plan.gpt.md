# Technical Workflow Plan: Dam Removal Risk Assessment System (GPT)

## Context

This plan implements Idea 1 by creating a data-driven framework to estimate dam removal risk across engineering, environmental, social, political, and economic dimensions. The workflow prioritizes trusted public data sources, transparent scoring logic, and statistical validation against historical removals.

### Scope and Key Decisions

- Geographic scope: Columbia River Basin-focused analysis with extension path to broader Pacific Northwest
- Analysis unit: Dam-level records with linked watershed, streamflow, and energy context
- Risk output: Low/Moderate/High tier plus probability and confidence indicators
- Method strategy: Interpretable baseline (MCDA + logistic regression), non-linear comparator (random forest)
- Validation anchor: Historical dam removal records from trusted databases

---

## Phase 1: Risk Definition and Factor Taxonomy

**Goal**: Define measurable risk factors and scoring rules before data engineering.

### Risk Categories and Indicators

| Category      | Example Indicators                                                                      | Trusted Sources                          |
| ------------- | --------------------------------------------------------------------------------------- | ---------------------------------------- |
| Engineering   | Dam age, condition, operational reliability, capacity factor                            | EIA 860, USACE, FERC licensing records   |
| Environmental | Fish passage impact, sediment disruption, water-quality stress, ecosystem fragmentation | USGS disturbance metrics, WA Ecology EIM |
| Social        | Population dependency, local vulnerability, recreational/tribal relevance               | Census ACS, official state records       |
| Political     | License expiration, legal/treaty exposure, formal regulatory pressure                   | FERC docs, Columbia River Treaty sources |
| Economic      | Generation value, grid role, replacement cost, O&M burden                               | EIA 923/930, BPA, PUDL                   |

**Deliverable**: Risk taxonomy and scoring definitions (units, directionality, thresholds, source constraints).

---

## Phase 2: Data Acquisition and Provenance Pipeline

**Goal**: Build repeatable ingestion from trusted data providers with quality metadata.

### Data Streams

1. USGS Water Data API (streamflow and gage context)
2. EIA 860/923/930 and PUDL (asset and generation context)
3. USACE hydropower data (engineering and operations context)
4. FERC licensing documents (regulatory/political features)
5. Census ACS and official WA socioeconomic data (social context)
6. USGS disturbance metrics and WA Ecology EIM (environmental context)
7. Columbia River Treaty official material (policy context)

### Implementation Notes

- Use scripted ingestion with explicit source timestamps.
- Preserve raw snapshots and write curated tables in columnar formats.
- Attach provenance and quality flags to each field.

**Deliverable**: Versioned raw and curated datasets with ingestion logs.

---

## Phase 3: Dam-Centric Data Integration and Linkage

**Goal**: Create one integrated analytical table per dam with spatial and temporal coherence.

### Approach

1. Build unified dam key crosswalk (NID/EIA/FERC where available).
2. Spatially link dams to watersheds and nearby upstream/downstream gages.
3. Align temporal granularity across metrics (monthly/annual rollups as needed).
4. Construct a completeness matrix by dam and risk category.

**Deliverable**: Integrated feature store + completeness table for downstream modeling.

---

## Phase 4: Statistical Risk Quantification

**Goal**: Convert heterogeneous indicators into defensible risk estimates.

### Methods

1. Normalization

- Apply z-score or min-max scaling for comparability across metrics.

2. MCDA Composite Risk

- Start with equal category weights.
- Store weights as configurable parameters for sensitivity analysis.
- Compute weighted composite score.

3. Supervised Probability Model (if labels sufficient)

- Baseline: logistic regression.
- Comparator: random forest.
- Output calibrated removal probability.

4. Uncertainty Estimation

- Use bootstrap intervals and/or Bayesian treatment for sparse/noisy factors.
- Propagate data-quality flags into confidence indicators.

5. Tiering

- Convert scores/probabilities into Low/Moderate/High tiers via percentile or ROC-informed cutoffs.

**Deliverable**: Dam-level risk score, probability, tier, and uncertainty/confidence outputs.

---

## Phase 5: Validation Against Trusted Data

**Goal**: Verify that high-risk outputs align with observed removal patterns.

### Validation Strategy

1. Ground truth: American Rivers Dam Removal Database and other verified historical removals.
2. Model/score diagnostics:

- Discrimination: ROC-AUC, PR-AUC
- Calibration: Brier score, reliability curves
- Ranking utility: top-k precision/recall for high-risk dams

3. Robustness checks:

- Temporal holdout splits
- Sensitivity to weighting changes
- Sensitivity to missing-data handling choices

**Deliverable**: Validation report with performance metrics and failure-case analysis.

---

## Phase 6: Data Gap and Confidence Analysis

**Goal**: Identify missing data that most affects risk reliability and actionability.

### Approach

1. Score completeness per dam and per category.
2. Flag high-impact missing fields (largest effect on uncertainty).
3. Distinguish imputable gaps from collection-required gaps.
4. Attach confidence bands/flags to each dam's risk output.

**Deliverable**: Data-gap prioritization report and confidence-annotated risk table.

---

## Phase 7: Delivery and Reporting

**Goal**: Package outputs for transparent decision support and competition submission.

### Output Package

1. Dam scorecards: risk tier, probability, top drivers, confidence level.
2. Ranked risk table: priority candidates for deeper review.
3. Map-ready geospatial layer for basin-scale visualization.
4. Methods appendix: source list, assumptions, validation summary, caveats.
5. Reproducibility runbook: rerun instructions and configuration options.

**Deliverable**: Final technical report and reproducible artifacts.

---

## Technical Stack Summary

| Component            | Recommended Tools                               |
| -------------------- | ----------------------------------------------- |
| Ingestion            | Python requests, API clients, scheduled scripts |
| Processing           | pandas, geopandas, DuckDB                       |
| Spatial linkage      | geopandas, shapely                              |
| Statistical modeling | scikit-learn, statsmodels, PyMC (optional)      |
| Visualization        | plotly, folium/leaflet, matplotlib              |
| Documentation        | Quarto/Markdown                                 |
| Versioning           | Git                                             |

---

## Verification Checklist

1. All risk factors map to trusted sources with explicit definitions.
2. Dam ID crosswalk and spatial joins pass spot checks on known examples.
3. Risk tiers remain reasonably stable under alternate weighting schemes.
4. Calibration metrics are acceptable for decision-support use.
5. Data-gap flags are present for low-confidence dams.

---

## Suggested 4-Week Timeline

| Week | Focus      | Deliverable                                          |
| ---- | ---------- | ---------------------------------------------------- |
| 1    | Phases 1-2 | Taxonomy, ingestion scripts, curated source tables   |
| 2    | Phase 3    | Integrated dam feature store and completeness matrix |
| 3    | Phases 4-5 | Risk scoring/models and validation report draft      |
| 4    | Phases 6-7 | Data-gap analysis, final outputs, submission package |

---

## Notes

- Political context features should rely only on official regulatory and treaty documentation to reduce partisan bias.
- Where labels are sparse or regionally biased, treat supervised model outputs as decision-support signals, not definitive predictions.
- Minimum viable delivery for competition timelines: Phases 1-5 plus confidence and caveat appendix.
