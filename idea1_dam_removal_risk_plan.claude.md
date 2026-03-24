# Technical Workflow Plan: Dam Removal Risk Assessment System

## Context

The AI Delinquents team is participating in the ESSD AI Competition and wants to build a data-driven risk assessment system that identifies dams at high risk of removal due to social, political, engineering, and environmental factors. The system should also identify data gaps needed for a complete risk profile. This addresses a real-world need: dam removal decisions are complex, multi-stakeholder processes, and a transparent, data-driven risk framework could inform policy and planning.

### Scope & Key Decisions
- **Geographic scope**: Columbia River Basin (includes most of WA, OR, ID, plus portions of MT, WY, NV, UT, and British Columbia)
- **Training/validation data**: American Rivers Dam Removal Database (available)
- **Weighting approach**: Equal initial weights across categories, with configurable parameters for sensitivity analysis

---

## Phase 1: Risk Framework Definition

**Goal**: Define measurable risk factors within each category

### Risk Categories and Proposed Indicators

| Category | Risk Factors | Data Source |
|----------|--------------|-------------|
| **Engineering** | Dam age, structural condition, capacity factor, operational issues | EIA 860, FERC license docs, USACE data |
| **Environmental** | Fish passage impact, sediment disruption, water quality degradation, ecosystem fragmentation | USGS dam impact metrics, WA Ecology EIM |
| **Social** | Local population affected, economic dependency, recreational value, tribal interests | Census ACS, tribal consultation records |
| **Political** | License expiration timeline, policy alignment, treaty obligations | FERC licensing, Columbia River Treaty |
| **Economic** | Generation capacity, grid importance, replacement cost, O&M costs | EIA 923/930, BPA data, PUDL |

**Deliverable**: Risk factor taxonomy document with clear definitions and scoring criteria

---

## Phase 2: Data Acquisition Pipeline

**Goal**: Automate retrieval of all relevant datasets

### Data Sources to Integrate

1. **Dam Inventory (Master List)**
   - National Inventory of Dams (NID) - baseline dam attributes
   - EIA 860 - generator-level data for hydropower facilities

2. **Streamflow & Hydrology**
   - USGS Water Data API (`api.waterdata.usgs.gov`) - daily/instantaneous flow
   - Link gages to dams via HUC codes or proximity

3. **Energy/Operations**
   - EIA 923 - monthly generation data
   - EIA 930 - hourly grid operations
   - BPA transmission data
   - PUDL for unified access

4. **Environmental Impact**
   - USGS dam disturbance metrics (1800-2018)
   - WA Ecology EIM - water quality data

5. **Social/Demographic**
   - Census ACS 5-year estimates via API
   - WA state socioeconomic data

6. **Regulatory/Political**
   - FERC eLibrary for license documents (may require manual extraction or NLP)
   - Columbia River Treaty context documents

**Implementation**: Python scripts using `requests`, `pandas`, existing APIs. Store raw data in structured format (Parquet/GeoPackage).

---

## Phase 3: Data Integration & Spatial Linkage

**Goal**: Create unified dam-centric dataset

### Approach
1. **Establish dam as primary key** - Use NID ID or EIA plant code
2. **Spatial joins**:
   - Link dams to HUC watersheds
   - Associate USGS gages (upstream/downstream)
   - Join Census tract demographics
3. **Temporal alignment** - Standardize to common time periods where applicable
4. **Schema design** - Relational database or wide-format table

**Tools**: GeoPandas for spatial operations, DuckDB or SQLite for data management

---

## Phase 4: Risk Quantification (Statistical Methods)

**Goal**: Score each risk factor using defensible statistical methods

### Proposed Methods

1. **Normalization**: Min-max or z-score scaling for comparable metrics

2. **Multi-Criteria Decision Analysis (MCDA)**
   - Weight factors by category (equal weighting initially)
   - **Weights stored as config parameters** - easily adjustable for sensitivity analysis
   - Weighted sum scoring for composite risk

3. **Classification Model (if sufficient training data)**
   - Use American Rivers Dam Removal Database as labels (dams actually removed)
   - Train logistic regression or random forest to predict removal probability
   - Features: all quantified risk factors
   - Validate with cross-validation and holdout set

4. **Bayesian Uncertainty**
   - Incorporate data quality/completeness as uncertainty
   - Report risk as distributions, not point estimates

5. **Risk Thresholds**
   - Percentile-based tiers (High/Medium/Low)
   - Or receiver operating characteristic (ROC) analysis if using classification

### Validation Against Trusted Data
- **Ground truth**: American Rivers Dam Removal Database (historical removals)
- **Test**: Do high-risk scores correlate with actual removal decisions?
- **Baseline**: Compare against expert rankings if available

---

## Phase 5: Data Gap Analysis

**Goal**: Identify missing data needed for complete risk profiles

### Approach
1. **Completeness scoring** - For each dam, calculate % of risk factors with data
2. **Imputation feasibility** - Flag which gaps can be reasonably imputed vs require collection
3. **Priority ranking** - Which data gaps most affect risk uncertainty?

**Output**: Per-dam data completeness report + recommendations for data collection priorities

---

## Phase 6: Visualization & Delivery

**Goal**: Communicate results effectively

### Components
1. **Interactive map** - Dams colored by risk tier, click for details (Leaflet/Mapbox)
2. **Risk breakdown charts** - Radar plots showing contribution of each category
3. **Data completeness dashboard** - Highlight gaps
4. **Methodology documentation** - Transparent scoring logic

**Implementation**: Quarto document with embedded Plotly/Leaflet, or standalone web app (given team has web dev expertise)

---

## Technical Stack Summary

| Component | Technology |
|-----------|------------|
| Data retrieval | Python (requests, dataretrieval for USGS) |
| Data processing | pandas, geopandas, DuckDB |
| Spatial analysis | geopandas, shapely |
| Statistical modeling | scikit-learn, statsmodels, PyMC (Bayesian) |
| Visualization | plotly, folium/leaflet, matplotlib |
| Documentation | Quarto |
| Version control | Git |

---

## Verification & Testing

1. **Unit tests** for data retrieval functions
2. **Spot-check** spatial joins against known dam-watershed relationships
3. **Sanity check** risk scores against manually reviewed cases
4. **Validate** classification model against held-out dam removals
5. **Sensitivity analysis** on MCDA weights

---

## Project Timeline (Suggested)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Risk framework + Data acquisition setup | Taxonomy doc, data scripts |
| 2 | Data integration & spatial linkage | Unified dam dataset |
| 3 | Risk quantification & modeling | Scoring system, trained classifier |
| 4 | Gap analysis + Visualization | Final dashboard/report |

---

## Notes

- Columbia River Basin boundary available from USGS Watershed Boundary Dataset (WBD)
- Cross-border (Canada) data may require additional sources (BC Hydro, Environment Canada)
- Political factors should draw only from official government sources (FERC, State Dept, treaty docs) to avoid partisan bias
