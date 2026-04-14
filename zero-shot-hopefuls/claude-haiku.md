User: List the types of data recorded in the ComEcDBM_CONUS Input files.

GitHub Copilot: Read [](file:///c%3A/Users/sasl988/OneDrive%20-%20PNNL/Documents/ESSD%20AI%20Competition%202026/ESSD_AI_Competition_1/ComExDBM_CONUS/01_Input/README.md#1-1), lines 1 to 100

Based on the README.md file, the ComExDBM_CONUS Input files contain the following types of data:

## **1. Fire Data**

- **MODIS Thermal Anomalies (MOD14A1 v6)**: Fire mask, pixel quality indicators, maximum fire radiative power (MaxFRP), fire hotspots (FHS) at ~1 km daily resolution
- **FIRED (Fire Events Delineation)**: Daily burned area derived from MODIS MCD64A1 at 500m monthly resolution for CONUS (2001-2020)

## **2. Temperature Data**

- **CPC Global Unified Temperature**: Gauge-based daily gridded temperature anomaly data at 0.5°×0.5° (~55km×55km) spatial resolution

## **3. Drought Data**

- **GridMET Drought Indices**: Standardized Precipitation Index (SPI), Standardized Precipitation Evapotranspiration Index (SPEI), Palmer Drought Severity Index (PDSI) at 4-km resolution and 5-day temporal resolution

## **4. Meteorological and Surface Variables**

**From NARR** (3-hourly, 32-km grid spacing):

- Surface temperature, precipitation (P3h), soil moisture (SM), relative humidity (RH), specific humidity (Qv), latent heat flux (LHF), sensible heat flux (SH)

**From MERRA-2** (3-hourly, ~0.5° × 0.625° resolution):

- Specific humidity, relative humidity, U-wind and V-wind, black carbon and organic carbon aerosols (BC+OC)

**Vegetation Data** (daily, 0.05° × 0.05°):

- Leaf Area Index (LAI), Normalized Difference Vegetation Index (NDVI)

**Fire Weather Index** (0.5° × 0.625°, daily):

- Fire Weather Index (FWI) from Global Fire Weather Database (GFWED)

User: How is the data organized and structured?

GitHub Copilot: Read [](file:///c%3A/Users/sasl988/OneDrive%20-%20PNNL/Documents/ESSD%20AI%20Competition%202026/ESSD_AI_Competition_1/ComExDBM_CONUS/01_Input/README.md#100-100), lines 100 to 120

Read [](file:///c%3A/Users/sasl988/OneDrive%20-%20PNNL/Documents/ESSD%20AI%20Competition%202026/ESSD_AI_Competition_1/ComExDBM_CONUS/01_Input/README.md)

Based on the README.md file, the data in the ComExDBM_CONUS Input files is organized as follows:

## **Directory Structure**

```
01_Input/
├── README.md
└── CONUS4326/
    ├── contig_us.dbf    (shapefile attribute database)
    ├── contig_us.prj    (projection information)
    ├── contig_us.shp    (shapefile geometry)
    └── contig_us.shx    (shapefile index)
```

## **Data Organization by Type**

The input data is conceptually organized into **four main categories**:

1. **Fire Data**: MODIS thermal anomalies (MOD14A1) and FIRED burned area products
2. **Temperature Data**: CPC Global Unified daily gridded temperature anomalies
3. **Drought Data**: GridMET drought indices (SPI, SPEI, PDSI)
4. **Meteorological & Surface Variables**: Data from NARR, MERRA-2, vegetation indices, and fire weather indices

## **Spatial Reference**

The CONUS4326 folder contains **shapefiles** that define the contiguous U.S. boundary, which serves as the spatial reference frame for organizing the gridded data products. The files follow the standard shapefile format (.shp, .shx, .dbf, .prj components).

**Note**: The README describes the _sources_ and _characteristics_ of the input data (spatial resolution, temporal frequency, variables) but doesn't explicitly specify the file formats or directory structure for where the actual input data files are stored within the 01_Input directory. This information may be in processing scripts or additional documentation.

User: How would you use ML and this data set to estimate the probability of compound hazard (co‑occurring heatwave and drought, or wildfire during those conditions) 1–2 weeks ahead and identify regional drivers.

GitHub Copilot: ## ML Approach for Forecasting Compound Hazards 1-2 Weeks Ahead

Here's a comprehensive methodology using the ComExDBM dataset:

### **1. Problem Definition & Data Preparation**

**Define Compound Hazard Events:**

- Heatwave + Drought: Temperature anomalies above 75th percentile + SPI/SPEI below -1.0
- Wildfire Risk: Fire occurrence during concurrent heatwave+drought conditions
- Binary classification: Event (1) vs. No Event (0) for 7-14 day forecast window

**Aggregate Data to Consistent Temporal Resolution:**

- Resample all inputs to daily resolution (NARR/MERRA-2: 3-hourly → daily mean/max)
- Spatially aggregate to a consistent grid (e.g., 0.5° to match fire weather index or coarsen to 4-km)
- Create CONUS regional subsets based on the CONUS4326 shapefile

---

### **2. Feature Engineering (1-2 Week Lookback)**

**Temporal Features (7-14 day lagged windows):**

- Temperature anomalies (daily, 7-day rolling mean/std)
- Drought indices (SPI, SPEI, PDSI at multiple time scales)
- Soil moisture deciles/anomalies
- Precipitation anomalies
- Relative/specific humidity
- Wind speed (U/V components)
- Heat fluxes (sensible, latent)
- Vegetation stress (LAI, NDVI trends—declining LAI + heat = higher fire risk)
- Fire weather index (FWI)

**Spatial/Static Features:**

- Latitude, longitude, elevation
- Climatological mean/std for each grid cell
- Ecosystem type (from NDVI patterns)
- Distance to coastline

**Compound Indices:**

- Evaporative stress (potential evapotranspiration - precipitation)
- Compound dryness (precipitation deficit × low soil moisture)
- Heat-drought intensity (temperature anomaly × drought index)

---

### **3. Model Architecture**

**Option A: Ensemble Methods (Interpretable, Faster)**

- **XGBoost/LightGBM** with spatial stratification
  - Train separate models for different regions (West, Great Plains, Southeast, Northeast)
  - Use temporal cross-validation: Train on years 2001-2015, validate on 2016-2020
  - Output: Probability scores for each grid cell

**Option B: Deep Learning (Better Temporal Patterns)**

- **LSTM/ConvLSTM** for spatiotemporal sequences
  - Input: 14-day sequences of multivariate gridded data
  - Output: Probability field for days +7 to +14
  - Capture non-linear temporal dependencies

**Option C: Hybrid**

- XGBoost for feature-level predictions + LSTM for spatial-temporal refinement

---

### **4. Lead Time & Validation Strategy**

**Temporal Cross-Validation (Avoid Data Leakage):**

```
Train: 2001-2015 → Validate: 2016-2017 → Test: 2018-2020
Train: 2001-2016 → Validate: 2017-2018 → Test: 2019-2020
```

**Forecast Skill Metrics:**

- ROC-AUC (discrimination ability)
- Brier Score (calibration)
- Lead time dependent skill (how skill decays from day +1 to +14)

---

### **5. Identifying Regional Drivers**

**Feature Importance Analysis:**

- **SHAP values**: Decompose each prediction into driver contributions
  - Example: "Heatwave in Southwest driven 60% by soil moisture deficit, 25% by SST anomalies"
- **Permutation importance**: Identify which variables, when shuffled, most degrade prediction

**Spatially Stratified Analysis:**

- Train separate models for regions: Southwest, Great Plains, Southeast, Pacific Northwest
- Compare feature rankings across regions
  - Southwest: Soil moisture + precipitation deficit dominant
  - Great Plains: Jet stream positioning + soil moisture
  - Southeast: Sea surface temperatures + soil moisture

**Temporal Driver Analysis:**

- Which 7-day window most predictive? (Days -14 to -7, or -7 to 0?)
- Lag correlation analysis between predictors and events

---

### **6. Implementation Workflow**

```python
# Pseudocode workflow
1. Load & preprocess all data sources
2. Define compound hazard events (spatiotemporal)
3. Create 14-day lagged feature arrays
4. Stratify by CONUS region (using shapefile)
5. For each region:
   - Train XGBoost/LSTM with temporal CV
   - Generate 7-14 day probability forecasts
   - Compute SHAP importance scores
   - Identify top 5 drivers
6. Map spatial driver variations
7. Validate: skill vs. lead time curves
```

---

### **7. Expected Outcomes**

**Forecasts:** Probabilistic maps showing compound hazard likelihood 1-2 weeks ahead at grid-cell level

**Regional Drivers (Example):**

- **Western US**: Recent precipitation deficit (SPI < -1.5), high temperature anomalies
- **Great Plains**: Jet stream southward displacement, soil moisture decline
- **Southeast**: High relative humidity retention from Gulf SST anomalies, heat stress
- **Fire risk layers**: FWI + NDVI stress + antecedent drought

---

### **8. Data Recommendations from ComExDBM**

Prioritize these variables:

1. **Temperature anomalies** (CPC) + HW indices
2. **Drought indices** (SPI, SPEI from gridMET)
3. **Soil moisture** (NARR)
4. **Vegetation stress** (NDVI, LAI trends)
5. **Fire weather index** (FWI) for fire compound events
6. **Upward shortwave radiation** (MERRA-2) as heat stress proxy

This integrated approach leverages the temporal and spatial structure of ComExDBM to not only forecast compound hazards but also diagnose what regional factors are driving the risk.
