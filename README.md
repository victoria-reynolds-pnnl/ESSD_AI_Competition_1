# Week 2 Submission - CONUS Fire Occurrence Classification v1

**Team:** The Philadelphia Eagles  
**Challenge:** ESSD AI Challenge  
**Week:** 2 - Data Preparation and Documentation  
**Date:** 2026-04-02

---

## ML Algorithms

**Selected Algorithms:**

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier (XGBoost)**
3. **Logistic Regression with L2 Regularization**

**Justification:**

Random Forest and XGBoost are well-suited for this binary classification task with severe class imbalance (~0.25% positive rate). Both algorithms handle non-linear relationships effectively, provide feature importance metrics, and support class weighting to address imbalance. Random Forest offers robustness through ensemble averaging, while XGBoost provides superior performance through gradient boosting and built-in regularization.

Logistic Regression serves as a strong baseline for comparison. Its interpretability allows us to understand feature contributions directly, and L2 regularization helps prevent overfitting despite the high-dimensional feature space (28 predictors in cleaned dataset). All three algorithms support stratified sampling and can be tuned to optimize precision-recall tradeoffs critical for rare event prediction.

---

## Data Dictionary

### Original Dataset (Raw Merged Table)

The original dataset contains 40.7M rows spanning 2001-2020 (all 12 months) with raw features from four ComExDBM CONUS products: Fire, Heatwave, Drought, and Meteorology.

| Field Name  | Description                                                | Data Type      | Example                                |
| ----------- | ---------------------------------------------------------- | -------------- | -------------------------------------- |
| date        | Daily timestamp for the grid cell observation              | datetime64[ns] | 2001-05-01T00:00:00                    |
| lat         | Latitude of the CONUS grid cell center                     | float32        | 34.7500                                |
| lon         | Longitude of the CONUS grid cell center                    | float32        | -118.2500                              |
| FHS_c9      | Highest-confidence fire hotspot count (target source)      | float32        | 2                                      |
| FHS_c8c9    | Combined confidence fire hotspot count                     | float32        | 3                                      |
| maxFRP_c9   | Same-day fire radiative power proxy (c9)                   | float32        | 8.7                                    |
| maxFRP_c8c9 | Same-day fire radiative power proxy (c8c9)                 | float32        | 9.4                                    |
| BA_km2      | Same-day burned area                                       | float32        | 0.6                                    |
| tmax        | Daily maximum near-surface air temperature                 | float32        | 33.2                                   |
| tmin        | Daily minimum near-surface air temperature                 | float32        | 17.9                                   |
| HI02        | Heatwave indicator (threshold 02)                          | float32        | 0                                      |
| HI04        | Heatwave indicator (threshold 04)                          | float32        | 0                                      |
| HI05        | Heatwave indicator (threshold 05)                          | float32        | 0                                      |
| HI06        | Heatwave indicator (threshold 06)                          | float32        | 0                                      |
| HI09        | Heatwave indicator (threshold 09)                          | float32        | 0                                      |
| HI10        | Heatwave indicator (threshold 10)                          | float32        | 0                                      |
| pdsi        | Palmer Drought Severity Index                              | float32        | -3.1                                   |
| spi14d      | 14-day Standardized Precipitation Index                    | float32        | -0.8                                   |
| spi30d      | 30-day Standardized Precipitation Index                    | float32        | -1.2                                   |
| spi90d      | 90-day Standardized Precipitation Index                    | float32        | -1.9                                   |
| spei14d     | 14-day Standardized Precipitation Evapotranspiration Index | float32        | -0.7                                   |
| spei30d     | 30-day Standardized Precipitation Evapotranspiration Index | float32        | -1.0                                   |
| spei90d     | 90-day Standardized Precipitation Evapotranspiration Index | float32        | -1.5                                   |
| air_sfc     | Daily near-surface air temperature                         | float32        | 297.4                                  |
| air_apcp    | Daily accumulated precipitation                            | float32        | 1.7                                    |
| soilm       | Daily soil moisture proxy                                  | float32        | 0.27                                   |
| rhum_2m     | Daily relative humidity at 2 meters                        | float32        | 42.5                                   |
| uwnd        | Daily zonal wind component                                 | float32        | 3.2                                    |
| vwnd        | Daily meridional wind component                            | float32        | -1.1                                   |
| LAI         | Leaf Area Index                                            | float32        | 2.8                                    |
| NDVI        | Normalized Difference Vegetation Index                     | float32        | 0.41                                   |
| FWI         | Fire Weather Index                                         | float32        | 17.3                                   |
| cell_id     | Deterministic spatial identifier (lat\|lon)                | string         | lat=34.7500\|lon=-118.2500             |
| row_id      | Deterministic row identifier (date\|cell_id)               | string         | 2001-05-01\|lat=34.7500\|lon=-118.2500 |

**Statistics:**

- **Total Rows:** 40,704,000 (2,035,200 per year × 20 years)
- **Spatial Coverage:** CONUS grid (512 × 256 cells = 131,072 cells)
- **Temporal Coverage:** 2001-2020, all 12 months
- **Missing Values:** Minimal (<0.1% in most fields)

---

### Cleaned Dataset (Engineered Features)

The cleaned dataset contains 20.5M rows filtered to May-October fire season with 10 engineered features added. All temporal lag features use past-only windows to prevent leakage.

| Field Name           | Description                                                | Data Type      | Example                                |
| -------------------- | ---------------------------------------------------------- | -------------- | -------------------------------------- |
| date                 | Daily timestamp for the grid cell observation              | datetime64[ns] | 2001-05-01T00:00:00                    |
| lat                  | Latitude of the CONUS grid cell center                     | float32        | 34.7500                                |
| lon                  | Longitude of the CONUS grid cell center                    | float32        | -118.2500                              |
| tmax                 | Daily maximum near-surface air temperature                 | float32        | 33.2                                   |
| tmin                 | Daily minimum near-surface air temperature                 | float32        | 17.9                                   |
| pdsi                 | Palmer Drought Severity Index                              | float32        | -3.1                                   |
| spi14d               | 14-day Standardized Precipitation Index                    | float32        | -0.8                                   |
| spi30d               | 30-day Standardized Precipitation Index                    | float32        | -1.2                                   |
| spi90d               | 90-day Standardized Precipitation Index                    | float32        | -1.9                                   |
| spei14d              | 14-day Standardized Precipitation Evapotranspiration Index | float32        | -0.7                                   |
| spei30d              | 30-day Standardized Precipitation Evapotranspiration Index | float32        | -1.0                                   |
| spei90d              | 90-day Standardized Precipitation Evapotranspiration Index | float32        | -1.5                                   |
| air_sfc              | Daily near-surface air temperature                         | float32        | 297.4                                  |
| air_apcp             | Daily accumulated precipitation                            | float32        | 1.7                                    |
| soilm                | Daily soil moisture proxy                                  | float32        | 0.27                                   |
| rhum_2m              | Daily relative humidity at 2 meters                        | float32        | 42.5                                   |
| uwnd                 | Daily zonal wind component                                 | float32        | 3.2                                    |
| vwnd                 | Daily meridional wind component                            | float32        | -1.1                                   |
| LAI                  | Leaf Area Index                                            | float32        | 2.8                                    |
| NDVI                 | Normalized Difference Vegetation Index                     | float32        | 0.41                                   |
| FWI                  | Fire Weather Index                                         | float32        | 17.3                                   |
| cell_id              | Deterministic spatial identifier (lat\|lon)                | string         | lat=34.7500\|lon=-118.2500             |
| row_id               | Deterministic row identifier (date\|cell_id)               | string         | 2001-05-01\|lat=34.7500\|lon=-118.2500 |
| year                 | Calendar year derived from date                            | int16          | 2001                                   |
| month                | Calendar month derived from date                           | int8           | 5                                      |
| split_bucket         | Fixed benchmark split assignment                           | string         | train                                  |
| **fire_occurrence**  | **Binary target: 1 when FHS_c9 > 0, else 0**               | **int8**       | **1**                                  |
| **wind_speed**       | **Wind speed magnitude: sqrt(uwnd² + vwnd²)**              | **float32**    | **3.38**                               |
| **temp_range**       | **Daily temperature range: tmax - tmin**                   | **float32**    | **15.3**                               |
| **pdsi_severe_flag** | **Severe drought flag: 1 when pdsi ≤ -4, else 0**          | **int8**       | **1**                                  |
| **apcp_7d_sum**      | **Trailing 7-day precipitation sum (past-only)**           | **float32**    | **6.9**                                |
| **fwi_7d_mean**      | **Trailing 7-day Fire Weather Index mean (past-only)**     | **float32**    | **14.2**                               |
| **fhs_c9_lag1**      | **Previous-day fire hotspot count (past-only)**            | **float32**    | **0.0**                                |
| **fhs_c9_lag7_sum**  | **Trailing 7-day fire hotspot sum (past-only)**            | **float32**    | **2.0**                                |
| **days_since_fire**  | **Days since last fire in cell (past-only)**               | **float32**    | **12.0**                               |
| **doy_sin**          | **Sine encoding of day-of-year seasonality**               | **float32**    | **0.8660**                             |
| **doy_cos**          | **Cosine encoding of day-of-year seasonality**             | **float32**    | **-0.5000**                            |

**Engineered Features (10 total, highlighted in bold above):**

1. **wind_speed**: Wind magnitude derived from vector components, relevant for fire spread modeling
2. **temp_range**: Diurnal temperature variation, indicator of atmospheric instability
3. **pdsi_severe_flag**: Binary indicator for extreme drought conditions (PDSI ≤ -4)
4. **apcp_7d_sum**: Recent precipitation accumulation, affects fuel moisture
5. **fwi_7d_mean**: Recent fire weather conditions, smoothed for stability
6. **fhs_c9_lag1**: Previous-day fire activity, captures short-term persistence
7. **fhs_c9_lag7_sum**: Recent fire activity in cell, captures weekly patterns
8. **days_since_fire**: Time since last fire, captures fuel recovery dynamics
9. **doy_sin**: Seasonal encoding (sine component), captures annual fire season patterns
10. **doy_cos**: Seasonal encoding (cosine component), complements sine for full cycle

**Statistics:**

- **Total Rows:** 20,490,240 (1,024,512 per year × 20 years)
- **Temporal Coverage:** 2001-2020, May-October only (fire season)
- **Target Distribution:** ~0.25% positive (fire_occurrence = 1), severe class imbalance
- **Split Strategy:**
  - Train: 2001-2016 (16 years, 16,392,192 rows)
  - Validation: 2017-2018 (2 years, 2,049,024 rows)
  - Test: 2019-2020 (2 years, 2,049,024 rows)

---

## Data Preparation Steps

### Overview

Our data preparation pipeline transforms 20 years of raw ComExDBM CONUS NetCDF files into ML-ready datasets for binary fire occurrence classification. The pipeline consists of three main stages: extraction, feature engineering, and export. All processing is reproducible through Python scripts with comprehensive logging and validation built in throughout.

### 1. Data Extraction and Merging

We started by extracting four NetCDF products per year from the ComExDBM CONUS dataset: Fire, Heatwave, Drought, and Meteorology. These products were merged on their common (lat, lon, time) keys to create unified daily grid cell observations spanning the entire CONUS region. During this process, we validated that all keys were unique and handled any missing values using forward-fill to maintain temporal continuity. We also added spatial identifiers (cell_id) and row identifiers (row_id) for deterministic tracking throughout the pipeline. The output from this stage was 40.7M rows in parquet format, representing 2,035,200 rows per year across all 20 years.

AI Role - Claude Sonnet 4.5 assisted in designing the merge strategy and key validation logic, and generated the code for NetCDF loading with xarray and pandas merge operations found in [`Scripts/extract_conus_fire_table.py`](Scripts/extract_conus_fire_table.py) lines 110-145.

### 2. Data Cleaning and Filtering

Once we had the merged dataset, we filtered it down to just the May-October months to focus on the primary fire season, which reduced the dataset by roughly 50%. We removed rows with missing values in key predictors, though this resulted in minimal data loss (less than 0.1%). To improve memory efficiency, we optimized numeric data types by converting float64 to float32 and int64 to int16 or int8 where appropriate. We also validated data ranges and flagged potential outliers, though none were found that required removal.

AI Role - Claude Sonnet 4.5 suggested the month filtering strategy based on fire season patterns and generated the dtype optimization logic to reduce the memory footprint. This cleaning code can be found in [`Scripts/engineer_fire_features.py`](Scripts/engineer_fire_features.py) lines 85-120.

### 3. Feature Engineering

Feature engineering was a critical step where we created 10 new features using domain knowledge and temporal patterns. All lag features use past-only windows (shifted by at least 1 day) to prevent any data leakage, and rolling aggregations were computed per cell_id to maintain spatial independence. We used seasonal encoding with sine and cosine functions to capture annual fire season patterns without making linear assumptions. Rows with NaN values in engineered features were dropped, which primarily affected the first 7 days per cell where incomplete lag windows existed.

The engineered features include wind_speed (calculated from wind components since fire spread is driven by wind magnitude), temp_range (diurnal temperature variation indicating atmospheric instability), pdsi_severe_flag (a binary indicator for extreme drought when PDSI ≤ -4), apcp_7d_sum (recent precipitation affecting fuel moisture), fwi_7d_mean (smoothed fire weather index to reduce daily noise), fhs_c9_lag1 (previous-day fire activity capturing short-term persistence), fhs_c9_lag7_sum (weekly fire activity capturing multi-day events), days_since_fire (time since last fire for fuel recovery dynamics), and doy_sin/doy_cos (circular encoding of seasonality to avoid discontinuity at year boundaries).

AI Role - Claude Sonnet 4.5 proposed the initial feature set based on fire science literature and generated the rolling window logic with proper shift to prevent leakage, as well as the seasonal encoding formulas. This feature engineering code is in [`Scripts/engineer_fire_features.py`](Scripts/engineer_fire_features.py) lines 125-210.

### 4. Target Creation

We created a binary target variable called fire_occurrence that equals 1 when FHS_c9 (highest-confidence fire hotspot count from MODIS/VIIRS satellite data) is greater than 0, and 0 otherwise. We chose a threshold of 0 to capture any fire activity using an inclusive definition, which results in a severe class imbalance with only about 0.25% positive cases.

AI Role - Claude Sonnet 4.5 recommended this binary threshold based on the dataset documentation and generated the target creation logic found in [`Scripts/engineer_fire_features.py`](Scripts/engineer_fire_features.py) lines 215-220.

### 5. Split Strategy

To respect the time-series nature of the data, we used a temporal split rather than random sampling. The training set includes 2001-2016 (16 years, 80% of data), the validation set covers 2017-2018 (2 years, 10% of data), and the test set spans 2019-2020 (2 years, 10% of data). We added the split assignment as a split_bucket column to ensure reproducibility across different runs.

AI Role - Claude Sonnet 4.5 suggested this temporal split approach to avoid leakage that could occur with random splits and generated the split assignment logic in [`Scripts/_fire_pipeline_common.py`](Scripts/_fire_pipeline_common.py) lines 630-640.

### 6. Export and Documentation

For the final deliverables, we exported both the original and cleaned datasets in CSV and JSON formats, compressing them with gzip to reduce file sizes to approximately 5.9 GB total. We generated a comprehensive data dictionary with field metadata including names, descriptions, data types, examples, roles, and formulas where applicable. We also created a QC summary documenting row counts, missing values, and target distribution. To handle the large dataset size of over 40M rows, we streamed the data year-by-year to avoid memory issues.

AI Role - Claude Sonnet 4.5 designed the streaming export strategy and generated the data dictionary schema and QC summary structure found in [`Scripts/export_week2_artifacts.py`](Scripts/export_week2_artifacts.py) lines 85-187.

### 7. Normalization

We deliberately chose not to apply explicit normalization at this stage, deferring it to the modeling phase instead. This decision was based on the fact that tree-based algorithms like Random Forest and XGBoost are scale-invariant and don't require normalized inputs. Logistic Regression will require standardization (z-score normalization) when we implement it in Week 3, but for now we've preserved all raw values in the cleaned dataset to maintain flexibility.

AI Role - Claude Sonnet 4.5 recommended this approach of deferring normalization to the modeling phase and provided the rationale based on algorithm requirements.

---

## AI Usage Disclosure

All Python scripts in the `Scripts/` directory were developed with assistance from Claude Sonnet 4.5 (Anthropic). Specific AI contributions are documented inline with comments like:

```python
# AI-generated: Claude Sonnet 4.5 - [description of contribution]
```

Key AI contributions include:

- NetCDF loading and merging logic
- Rolling window feature engineering with proper lag shifts
- Streaming export strategy for large datasets
- Data dictionary schema design
- QC validation checks

All AI-generated code was reviewed, tested, and validated by the team before inclusion in the pipeline.

---
