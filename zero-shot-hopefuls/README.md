# Zero Shot Hopefuls - Week 2: Data Preparation and Documentation

## ML Algorithms

**Primary: LightGBM (Gradient Boosted Decision Trees)**

LightGBM excels on tabular data with mixed feature types and class imbalance—compound heatwave+drought events occur in only ~4% of weekly observations. Its leaf-wise tree growth provides strong predictive performance while built-in feature importance and SHAP compatibility support the interpretability requirements of Week 3.

**Baseline: Logistic Regression (L2 Regularization)**

Logistic Regression establishes a performance floor and provides a fully interpretable linear reference point against which to measure gains from the gradient-boosted model, enabling a meaningful comparison of model complexity versus performance.

## Data Dictionary for Data Source 1: ComExDBM Compound Hazard Weekly Dataset

### Identifiers

| Field Name | Description                                                        | Data Type | Example |
| ---------- | ------------------------------------------------------------------ | --------- | ------- |
| lat        | Latitude of grid cell center (WGS84, 0.5-degree resolution)        | float64   | 37.25   |
| lon        | Longitude of grid cell center (WGS84, 0.5-degree resolution)       | float64   | -105.75 |
| year       | ISO calendar year of the observation week                          | int       | 2015    |
| week       | ISO calendar week number (1-53), filtered to fire season (Apr-Oct) | int       | 28      |

### Heatwave Variables (Source: CPC Global Unified Temperature via ComExDBM)

| Field Name | Description                                                                                  | Data Type | Example |
| ---------- | -------------------------------------------------------------------------------------------- | --------- | ------- |
| tmax       | Weekly mean of daily maximum temperature (deg C)                                             | float64   | 35.2    |
| tmin       | Weekly mean of daily minimum temperature (deg C)                                             | float64   | 18.7    |
| tmean      | Weekly mean of daily mean temperature (deg C)                                                | float64   | 27.0    |
| HI02       | Heatwave indicator: 1 if any day in week had tmean > 95th percentile for 3+ consecutive days | int       | 1       |
| HI05       | Heatwave indicator: 1 if any day in week had tmax > 95th percentile for 3+ consecutive days  | int       | 0       |

### Fire Variables (Source: MODIS MOD14A1/MCD64A1 via ComExDBM)

| Field Name | Description                                                                   | Data Type | Example |
| ---------- | ----------------------------------------------------------------------------- | --------- | ------- |
| FHS_c9     | Weekly total count of highest-confidence fire hotspot pixels within grid cell | float64   | 5.0     |
| maxFRP_c9  | Weekly mean of daily max fire radiative power (MW) at highest confidence      | float64   | 42.3    |
| BA_km2     | Weekly total fire burned area (km2) within grid cell                          | float64   | 1.2     |

### Drought Variables (Source: gridMET via ComExDBM)

| Field Name    | Description                                                                                | Data Type | Example |
| ------------- | ------------------------------------------------------------------------------------------ | --------- | ------- |
| pdsi          | Weekly mean Palmer Drought Severity Index (-5 extreme drought to +5 extreme wet)           | float64   | -2.5    |
| pdsi_category | Most frequent PDSI category during week (0=extreme drought, 5=near normal, 10=extreme wet) | int       | 3       |
| spi90d        | Weekly mean Standardized Precipitation Index at 90-day timescale                           | float64   | -1.2    |
| spei90d       | Weekly mean Standardized Precipitation-Evapotranspiration Index at 90-day timescale        | float64   | -0.8    |

### Meteorological Variables (Sources: NARR, MERRA-2, NOAA CDR, GFWED)

| Field Name | Description                                                     | Data Type | Example |
| ---------- | --------------------------------------------------------------- | --------- | ------- |
| air_sfc    | Weekly mean surface air temperature (K)                         | float64   | 300.5   |
| apcp       | Weekly mean daily accumulated precipitation (kg/m2)             | float64   | 2.1     |
| soilm      | Weekly mean soil moisture (kg/m2)                               | float64   | 320.0   |
| lhtfl      | Weekly mean latent heat flux (W/m2)                             | float64   | 85.3    |
| shtfl      | Weekly mean sensible heat flux (W/m2)                           | float64   | 45.6    |
| rhum_2m    | Weekly mean relative humidity at 2m (%)                         | float64   | 42.5    |
| BCOC       | Weekly mean black carbon + organic carbon aerosol concentration | float64   | 0.0025  |
| LAI        | Weekly mean Leaf Area Index (m2/m2)                             | float64   | 2.1     |
| NDVI       | Weekly mean Normalized Difference Vegetation Index (-1 to 1)    | float64   | 0.45    |
| FWI        | Weekly mean Fire Weather Index                                  | float64   | 18.5    |

### Engineered Features

| Field Name           | Description                                                                      | Data Type | Example |
| -------------------- | -------------------------------------------------------------------------------- | --------- | ------- |
| compound_hw_drought  | Binary: heatwave (HI02=1) AND drought (pdsi_category<=3) co-occurred during week | int       | 1       |
| fire_during_compound | Binary: fire activity during compound heatwave+drought conditions                | int       | 0       |
| wind_speed           | Weekly mean surface wind speed (m/s), derived: sqrt(uwnd^2 + vwnd^2)             | float64   | 4.2     |
| seasonal_sin         | Sine of cyclical day-of-year encoding: sin(2*pi*DOY/365)                         | float64   | 0.97    |
| seasonal_cos         | Cosine of cyclical day-of-year encoding: cos(2*pi*DOY/365)                       | float64   | -0.25   |
| tmean_7d_avg         | 7-day rolling average of daily mean temperature at end of week (deg C)           | float64   | 28.3    |
| drought_trend_14d    | 14-day change in PDSI; negative = worsening drought                              | float64   | -0.15   |
| fwi_7d_max           | Maximum Fire Weather Index over preceding 7 days at end of week                  | float64   | 25.8    |

### Target Variables

| Field Name          | Description                                                            | Data Type | Example |
| ------------------- | ---------------------------------------------------------------------- | --------- | ------- |
| target_compound_1wk | Forecast target: compound_hw_drought value for NEXT week (1-week lead) | int       | 0       |
| target_compound_2wk | Forecast target: compound_hw_drought value 2 weeks ahead               | int       | 0       |

## Data Preparation Steps

Data preparation began by loading 20 years (2001-2020) of daily gridded observations from the ComExDBM NetCDF files across four data categories—heatwave labels and temperature, fire hotspots and burned area, drought indices, and meteorological variables—and merging them on the shared 0.5-degree spatial grid and daily time dimension using xarray. Cleaning steps included: dropping ocean/non-CONUS grid cells (identified by NaN temperature), zero-filling fire variables where NaN represents absence of fire activity, forward- and back-filling satellite-derived variables (LAI, NDVI, FWI) that have intermittent coverage gaps due to cloud cover and sensor availability, forward-filling meteorological variables with small gap fractions (<1%), flagging physically impossible values (negative precipitation, temperatures outside [-60, 60] C, negative fire radiative power), enforcing correct data types for binary heatwave indicators and integer drought categories, and checking for duplicate entries. Feature engineering produced seven new variables: a binary compound heatwave+drought indicator (HI02=1 AND pdsi_category<=3), a triple-compound fire-during-drought flag, surface wind speed derived from U/V wind components, cyclical seasonal encoding (sin/cos of day-of-year), a 7-day rolling mean temperature capturing recent warming trends, a 14-day PDSI change capturing drought trajectory, and a 7-day rolling maximum FWI capturing persistent fire-weather conditions. Rolling features were computed on full-year daily data before filtering to the April-October fire season to ensure rolling windows at the start of April have prior months' data. Daily data was then aggregated to weekly resolution—binary indicators via weekly max, continuous variables via weekly mean, fire counts via weekly sum—aligning with the 1-2 week forecast horizon. Finally, target variables were created by forward-shifting the weekly compound event indicator by 1 and 2 weeks within each grid cell's time series.

**Role of AI:** Claude (Anthropic) was used to generate the data loading, cleaning, feature engineering, and export scripts, with each AI-generated section cited in code comments. Claude also assisted with feature engineering design (compound event definitions, rolling window strategies, seasonal encoding approach) and data dictionary generation. All AI-generated code was reviewed by the team.

## Dataset Summary

- **Full dataset:** 2,123,873 weekly observations, 36 columns, 2001-2020
- **Grid cells:** 3,378 land cells across CONUS at 0.5-degree resolution
- **Fire season:** April-October (~30 weeks/year)
- **Class distribution:** compound_hw_drought=1 in 4.08% of observations
- **CSV deliverable:** 2019-2020 subset (206,058 rows, 81.9 MB); full dataset reproducible from scripts
- **JSON deliverable:** 10,000-row representative sample

## Reproducibility

First create a directory named "dataset" at root, within it have the "ComExDBM_CONUS" dataset.

```bash
# From the repository root:
source .venv/Scripts/activate
uv pip install -r zero-shot-hopefuls/requirements.txt

# Run the pipeline (requires dataset/ directory with ComExDBM NetCDF files):
python zero-shot-hopefuls/Scripts/01_load_and_merge.py    # ~3 min: loads NetCDF, merges, saves daily parquet
python zero-shot-hopefuls/Scripts/02_clean_and_engineer.py # ~10 min: cleans, engineers features, aggregates weekly
python zero-shot-hopefuls/Scripts/03_export.py             # ~5 min: exports CSV, JSON, data dictionary
```

## File Structure

```
zero-shot-hopefuls/
  Data/
    original_sample.csv                        # 1,000-row raw daily data sample
    cleaned_compound_hazard_weekly.csv         # Cleaned dataset (2019-2020 subset, CSV format)
    cleaned_compound_hazard_weekly.json        # Cleaned dataset (10K-row sample, JSON format)
    cleaned_compound_hazard_weekly.csv.gz      # Full 20-year dataset (gzipped, local use)
    cleaned_compound_hazard_weekly.parquet     # Full 20-year dataset (parquet, local use)
    data_dictionary.json                       # Feature dictionary with descriptions and sources
    intermediate/                              # Per-year daily parquet files (generated by scripts)
  Scripts/
    01_load_and_merge.py                       # Load 4 NetCDF sources, merge, save daily data
    02_clean_and_engineer.py                   # Clean, engineer 7 features, aggregate weekly
    03_export.py                               # Export to CSV, JSON, data dictionary
  requirements.txt                             # Python environment dependencies
  README.md                                    # This file
```
