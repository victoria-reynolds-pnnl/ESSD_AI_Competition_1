# Data Dictionary

## Feature Matrix (`data/clean/feature_matrix.csv`)

This is the primary modeling input — one row per water year with all features and the target variable.

| Field | Description | Type | Units | Source | Example |
|---|---|---|---|---|---|
| `water_year` | Water year (Oct Y-1 through Sep Y) | int | year | BPA | 2005 |
| `target_volume` | April-September naturalized flow volume at The Dalles | float | kcfs-days | BPA modified flow (TDA6M) | 47807 |
| `apr1_swe_inches` | Basin-average April 1 snow water equivalent | float | inches | NRCS SNOTEL (16 stations) | 28.7 |
| `apr1_swe_anomaly_pct` | April 1 SWE as percent departure from median | float | % | Derived from SNOTEL | 12.5 |
| `djf_nino34` | December-February mean Nino 3.4 SST anomaly | float | deg C | NOAA PSL | -0.45 |
| `djf_pdo` | December-February mean Pacific Decadal Oscillation index | float | unitless | NOAA PSL | -1.2 |
| `djf_pna` | December-February mean Pacific-North American pattern index | float | unitless | NOAA PSL | 0.3 |
| `jan_mar_mean_q_cfs` | January-March mean naturalized flow at The Dalles | float | cfs | BPA modified flow (TDA6M) | 120000 |
| `oct_mar_volume_kcfs_days` | October-March cumulative naturalized flow volume | float | kcfs-days | BPA modified flow (TDA6M) | 22500 |

## Target Variable (`data/clean/target_apr_sep_volume.csv`)

| Field | Description | Type | Units |
|---|---|---|---|
| `water_year` | Water year | int | year |
| `apr_sep_volume_kcfs_days` | April-September naturalized flow volume at The Dalles | float | kcfs-days |

## Raw Data Sources

### BPA Natural Flow (`data/raw/bpa_the_dalles_natural_monthly.csv`)

Monthly mean naturalized (modified) streamflow at The Dalles, OR. "Modified" flow removes the effects of reservoir regulation to represent natural conditions. Source: BPA Historical Streamflow Reports, control point TDA6M.

- Period: WY 1929-2019 (91 years)
- Units: cfs (monthly mean)
- Columns: water_year, Oct, Nov, Dec, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, AVE

### USGS Observed Flow (`data/raw/usgs_the_dalles_daily_q.csv`)

Daily mean discharge at USGS site 14105700 (Columbia River at The Dalles, OR). This is the *regulated* (actual) flow, used for QC comparison against the natural flow.

- Period: 1950-01-01 to 2025-12-31
- Units: cfs (daily mean)
- Retrieved via: `dataretrieval` Python package (USGS NWIS)

### SNOTEL SWE (`data/raw/snotel_swe.csv`)

Monthly start-of-period snow water equivalent from 16 SNOTEL stations across the Columbia Basin. Stations span Montana (upper Columbia), Idaho (Snake River), Oregon (Cascades), and Washington (Cascades).

- Period: 1985-2025 (40 water years)
- Units: inches
- Retrieved via: NRCS Report Generator API
- Stations: Hoodoo Basin MT, Kraft Creek MT, Nez Perce Camp MT, Noisy Basin MT, Stahl Peak MT, Galena ID, Jackson Peak ID, Lost-Wood Divide ID, Brundage Reservoir ID, Trinity Mtn ID, Blazed Alder OR, Mt Hood Test Site OR, Chemult Alternate OR, Aneroid Lake #2 OR, Stampede Pass WA, Stevens Pass WA

### Climate Indices (`data/raw/climate_indices.csv`)

Monthly teleconnection indices from NOAA Physical Sciences Laboratory.

| Index | Full Name | Period |
|---|---|---|
| `pdo` | Pacific Decadal Oscillation | 1948-2025 |
| `nino34` | Nino 3.4 SST Anomaly | 1950-2026 |
| `pna` | Pacific-North American Pattern | 1950-2026 |
| `amo` | Atlantic Multidecadal Oscillation | 1948-2023 |

## Key Derived Datasets

### Cleaned Monthly Flow (`data/clean/natural_flow_monthly.csv`)

| Field | Description | Type | Units |
|---|---|---|---|
| `water_year` | Water year | int | year |
| `month` | Month abbreviation (Oct, Nov, ..., Sep) | str | - |
| `cal_year` | Calendar year | int | year |
| `cal_month` | Calendar month (1-12) | int | - |
| `mean_flow_cfs` | Monthly mean naturalized flow | float | cfs |
| `volume_kcfs_days` | Monthly flow volume | float | kcfs-days |

### Basin-Average SWE (`data/clean/snotel_apr1_swe.csv`)

| Field | Description | Type | Units |
|---|---|---|---|
| `water_year` | Water year | int | year |
| `apr1_swe_inches` | Mean April 1 SWE across 16 stations | float | inches |
