# SubSignal — Week 2 Deliverable
**AI Challenge | Data Preparation and Documentation**

---

## Model Selection

Team SubSignal will use a **Spatio-Temporal Graph Neural Network (ST-GNN)** to gain insight into geothermal reservoir connectivity and to detect significant changes in subsurface pressure response to fluid injection.

The ST-GNN is comprised of nodes and edges. For our data, nodes represent well locations (and depths) and edges represent inferred connectivity between wells. The model learns temporal dynamics — how signals evolve over time — which for subsurface systems includes pressure propagation, thermal diffusion, solute/EC transport, and response lags during injection cycles. Our analysis will focus on pressure propagation.

An ST-GNN is a neural network that (1) processes graph-structured data and (2) learns how strongly nodes influence each other, which can be used to infer connectivity, causality, or flow paths. The ST-GNN will simultaneously learn spatial coupling between wells (connectivity, influence, directionality) and temporal evolution of each signal (i.e., pressure). ST-GNNs are well-suited for subsurface monitoring and characterization because they can model distributed sensors with nonlinear pressure propagation and time-lagged interactions to gain insight into dynamic connectivity changes.

---

## Data Preparation

### Source Dataset
**FTES-Full_Test_1hour_avg.csv** — 1-hour averaged measurements from the DEMO-FTES Test 1 geothermal injection experiment (December 10, 2024 – March 24, 2025). 2,609 rows × 68 columns covering water flow rate, pressure, temperature, and electrical conductivity at five instrumented wells (TL, TN, TC, TU, TS).

### Processing Pipeline

| File | Rows | Cols | Description |
|---|---|---|---|
| `FTES-Full_Test_1hour_avg_original.csv` | 2,609 | 68 | Original raw dataset, untouched |
| `ftes_1hour_cleaned.csv` | 2,609 | 76 | Full cleaned dataset — snake_case column names, ISO timestamps, QA flags added |
| `ftes_1hour_cleaned_reduced.csv` | 2,591 | 17 | Pressure and flow columns only; duplicate-timestamp and time-gap rows flagged but retained |
| `ftes_1hour_cleaned_reduced_QC.csv` | 2,474 | 17 | Flagged rows removed (117 rows dropped: 116 duplicate timestamps + 1 time gap) |
| `ftes_1hour_cleaned_reduced_QC_engineered.csv` | 2,474 | 20 | QC'd dataset with 3 engineered features added |
| `ftes_scaled_for_GNN.csv` | 2,474 | 21 | Final GNN-ready dataset scaled using StandardScaler and RobustScaler |

### Data Preparation Steps

The FTES source data were cleaned into reproducible, audit-preserving, and model-input-friendly outputs using a scripted workflow. The cleaning process standardized column names to a stable snake_case schema, preserved the original timestamp text as `time_raw`, parsed a canonical ISO-format `time` field, normalized boolean state columns and the triplex on/off state where present, and checked expected numeric sensor fields for parseability. Data quality issues — including malformed timestamps, duplicate timestamps, non-monotonic time, unexpected time gaps, row-length mismatches, and non-numeric values — were not silently removed. Instead, they were surfaced through explicit QA flag columns so downstream users can trace anomalies back to the source data. Because the hourly source file contains duplicate timestamps with materially different sensor values, those rows were retained in the initial cleaned file and documented as source-data anomalies rather than automatically deduplicated. For the ST-GNN pipeline, flagged rows were subsequently removed, retaining 2,474 rows for modeling.

Feature engineering included adding binary flag columns indicating whether the primary well (TC) was either injecting heated water into the subsurface or pumping water from the subsurface. The third engineered feature derives the hourly pressure response to change in flow rate, expressed as change in pressure (PSI) divided by change in flow (L/min). Engineered features were created after the dataset was cleaned and QC-filtered.

Data scaling was performed on the cleaned, QC-filtered dataset with engineered features. StandardScaler was applied to pressure columns at 10 well locations, injection pressure, and injection flow rate — these variables are continuous with meaningful absolute magnitudes, and StandardScaler preserves relative differences while keeping data centered for GNN numerical stability. RobustScaler (median/IQR, quantile range 10–90) was applied to the ΔP/ΔQ ratio feature, which has heavy tails and occasional large excursions near ΔQ ≈ 0. Binary operational flags were kept as 0/1 without scaling.

AI tools were used to assist with data cleaning (GitHub Copilot / Claude Sonnet) and feature scaling script development (GitHub Copilot). All AI-generated code sections are cited in comments within the respective scripts.

---

## Data Dictionary

The table below covers the columns present in the final ST-GNN-ready dataset (`ftes_1hour_cleaned_reduced_QC_engineered.csv`). For the full data dictionary including all 68 original sensor columns, see `Data/FTES_data_dictionary.json`.

| Field Name | Description | Type | Unit |
|---|---|---|---|
| `source_dataset` | Identifier for the source dataset this row originates from | string | — |
| `time_raw` | Original timestamp string as it appeared in the source file | string | — |
| `time` | Parsed canonical timestamp in ISO format | string | datetime |
| `net_flow` | Injection water flow rate from Triplex pump | float | L/min |
| `injection_pressure` | Injection pressure associated with the active injection interval | float | psi |
| `tl_interval_pressure` | Water pressure TL interval | float | psi |
| `tl_bottom_pressure` | Water pressure TL bottom | float | psi |
| `tn_interval_pressure` | Water pressure TN interval | float | psi |
| `tn_bottom_pressure` | Water pressure TN bottom | float | psi |
| `tc_interval_pressure` | Water pressure TC interval | float | psi |
| `tc_bottom_pressure` | Water pressure TC bottom | float | psi |
| `tu_interval_pressure` | Water pressure TU interval | float | psi |
| `tu_bottom_pressure` | Water pressure TU bottom | float | psi |
| `ts_interval_pressure` | Water pressure TS interval | float | psi |
| `ts_bottom_pressure` | Water pressure TS bottom | float | psi |
| `flag_duplicate_timestamp` | 1 if this row's timestamp appears more than once in the source data, 0 otherwise | integer | 0/1 |
| `flag_time_gap_gt_expected` | 1 if the time gap before this row exceeds the expected 1-hour interval, 0 otherwise | integer | 0/1 |
| `tc_injecting` | Binary flag — well TC is actively injecting heated fluid into the subsurface (engineered feature) | integer | 0 = not injecting, 1 = injecting |
| `tc_producing` | Binary flag — well TC is actively pumping fluid from the subsurface (engineered feature) | integer | 0 = not producing, 1 = producing |
| `delta_P_delta_Q` | Ratio of hourly change in injection pressure to change in injection flow rate (engineered feature) | float | PSI / (L/min) |
