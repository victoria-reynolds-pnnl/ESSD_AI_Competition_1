# FTES Dataset — Week 2 Deliverable
**Team:** byteMe | **Competition:** ESSD AI Competition | **Due:** April 2, 2026

---

## 1. ML Model Selection

We selected a **Random Forest regressor with engineered lag features (RF-Lag)** as our primary model because it handles the nonlinear, time-delayed relationship between injection conditions (rate, temperature) and production-well thermal response without requiring assumptions about the functional form of heat transport through a fracture network. Random Forest is also robust to the moderate dataset size (~1,700 hourly training rows), naturally ranks feature importance for physical validation, and allows direct multi-step forecasting across the four prediction horizons (+15 min, +60 min, +240 min, +1440 min) by training one independent regressor per well–horizon pair.

---

## 2. Data Dictionary

The processed output file (`processedData/FTES_cleaned_1hour.csv`) contains the columns described below. All rows cover the hot-water injection phase of FTES Test 1 at the Sanford Underground Research Facility (SURF): **December 13, 2024 20:00 – February 23, 2025**.

Wells: **TC** = injection well | **TL, TN** = production wells | **TU, TS** = monitor wells

Sensor naming convention: `XX-TEC-INT-U` = well XX, thermoelectric chain (TEC), interval zone, upper sensor.

### 2a. Index & Split Label

| Field | Description | Type | Example |
|-------|-------------|------|---------|
| `timestamp` | UTC datetime of the hourly average record | datetime | `2024-12-14 08:00:00` |
| `split` | ML pipeline role for this row (`ramp_up` / `train` / `test`) | string | `train` |

### 2b. System-Level Measurements

| Field | Description | Units | Type | Example |
|-------|-------------|-------|------|---------|
| `Injection EC` | Electrical conductivity of injected water at surface | µS/cm | float | `371.997` |
| `Net Flow` | Net volumetric flow rate of the injection system | L/min | float | `1.770` |

### 2c. Well TL — Production Well (Borehole TL)

| Field | Description | Units | Type | Example |
|-------|-------------|-------|------|---------|
| `TL Interval Flow` | Flow rate through the TL interval packer zone | L/min | float | `0.00590` |
| `TL Bottom Flow` | Flow rate through the TL bottom packer zone | L/min | float | `0.00290` |
| `TL Collar Flow` | Flow rate measured at TL borehole collar | L/min | float | `0.0955` |
| `TL Interval EC` | Electrical conductivity in TL interval zone | µS/cm | float | `412.235` |
| `TL Bottom EC` | Electrical conductivity in TL bottom zone | µS/cm | float | `470.889` |
| `TL Interval Pressure` | Fluid pressure in TL interval zone | psi | float | `4.622` |
| `TL Bottom Pressure` | Fluid pressure in TL bottom zone | psi | float | `21.522` |
| `TL-TEC-INT-U` | Temperature at TL interval zone — upper sensor | °C | float | `25.814` |
| `TL-TEC-INT-L` | Temperature at TL interval zone — lower sensor | °C | float | `25.930` |
| `TL-TEC-BOT-U` | Temperature at TL bottom zone — upper sensor | °C | float | `25.880` |
| `TL-TEC-BOT-L` | Temperature at TL bottom zone — lower sensor | °C | float | `26.043` |
| `TL Packer Center Depth` | Fixed depth of TL packer midpoint | m | float | `170.2` |
| `TL_INT_mean` | Average of TL-TEC-INT-U and TL-TEC-INT-L | °C | float | `25.872` |
| `TL_BOT_mean` | Average of TL-TEC-BOT-U and TL-TEC-BOT-L | °C | float | `25.961` |

### 2d. Well TN — Production Well (Borehole TN)

| Field | Description | Units | Type | Example |
|-------|-------------|-------|------|---------|
| `TN Interval Flow` | Flow rate through the TN interval packer zone | L/min | float | `0.00221` |
| `TN Bottom Flow` | Flow rate through the TN bottom packer zone | L/min | float | `0.00114` |
| `TN Collar Flow` | Flow rate measured at TN borehole collar | L/min | float | `0.00608` |
| `TN Interval EC` | Electrical conductivity in TN interval zone | µS/cm | float | `1046.085` |
| `TN Bottom EC` | Electrical conductivity in TN bottom zone | µS/cm | float | `2169.390` |
| `TN Interval Pressure` | Fluid pressure in TN interval zone | psi | float | `-6.492` |
| `TN Bottom Pressure` | Fluid pressure in TN bottom zone | psi | float | `-10.462` |
| `TN Packer Pressure` | Pressure reading at TN packer | psi | float | `48.187` |
| `TN-TEC-INT-U` | Temperature at TN interval zone — upper sensor | °C | float | `24.355` |
| `TN-TEC-INT-L` | Temperature at TN interval zone — lower sensor | °C | float | `22.926` |
| `TN-TEC-BOT-U` | Temperature at TN bottom zone — upper sensor | °C | float | `22.893` |
| `TN-TEC-BOT-L` | Temperature at TN bottom zone — lower sensor | °C | float | `24.407` |
| `TN_INT_mean` | Average of TN-TEC-INT-U and TN-TEC-INT-L | °C | float | `23.641` |
| `TN_BOT_mean` | Average of TN-TEC-BOT-U and TN-TEC-BOT-L | °C | float | `23.650` |

### 2e. Well TC — Injection Well (Borehole TC)

| Field | Description | Units | Type | Example |
|-------|-------------|-------|------|---------|
| `TC Interval Flow` | Flow rate through the TC interval packer zone | L/min | float | `0.230` |
| `TC Bottom Flow` | Flow rate through the TC bottom packer zone | L/min | float | `-0.000149` |
| `TC Collar Flow` | Flow rate measured at TC borehole collar | L/min | float | `0.109` |
| `TC Interval EC` | Electrical conductivity in TC interval zone | µS/cm | float | `259.758` |
| `TC Bottom EC` | Electrical conductivity in TC bottom zone | µS/cm | float | `2076.720` |
| `TC Bottom Pressure` | Fluid pressure in TC bottom zone | psi | float | `52.604` |
| `TC-TEC-INT-U` | Temperature at TC interval zone — upper sensor | °C | float | `25.576` |
| `TC-TEC-INT-L` | Temperature at TC interval zone — lower sensor | °C | float | `26.559` |
| `TC-TEC-BOT-U` | Temperature at TC bottom zone — upper sensor | °C | float | `26.127` |
| `TC-TEC-BOT-L` | Temperature at TC bottom zone — lower sensor | °C | float | `25.214` |
| `TC_INT_mean` | Average of TC-TEC-INT-U and TC-TEC-INT-L | °C | float | `26.067` |
| `TC_BOT_mean` | Average of TC-TEC-BOT-U and TC-TEC-BOT-L | °C | float | `25.671` |

### 2f. Well TU — Monitor Well (Borehole TU)

| Field | Description | Units | Type | Example |
|-------|-------------|-------|------|---------|
| `TU Interval Flow` | Flow rate through the TU interval packer zone | L/min | float | `-0.00468` |
| `TU Bottom Flow` | Flow rate through the TU bottom packer zone | L/min | float | `0.01526` |
| `TU Collar Flow` | Flow rate measured at TU borehole collar | L/min | float | `0.00294` |
| `TU Interval EC` | Electrical conductivity in TU interval zone | µS/cm | float | `2093.593` |
| `TU Bottom EC` | Electrical conductivity in TU bottom zone | µS/cm | float | `511.704` |
| `TU Bottom Pressure` | Fluid pressure in TU bottom zone | psi | float | `9.606` |
| `TU-TEC-INT-U` | Temperature at TU interval zone — upper sensor | °C | float | `25.298` |
| `TU-TEC-INT-L` | Temperature at TU interval zone — lower sensor | °C | float | `25.396` |
| `TU-TEC-BOT-U` | Temperature at TU bottom zone — upper sensor | °C | float | `25.462` |
| `TU-TEC-BOT-L` | Temperature at TU bottom zone — lower sensor | °C | float | `25.288` |
| `TU Packer Center Depth` | Fixed depth of TU packer midpoint | m | float | `177.8` |
| `TU_INT_mean` | Average of TU-TEC-INT-U and TU-TEC-INT-L | °C | float | `25.347` |
| `TU_BOT_mean` | Average of TU-TEC-BOT-U and TU-TEC-BOT-L | °C | float | `25.375` |

### 2g. Well TS — Monitor Well (Borehole TS)

| Field | Description | Units | Type | Example |
|-------|-------------|-------|------|---------|
| `TS Interval Flow` | Flow rate through the TS interval packer zone | L/min | float | `-0.00478` |
| `TS Bottom Flow` | Flow rate through the TS bottom packer zone | L/min | float | `-0.000984` |
| `TS Collar Flow` | Flow rate measured at TS borehole collar | L/min | float | `0.02555` |
| `TS Interval EC` | Electrical conductivity in TS interval zone | µS/cm | float | `413.950` |
| `TS Bottom EC` | Electrical conductivity in TS bottom zone | µS/cm | float | `654.982` |
| `TS Interval Pressure` | Fluid pressure in TS interval zone | psi | float | `-13.437` |
| `TS Bottom Pressure` | Fluid pressure in TS bottom zone | psi | float | `-9.160` |
| `TS Packer Pressure` | Pressure reading at TS packer | psi | float | `-39.649` |
| `PT 503` | Pressure transducer 503 reading | psi | float | `132.538` |
| `PT 504` | Pressure transducer 504 reading | psi | float | `115.441` |

### 2h. Engineered Features

| Field | Description | Units | Type | Example | Feature Family |
|-------|-------------|-------|------|---------|----------------|
| `TC_INT_delta` | Hourly rate of change of TC_INT_mean (injection temperature acceleration) | °C/hr | float | `-0.121` | A |
| `net_flow_rolling_6h` | 6-hour trailing mean of Net Flow; smooths pump transients | L/min | float | `1.770` | B |
| `dT_TL_dt` | Hourly rate of temperature rise at production well TL | °C/hr | float | `0.031` | C |
| `dT_TN_dt` | Hourly rate of temperature rise at production well TN | °C/hr | float | `0.018` | C |
| `elapsed_injection_min` | Minutes elapsed since injection start (2024-12-13 20:00); clips to 0 before injection | min | float | `1440.0` | D |
| `delta_T_above_T0_TL` | TL_INT_mean minus pre-injection ambient T0 for TL; **primary prediction target** | °C | float | `2.143` | D |
| `delta_T_above_T0_TN` | TN_INT_mean minus pre-injection ambient T0 for TN; **primary prediction target** | °C | float | `1.076` | D |
| `cumulative_heat_input` | Running sum of (Net Flow × TC_INT_mean × Δt); encodes total thermal energy delivered to the fracture system | °C·L·s | float | `4,218,032.5` | D |
| `T_gradient_INT_TL` | TL-TEC-INT-U minus TL-TEC-INT-L; vertical temperature gradient within TL packer interval | °C | float | `-0.116` | D |
| `T_gradient_INT_TN` | TN-TEC-INT-U minus TN-TEC-INT-L; vertical temperature gradient within TN packer interval | °C | float | `1.429` | D |
| `T_gradient_INT_TC` | TC-TEC-INT-U minus TC-TEC-INT-L; vertical temperature gradient within TC packer interval | °C | float | `-0.984` | D |
| `T_gradient_INT_TU` | TU-TEC-INT-U minus TU-TEC-INT-L; vertical temperature gradient within TU packer interval | °C | float | `-0.098` | D |
| `days_since_injection` | Decimal days since injection start; derived from elapsed_injection_min | days | float | `1.0` | D (legacy) |
| `hour_sin` | Sine of fractional hour-of-day mapped to [0, 2π]; cyclic time encoding | dimensionless | float | `-0.866` | — |
| `hour_cos` | Cosine of fractional hour-of-day mapped to [0, 2π]; cyclic time encoding | dimensionless | float | `0.500` | — |
| `delta_T_inj_prod` | TC-TEC-INT-U minus mean of TL/TN TEC-INT-U; instantaneous injection-to-production thermal contrast | °C | float | `0.491` | — |
| `cumulative_injected_volume` | Running sum of positive Net Flow × Δt; flow-volume proxy (superseded by cumulative_heat_input) | L·s | float | `6371.061` | D (legacy) |

---

## 3. Data Preparation

The raw FTES dataset was cleaned and transformed using the Python script `byteMe/scripts/clean_and_feature_engineer.py` (dependencies: pandas ≥ 2.0, numpy ≥ 1.24, scipy ≥ 1.11). The pipeline proceeds as follows.

**Pre-injection baseline (AI-assisted):** Before any filtering, the three days of pre-injection sensor readings (December 10–13) are used to compute a per-well ambient temperature baseline T0 — the median of the averaged upper/lower interval sensors. GitHub Copilot (Claude Sonnet 4.6) identified this step as necessary to shift the prediction target from absolute temperature to temperature rise above ambient, which removes well-to-well underground offset and improves model generalizability. GitHub Copilot wrote the `compute_T0()` function and determined that it must execute on the full unfiltered dataset before Phase 1 filtering removes the pre-injection rows.

**Phase 1 filtering (AI-assisted):** Only the hot-water injection phase (December 13, 2024 20:00 – February 23, 2025) is retained. Phase 2 ambient circulation data is excluded entirely, as it represents a physically distinct flow regime that would cause the model to learn a mixture of incompatible behaviors. GitHub Copilot recommended tightening the start boundary from midnight to 20:00 on December 13 to match the actual pump-on timestamp, and advised excluding Phase 2 specifically to prevent cross-regime data leakage.

**Cleaning (AI-assisted):** Duplicate timestamps (116 found in the hourly file) are removed, keeping the first occurrence. Columns with a single constant value across Phase 1 (e.g., fixed packer depths) are dropped as they carry zero information. Physically implausible sensor readings are clamped to NaN using domain-knowledge thresholds: temperatures outside –5 °C to 150 °C, pressures above 500 psi, and electrical conductivity values outside 0–5,000 µS/cm. Columns that are still entirely NaN after clamping are dropped. Short gaps of one to two consecutive missing readings are recovered by time-based linear interpolation. Columns remaining more than 50% missing after interpolation (primarily injection-side pressure sensors operating outside their rated range) are dropped entirely. GitHub Copilot structured the full cleaning pipeline, selected the physical plausibility thresholds using domain reasoning, and recommended the 50% NaN threshold as the cut-off for unrecoverable columns rather than raising the pressure ceiling globally.

**Sensor drift detrending (AI-assisted):** GitHub Copilot flagged that the monitor wells TU and TS — located far from the injection well — should not show sustained temperature trends during Phase 1; any linear rise is instrument drift rather than real thermal signal. GitHub Copilot wrote the `check_monitor_drift()` function, which fits a linear regression to each monitor well's upper interval sensor over the full Phase 1 window and subtracts the trend from all temperature columns of that well if the slope exceeds 0.12 °C/hr (0.002 °C/min, the threshold specified in the experimental design).

**Interval means and ramp-up exclusion (AI-assisted):** Upper and lower packer sensors for each well are averaged into interval-mean and bottom-mean columns to reduce per-sensor noise. GitHub Copilot recommended this averaging step to reduce noise propagation into downstream rolling statistics and cumulative features, and wrote the `compute_interval_means()` function. GitHub Copilot also designed the dynamic ramp-up detection logic in `detect_ramp_up_end()` — identifying the first period where Net Flow sustains ≥ 80% of its 6-hour rolling maximum for 60 consecutive minutes — and wrote `add_split_labels()` to encode the `ramp_up` / `train` / `test` partition directly in the output CSV so that all downstream model training respects the strictly chronological split required by the experimental design.

**Feature engineering (AI-assisted):** GitHub Copilot designed a set of physics-motivated features grouped into families. Family A captures injection-well signal: the hourly rate of change of injection temperature (`TC_INT_delta`). Family B captures flow rate history: a 6-hour trailing mean of Net Flow (`net_flow_rolling_6h`), which smooths pump transients and gives the model memory of recent injection activity — the primary driver of advective heat transport. Family C captures autoregressive target history: the hourly rate of temperature rise at each production well (`dT_TL_dt`, `dT_TN_dt`). Family D encodes the cumulative thermal state of the fracture system: elapsed injection time in minutes (`elapsed_injection_min`), temperature rise above T0 per production well (`delta_T_above_T0_TL`, `delta_T_above_T0_TN`) as the primary prediction targets, cumulative heat input as the running integral of flow multiplied by injection temperature and time-step length (`cumulative_heat_input` — rated Critical importance in the experimental design), and vertical thermal gradient within each packer interval (`T_gradient_INT_{well}`). GitHub Copilot proposed all feature families, wrote the `engineer_features()` function, and recommended replacing the original flow-only `cumulative_injected_volume` with `cumulative_heat_input` on the basis that hotter injected water carries more thermal energy per unit volume — the physical mechanism driving production-well breakthrough. Additional cyclic hour-of-day encodings (sine and cosine) are included to handle any diurnal sensor or operational patterns without ordinal artifacts. All features are computed using only past or present information to prevent data leakage into future target values.
