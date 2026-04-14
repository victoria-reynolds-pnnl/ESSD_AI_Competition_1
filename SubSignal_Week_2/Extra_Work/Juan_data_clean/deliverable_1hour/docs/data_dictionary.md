# Week 2 Data Dictionary

## Purpose
This document defines the cleaned-data contract for the 1-hour FTES Week 2 deliverable in this folder and covers every source column present in `data/original/FTES-Full_Test_1hour_avg.csv`.

## Deliverable Use
- Audit-preserving cleaned outputs keep the normalized schema, original value text, and QA flags for traceability.
- ML-ready cleaned outputs use the same normalized field names, coerce numeric fields into numeric-compatible values, and retain QA flags for downstream filtering or feature selection.

## Cleaned Output Contract Fields
| Clean Field | Type | Source | Description | Notes |
| --- | --- | --- | --- | --- |
| `source_dataset` | string | derived | Dataset identifier (`ftes_1hour`) | Added by pipeline |
| `time_raw` | string | `Time` | Original timestamp text from the source file | Preserved verbatim |
| `time` | string | derived from `Time` | Parsed canonical timestamp in ISO-like `YYYY-MM-DDTHH:MM:SS` format | Empty if parse fails |
| `flag_bad_timestamp_format` | string | derived | Row-level timestamp parse failure flag | `true` / `false` |
| `flag_non_monotonic_time` | string | derived | Row timestamp moved backward relative to previous row | `true` / `false` |
| `flag_duplicate_timestamp` | string | derived | Row timestamp equals previous row timestamp | `true` / `false` |
| `flag_time_gap_gt_expected` | string | derived | Gap larger than expected cadence | `true` / `false` |
| `flag_row_length_mismatch` | string | derived | Raw row column count differed from header width | `true` / `false` |
| `flag_non_numeric_in_numeric_field` | string | derived | At least one expected numeric field failed parsing | `true` / `false` |

## ML-Ready Output Differences
- `time_raw` is omitted from the ML-ready file.
- Numeric sensor fields are emitted as numeric-compatible text values.
- QA flag fields are encoded as `0` or `1` in the ML-ready file instead of `true` or `false`.
- The hourly dataset does not include the 1-second-only `raw_row_index`, triplex-state field, or boolean valve-state fields.

## Header Normalization Rules
1. Blank source headers would become `raw_row_index`.
2. Non-blank headers are converted to lowercase snake_case.
3. Non-alphanumeric characters become underscores.
4. Repeated underscores collapse to one underscore.
5. `(True/False)` is removed from boolean field names before output.
6. If a normalized name collides, a numeric suffix is appended.

## Value Normalization Rules
### Timestamps
- 1-hour source input format: `%Y-%m-%d %H:%M:%S`
- Output format: `YYYY-MM-DDTHH:MM:SS`

### Numeric fields
- First pass behavior:
  - preserve original value text in cleaned output
  - record row-level numeric anomaly via `flag_non_numeric_in_numeric_field`
  - aggregate offending columns in summary JSON

## Units And Interpretation Notes
- Flow fields use `L/min` per source description.
- Pressure fields use `psi` per source description.
- Temperature fields use `C` per source description.
- Packer depth fields use `ft` per source description (`Feet` in the docx source).
- EC fields are marked as `EC units (source doc unspecified)` because the description names the signal but does not define the unit.
- All source fields remain string-valued in cleaned CSV output during Week 2; numeric and boolean interpretation is documented here and checked via QA.

## Source Column Coverage: `FTES-Full_Test_1hour_avg.csv`
| Source Column | Clean Field | Type | Units | Description | Notes |
| --- | --- | --- | --- | --- | --- |
| `Time` | `time` | string (raw timestamp) |  | Date | Also emitted as derived `time_raw` and parsed `time`. |
| `Injection EC` | `injection_ec` | string numeric | EC units (source doc unspecified) | Injection water electrical conductivity | Hourly-only measurement column. |
| `Net Flow` | `net_flow` | string numeric | L/min | Injection water flow rate from Triplex pump |  |
| `TL Interval Flow` | `tl_interval_flow` | string numeric | L/min | Production water flow rate TL interval |  |
| `TL Bottom Flow` | `tl_bottom_flow` | string numeric | L/min | Production water flow rate TL bottom |  |
| `TL Collar Flow` | `tl_collar_flow` | string numeric | L/min | Production water flow rate TL collar |  |
| `TL Interval EC` | `tl_interval_ec` | string numeric | EC units (source doc unspecified) | Water EC TL interval |  |
| `TL Bottom EC` | `tl_bottom_ec` | string numeric | EC units (source doc unspecified) | Water EC TL bottom |  |
| `TL Interval Pressure` | `tl_interval_pressure` | string numeric | psi | Water pressure TL interval |  |
| `TL Bottom Pressure` | `tl_bottom_pressure` | string numeric | psi | Water pressure TL bottom |  |
| `TL Packer Pressure` | `tl_packer_pressure` | string numeric | psi | Packer pressure TL |  |
| `TL-TEC-INT-U` | `tl_tec_int_u` | string numeric | C | Temperature TL interval upper |  |
| `TL-TEC-INT-L` | `tl_tec_int_l` | string numeric | C | Temperature TL interval lower |  |
| `TL-TEC-BOT-U` | `tl_tec_bot_u` | string numeric | C | Temperature TL bottom upper |  |
| `TL-TEC-BOT-L` | `tl_tec_bot_l` | string numeric | C | Temperature TL bottom lower |  |
| `TN Interval Flow` | `tn_interval_flow` | string numeric | L/min | Production water flow rate TN interval |  |
| `TN Bottom Flow` | `tn_bottom_flow` | string numeric | L/min | Production water flow rate TN bottom |  |
| `TN Collar Flow` | `tn_collar_flow` | string numeric | L/min | Production water flow rate TN collar |  |
| `TN Interval EC` | `tn_interval_ec` | string numeric | EC units (source doc unspecified) | Water EC TN interval |  |
| `TN Bottom EC` | `tn_bottom_ec` | string numeric | EC units (source doc unspecified) | Water EC TN bottom |  |
| `TN Interval Pressure` | `tn_interval_pressure` | string numeric | psi | Water pressure TN interval |  |
| `TN Bottom Pressure` | `tn_bottom_pressure` | string numeric | psi | Water pressure TN bottom |  |
| `TN Packer Pressure` | `tn_packer_pressure` | string numeric | psi | Packer pressure TN |  |
| `TN-TEC-INT-U` | `tn_tec_int_u` | string numeric | C | Temperature TN interval upper |  |
| `TN-TEC-INT-L` | `tn_tec_int_l` | string numeric | C | Temperature TN interval lower |  |
| `TN-TEC-BOT-U` | `tn_tec_bot_u` | string numeric | C | Temperature TN bottom upper |  |
| `TN-TEC-BOT-L` | `tn_tec_bot_l` | string numeric | C | Temperature TN bottom lower |  |
| `TC Interval Flow` | `tc_interval_flow` | string numeric | L/min | Production water flow rate TC interval |  |
| `TC Bottom Flow` | `tc_bottom_flow` | string numeric | L/min | Production water flow rate TC bottom |  |
| `TC Collar Flow` | `tc_collar_flow` | string numeric | L/min | Production water flow rate TC collar |  |
| `TC Interval EC` | `tc_interval_ec` | string numeric | EC units (source doc unspecified) | Water EC TC interval |  |
| `TC Bottom EC` | `tc_bottom_ec` | string numeric | EC units (source doc unspecified) | Water EC TC bottom |  |
| `TC Interval Pressure` | `tc_interval_pressure` | string numeric | psi | Water pressure TC interval |  |
| `TC Bottom Pressure` | `tc_bottom_pressure` | string numeric | psi | Water pressure TC bottom |  |
| `Injection Pressure` | `injection_pressure` | string numeric | psi | Injection water pressure | Hourly-only measurement column. |
| `TC Packer Pressure` | `tc_packer_pressure` | string numeric | psi | Packer pressure TC |  |
| `TC-TEC-INT-U` | `tc_tec_int_u` | string numeric | C | Temperature TC interval upper |  |
| `TC-TEC-INT-L` | `tc_tec_int_l` | string numeric | C | Temperature TC interval lower |  |
| `TC-TEC-BOT-U` | `tc_tec_bot_u` | string numeric | C | Temperature TC bottom upper |  |
| `TC-TEC-BOT-L` | `tc_tec_bot_l` | string numeric | C | Temperature TC bottom lower |  |
| `TU Interval Flow` | `tu_interval_flow` | string numeric | L/min | Production water flow rate TU interval |  |
| `TU Bottom Flow` | `tu_bottom_flow` | string numeric | L/min | Production water flow rate TU bottom |  |
| `TU Collar Flow` | `tu_collar_flow` | string numeric | L/min | Production water flow rate TU collar |  |
| `TU Interval EC` | `tu_interval_ec` | string numeric | EC units (source doc unspecified) | Water EC TU interval |  |
| `TU Bottom EC` | `tu_bottom_ec` | string numeric | EC units (source doc unspecified) | Water EC TU bottom |  |
| `TU Interval Pressure` | `tu_interval_pressure` | string numeric | psi | Water pressure TU interval |  |
| `TU Bottom Pressure` | `tu_bottom_pressure` | string numeric | psi | Water pressure TU bottom |  |
| `PT 403` | `pt_403` | string numeric | psi | Pressure transducer 403 reading | Hourly-only measurement column. |
| `TU Packer Pressure` | `tu_packer_pressure` | string numeric | psi | Packer pressure TU |  |
| `TU-TEC-INT-U` | `tu_tec_int_u` | string numeric | C | Temperature TU interval upper |  |
| `TU-TEC-INT-L` | `tu_tec_int_l` | string numeric | C | Temperature TU interval lower |  |
| `TU-TEC-BOT-U` | `tu_tec_bot_u` | string numeric | C | Temperature TU bottom upper |  |
| `TU-TEC-BOT-L` | `tu_tec_bot_l` | string numeric | C | Temperature TU bottom lower |  |
| `TS Interval Flow` | `ts_interval_flow` | string numeric | L/min | Production water flow rate TS interval |  |
| `TS Bottom Flow` | `ts_bottom_flow` | string numeric | L/min | Production water flow rate TS bottom |  |
| `TS Collar Flow` | `ts_collar_flow` | string numeric | L/min | Production water flow rate TS collar |  |
| `TS Interval EC` | `ts_interval_ec` | string numeric | EC units (source doc unspecified) | Water EC TS interval |  |
| `TS Bottom EC` | `ts_bottom_ec` | string numeric | EC units (source doc unspecified) | Water EC TS bottom |  |
| `TS Interval Pressure` | `ts_interval_pressure` | string numeric | psi | Water pressure TS interval |  |
| `TS Bottom Pressure` | `ts_bottom_pressure` | string numeric | psi | Water pressure TS bottom |  |
| `PT 503` | `pt_503` | string numeric | psi | Pressure transducer 503 reading | Hourly-only measurement column. |
| `PT 504` | `pt_504` | string numeric | psi | Pressure transducer 504 reading | Hourly-only measurement column. |
| `TS Packer Pressure` | `ts_packer_pressure` | string numeric | psi | Packer pressure TS |  |
| `TL Packer Center Depth` | `tl_packer_center_depth` | string numeric | ft | Depth along borehole TL packer |  |
| `TN Packer Center Depth` | `tn_packer_center_depth` | string numeric | ft | Depth along borehole TN packer |  |
| `TC Packer Center Depth` | `tc_packer_center_depth` | string numeric | ft | Depth along borehole TC packer |  |
| `TU Packer Center Depth` | `tu_packer_center_depth` | string numeric | ft | Depth along borehole TU packer |  |
| `TS Packer Center Depth` | `ts_packer_center_depth` | string numeric | ft | Depth along borehole TS packer |  |

## Remaining Open Items
1. The supplied Week 1 and Week 2 resource `.docx` files did not expose a more precise unit definition for EC, so EC units remain intentionally unspecified here.
2. The supplied `.docx` resources did not provide clearer domain labels for `PT 403`, `PT 503`, or `PT 504`, so they remain described as pressure transducer readings.
