# XGBoost SHAP Notes

## Summary Plot

The SHAP summary for XGBoost indicates that event severity variables dominate the model's ranking behavior. In particular, yearly maximum heat-wave duration, yearly maximum heat-wave intensity, event duration, and their trend variables are the strongest recurring drivers of predicted high risk.

## Individual Cases

### Predicted High-Risk Case

- Predicted probability: 0.8448
- Actual / predicted class: 1 / 1
- Main features pushing risk up: `yearly_max_heat_wave_duration`, `yearly_max_heat_wave_intensity`, `duration_days`, `yearly_max_heat_wave_duration_trend`, `yearly_max_heat_wave_intensity_trend`
- Main features pushing risk down: none among the top contributors
- Interpretation: this explanation is physically plausible because longer and more intense heat-wave conditions increase the model's estimated risk.

### Predicted Low-Risk Case

- Predicted probability: 0.0005
- Actual / predicted class: 0 / 0
- Main features pushing risk up: `duration_days`, `spatial_coverage`, `lowest_temperature_k`
- Main features pushing risk down: `yearly_max_heat_wave_duration`, `yearly_max_heat_wave_intensity`, `yearly_max_heat_wave_duration_trend`, `yearly_max_heat_wave_intensity_trend`
- Interpretation: this explanation is physically plausible because weak heat-wave severity signals strongly suppress the model's risk estimate.

### Incorrect Prediction

- Predicted probability: 0.0016
- Actual / predicted class: 1 / 0
- Main features pushing risk up: `yearly_max_heat_wave_duration`
- Main features pushing risk down: `yearly_max_heat_wave_intensity`, `duration_days`, `spatial_coverage`, `yearly_max_heat_wave_duration_trend`, `yearly_max_heat_wave_intensity_trend`
- Interpretation: this false negative suggests the model can miss some true high-risk cases when intensity and duration-related signals conflict, even if one duration feature points toward higher risk.

## Overall Interpretation

The local SHAP explanations are broadly consistent with domain expectations because weather severity features, rather than region proxy terms, drive the strongest positive and negative contributions in these selected examples. The most important caution is the false negative case, where the model appears overly dismissive when several severity signals point downward at the same time.
