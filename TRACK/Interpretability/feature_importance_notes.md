# Feature Importance Interpretation

## Logistic Regression

The logistic regression model is driven most strongly by `yearly_max_heat_wave_intensity`, `yearly_max_heat_wave_duration`, and `duration_days`, with `spatial_coverage` as a secondary contributor. That overall pattern is physically believable because more intense and longer-lasting heat conditions should increase the chance that an event is labeled high risk. One nuance is that `duration_days` has a relatively large negative coefficient in the fitted linear model, which suggests its effect is being interpreted conditionally alongside the annual maximum severity features rather than as a simple standalone risk amplifier.

## XGBoost

XGBoost shows a very similar ranking, with `yearly_max_heat_wave_intensity`, `yearly_max_heat_wave_duration`, `duration_days`, and `spatial_coverage` contributing the most importance, while the trend features are clearly secondary. This is also physically sensible: the model is primarily responding to event severity and persistence rather than obscure engineered artifacts.

## Temperature, Duration, and Trend Variables

Across both models, the most influential predictors are severity-related variables tied to heat-wave intensity and duration, which is the expected behavior for a climate-risk classification task. Trend variables matter, but they are not dominating the models, so they appear to be adding context rather than overwhelming the direct event measurements.

## `nerc_id` Check

`nerc_id` is not dominating either model. In fact, its measured importance is effectively zero because the labeled dataset contains only one observed category (`NERC`), so it provides almost no usable discrimination in the current training setup.

## Bottom Line

The feature-importance outputs are broadly credible because both models prioritize physically meaningful severity features over location proxies. The main caveat is the logistic regression sign on `duration_days`, which should be interpreted carefully as a conditional linear effect rather than proof that longer events are inherently safer.
