# Sub‑Terra Tensor Tinkerers - Week 4 Submission

  

**Project Title:** Signal‑to‑Storage: Early Warning Forecasts for FTES Charging Operations

  

**Team:** Scott Unger, Theresa Pham, Xi Tan, Ashton Kirol, Mike Parker

README generated from code and results prompts using AI Incubator gpt 5.2   

1. Approach Report

Part of the LLM task was framed as binary threshold classification rather than pressure regression. At each timestep the model is asked whether injection pressure will exceed 5,000 psi at +1h, +6h, +12h, and +24h horizons. This reframing was deliberate: the operationally meaningful question is whether the threshold will be crossed, and classification plays to LLM strengths (pattern recognition and directional reasoning). Input features mirror our previous Kalman Filter inputs: injection pressure, net flow, net subsurface flow, cumulative net flow. We augmented these with engineered signals that give the LLM trajectory context it cannot derive from a single snapshot: pressure change rates over 1h and 6h windows, rolling pressure stability (std over 24h), consecutive hours above threshold, injection reduction signals, and a pre-computed regime label (e.g. sustained_above, rising_distant, imminent_crossing). Three few-shot examples drawn exclusively from the training period cover the key operational regimes (rising below alert, in alert band, above threshold), and temperature was set to 0.0 for deterministic outputs. None of the few shot examples contained mixed signals or crossing events (ex. above at +12h but below at +6h). Interestingly, the LLM did create mixed forecasts at different horizons that sometimes successfully captured threshold crossing event timing.

We included data from the entire field campaign (Train, Test, Validate from ML modeling) because the operational regime differed so much within the data. The regime label was the most critical design decision. Early iterations showed the LLM correctly identifying rising pressure but over-predicting threshold crossings in the training period when pressure was rising slowly from well below threshold — it recognised the directional signal but could not perform the implicit arithmetic to determine that the crossing was weeks away rather than hours. Rather than providing a raw hours-to-threshold estimate and expecting the LLM to reason quantitatively, the crossing likelihood was pre-computed into categorical labels that map directly to classification decisions. This removed the arithmetic burden entirely and improved precision significantly, particularly for the test period where the LLM achieved F1=0.984 at +1h, competitive with the Kalman Filter baseline of F1=0.975.

2. Error Analysis

The LLM achieves near-perfect recall (1.000) across all splits and horizons — it never misses a genuine threshold crossing. Performance on Test is excellent (F1=0.984 at +1h), competitive with the Kalman Filter. The weakness is precision in Train and Validate periods, where false positives occur when pressure is rising slowly from well below threshold or transitioning between regimes — the model correctly identifies the upward trajectory but overestimates crossing imminence. Crucially, false positives are consistently lower-confidence (0.48–0.78) while true positives carry high confidence (0.88–0.98), meaning a simple confidence threshold filter would substantially reduce false alarms without sacrificing recall.

3. Comparison Summary

| Model   | Split    | Horizon | Precision | Recall | F1    |
|---------|----------|---------|-----------|--------|-------|
| KF      | Test     | +1h     | 0.996     | 0.955  | 0.975 |
| KF      | Test     | +24h    | 0.994     | 0.710  | 0.828 |
| KF      | Validate | +1h     | 0.990     | 1.000  | 0.995 |
| KF      | Validate | +24h    | 0.872     | 0.851  | 0.861 |
| XGBoost | Test     | +1h     | 0.996     | 0.962  | 0.979 |
| XGBoost | Validate | +1h     | 0.677     | 0.946  | 0.789 |
| LLM     | Test     | +1h     | 0.989     | 0.978  | 0.984 |
| LLM     | Test     | +24h    | 0.923     | 0.903  | 0.913 |
| LLM     | Validate | +1h     | 0.552     | 1.000  | 0.712 |
| LLM     | Validate | +24h    | 0.530     | 1.000  | 0.693 |

The Kalman Filter is the strongest overall baseline. It achieves near-perfect threshold detection (Validate F1=0.995 at +1h, 0.861 at +24h) with physically interpretable state decomposition and graceful degradation across horizons. The continuous state estimate naturally captures reservoir pressurisation history without requiring explicit feature engineering of duration or trajectory.

The LLM is competitive on Test but struggles with false positives. Recall is consistently 1.000 — the LLM never misses a real threshold crossing — but Precision is limited by over-prediction of "above" in periods where pressure is rising but far from threshold. The LLM correctly identifies trajectory direction but does not reliably reason about rate — it cannot infer that "rising at 3 psi/hr from 3500 psi" will not cross 5000 psi within 24 hours without an explicit pre-computed signal.

XGBoost is limited by regime change. Strong within-distribution performance collapses when validate-period operating conditions fall outside the training range. This is a fundamental constraint of tree-based models on single-phase directional datasets.

The regime change problem is the binding constraint for all methods. A single operational episode moving in one direction means each split is a different regime. All models were trained on rising-pressure active-injection data and evaluated on curtailed/stabilising conditions. More data covering multiple injection cycles would substantially improve all approaches.

4. Repeatability Steps

All LLM queries were run at temperature=0.0 with a fixed random seed (seed=0), ensuring deterministic outputs for a given model and prompt. The three few-shot examples are drawn programmatically from the training period using fixed selection criteria (first qualifying row per scenario), so the same examples will be selected on any re-run against the same dataset. The stratified sampling of 100 rows per split uses a fixed random seed (seed=42), producing an identical evaluation subset on every run. Results can be reproduced by running the classification script against the same cleaned data file with the same model endpoint and the configuration values documented above.