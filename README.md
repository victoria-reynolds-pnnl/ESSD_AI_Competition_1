# Week 3 Submission - ML Model Training and Assessment

**Team:** The Philadelphia Eagles  
**Date:** 2026-04-09  
**Status:** Complete

## Overview

This folder contains the Week 3 deliverables for the CONUS fire occurrence classification benchmark. We trained and evaluated two machine learning models following the locked v1 pipeline specification defined in [`../wiki/02.03 CONUS fire occurrence v1 pipeline plan.md`](../wiki/02.03%20CONUS%20fire%20occurrence%20v1%20pipeline%20plan.md).

**Upstream Data Source:** Week 3 loads its starting dataset from the Week 2 deliverable at `../week_2/Data/cleaned_dataset.*` as the canonical handoff.

## Test/Training Split Strategy

### Temporal Split

We used a fixed temporal split to prevent data leakage and ensure realistic evaluation:

- **Training:** 2001-2016 (16 years)
- **Validation:** 2017-2018 (2 years)
- **Test:** 2019-2020 (2 years)

### Rationale

- **Temporal ordering:** Models are trained on past data and evaluated on future data, simulating real-world deployment
- **No random splits:** Prevents temporal leakage where future information could influence past predictions
- **Validation for threshold tuning:** Used validation set to select optimal classification thresholds before final test evaluation

### Negative Sampling

Due to severe class imbalance (fire occurrence rate ~0.3%), we applied negative downsampling to the training set only:

- **Method:** Stratified sampling within year-month buckets
- **Ratio:** 10:1 negative-to-positive samples
- **Applied to:** Training set only (validation and test sets remain unmodified)
- **Result:** Training set reduced from ~16M rows to 574,101 rows while preserving all positive samples

This approach:
- Reduces training time and memory requirements
- Maintains temporal distribution
- Preserves realistic class distribution in evaluation sets

## Model Performance Summary

### Test Set Metrics (2019-2020)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Threshold |
|-------|--------|---------|-----|-----------|--------|-----------|
| **Logistic Regression** | 0.2031 | 0.8549 | 0.2908 | 0.3754 | 0.2373 | 0.9160 |
| **Hist Gradient Boosting** | **0.2772** | **0.9381** | **0.3303** | **0.4968** | 0.2474 | 0.8155 |

**Test set composition:** 2,049,024 samples (6,355 fires, 2,042,669 non-fires, 0.31% positive rate)

### Key Findings

1. **Histogram Gradient Boosting outperforms Logistic Regression** across all metrics
   - 36% improvement in PR-AUC (0.2031 → 0.2772)
   - 10% improvement in ROC-AUC (0.8549 → 0.9381)
   - 14% improvement in F1 score (0.2908 → 0.3303)
   - 32% improvement in precision (0.3754 → 0.4968)

2. **PR-AUC is the primary metric** for this imbalanced classification task
   - Baseline (random classifier) PR-AUC ≈ 0.0031 (positive rate)
   - Both models substantially exceed baseline
   - HGB achieves 89x improvement over baseline

3. **High precision, moderate recall trade-off**
   - HGB achieves ~50% precision at ~25% recall
   - Suitable for screening applications where false positives are costly
   - Threshold tuning on validation set optimized F1 score

### Confusion Matrices

**Logistic Regression:**
```
                Predicted
              No Fire    Fire
Actual No Fire  2,040,160  2,509
       Fire        4,847  1,508
```

**Histogram Gradient Boosting:**
```
                Predicted
              No Fire    Fire
Actual No Fire  2,041,077  1,592
       Fire        4,783  1,572
```

## Failures and Limitations

The models we developed show several fundamental limitations that stem from both the problem's inherent difficulty and our pipeline design choices. The most significant issue is the low recall of around 24-25%, meaning we're only catching about one in four actual fires, which reflects the extreme class imbalance (0.3% positive rate) and our conservative threshold selection, though it also suggests that many fires occur under rare environmental conditions not well-represented in our training data. Our precision is moderate at 38-50%, meaning roughly half to two-thirds of positive predictions are false alarms, though this still represents substantial improvement over baseline and might be acceptable for screening systems if not automated response. Beyond performance metrics, we're predicting same-day fire occurrence rather than building a true forecasting system, so there's no lead time for intervention—by the time the model flags high risk, fires may already be burning. On the data side, we're missing important spatial context since v1 doesn't include spatial neighbor features that could capture fire spread patterns and regional clustering, and our temporal context is limited to just 7-day rolling windows that may miss longer-term drought and vegetation stress patterns. From an evaluation perspective, we're testing on a single period (2019-2020) without cross-validation, our F1-optimized thresholds may not align with operational cost-benefit requirements, and we're aggregating metrics across all CONUS which masks potentially significant regional variation. Finally, technical constraints around the large dataset size (20M+ rows) forced chunked processing that limited hyperparameter tuning, leaving us with relatively simple model configurations (50 trees for gradient boosting, 100 iterations for logistic regression) and basic feature engineering without exploring interaction terms or non-linear transformations that might capture more complex relationships in the data.

## HITL Review

### Human-in-the-Loop Assessment

**Reviewers:** Haiden Deaton and Brian Dean
**Review Date:** 2026-04-09
**Review Focus:** Model behavior, feature patterns, and operational viability

After reviewing the model outputs, evaluation metrics, and feature importance patterns, we are generally satisfied with the Week 3 results as a baseline for fire occurrence prediction. The histogram gradient boosting model shows meaningful predictive skill, achieving a PR-AUC of 0.277 on the 2019-2020 test set—roughly 89 times better than random guessing given our 0.31% fire occurrence rate. For such an imbalanced problem where fires are genuinely rare events driven by complex interactions of weather, vegetation, and human factors, this is honestly quite reasonable.

**Does the model behavior make sense given domain expectations?** Yes, the models behave as expected from a fire science perspective. The feature importance analysis reveals that past fire activity (captured by `fhs_c9_lag7_sum`) is the strongest predictor, which aligns with the reality that fires often persist across multiple days or reignite in nearby areas. The seasonal encoding features (`doy_sin` and `doy_cos`) rank highly, correctly capturing the May-October fire season pattern we filtered for. Temperature-related features like `tmax` and `temp_range` also show up prominently, which makes sense since hot, dry conditions with large diurnal temperature swings create favorable fire weather.

**Any concerning spurious correlations?** There aren't any obvious red flags in the feature patterns, but there are some concerns worth monitoring. The heavy reliance on lagged fire features (`fhs_c9_lag1` and `fhs_c9_lag7_sum`) means the model is essentially learning "fires happen where fires recently happened," which is true but not particularly useful for early warning in new locations. This could lead to spatial bias where the model performs well in historically fire-prone regions but misses emerging risks in areas without recent fire history. The moderate precision (~50% for gradient boosting) also suggests the model is picking up on some conditions that look fire-like but aren't, possibly confusing high fire weather with actual ignition events. We're also aggregating performance across the entire CONUS, which masks potential regional differences—what works in California's Mediterranean climate might not work in Florida's subtropical regime.

**Any high-risk failure modes?** The most concerning failure mode is the low recall of 24-25%, meaning we're missing about three out of every four actual fires. For an operational early warning system, this could be dangerous if stakeholders assume comprehensive coverage. The model seems to be optimized for precision over recall (which makes sense given our F1-based threshold tuning), but this trade-off might not align with real-world risk tolerance where missing a catastrophic fire could be far more costly than a few false alarms. Another risk is that we're predicting same-day fire occurrence rather than forecasting future risk, so there's no lead time for preventive action—by the time the model flags high risk, the fire may already be burning. The temporal split also means we're testing on 2019-2020 data, which may not represent future climate conditions or extreme events outside the training distribution. If fire regimes shift due to climate change, model performance could degrade significantly.

Overall, this is a solid baseline that demonstrates the feasibility of ML-based fire prediction, but it's not ready for operational deployment without regional calibration.

## Reproducibility Steps

### Prerequisites

- Python 3.8+
- Dependencies: `pip install -r requirements.txt`
- Week 2 cleaned dataset at `../week_2/Data/cleaned_dataset.csv.gz`

### Execution

Run the complete Week 3 pipeline:

```bash
# 1. Train models (generates models and split-indexed data)
python3 week_3/Scripts/train.py

# 2. Evaluate on test set (generates ML_results.csv and metrics)
python3 week_3/Scripts/evaluate.py

# 3. Generate interpretability visualizations
python3/Scripts/interpretability.py
```

### Expected Outputs

```
week_3/
├── Data/
│   ├── week_2_data.csv.gz           # Copy of input data
│   ├── split_indexed_data.csv.gz    # Validation + test splits
│   └── ML_results.csv                # Test predictions from both models
├── Models/
│   ├── logistic_regression.pkl       # Trained baseline model
│   ├── hist_gradient_boosting.pkl    # Trained main model
│   ├── training_summary.json         # Training metadata and validation metrics
│   └── evaluation_summary.json       # Test set evaluation results
├── Visualizations/
│   ├── logistic_regression_feature_importance.png
│   ├── logistic_regression_pr_roc_curves.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── hist_gradient_boosting_pr_roc_curves.png
│   ├── hist_gradient_boosting_confusion_matrix.png
│   └── model_comparison.png
└── Scripts/
    ├── train.py
    ├── evaluate.py
    └── interpretability.py
```

### Runtime

- Training: ~30 seconds (with negative sampling)
- Evaluation: ~15 seconds
- Visualizations: ~10 seconds
- **Total:** ~1 minute on standard hardware

### Verification

```bash
# Run smoke tests
python -m unittest discover week_3/tests

# Check outputs exist
ls -lh week_3/Models/*.pkl
ls -lh week_3/Data/*.csv
ls -lh week_3/Visualizations/*.png
```

## References

- [Week 3 Deliverables](../wiki/01.04%20Week%203%20Deliverables.md)
- [Week 3 Action Plan](../wiki/03.01%20Week%203%20Action%20Plan.md)
- [CONUS fire occurrence v1 pipeline plan](../wiki/02.03%20CONUS%20fire%20occurrence%20v1%20pipeline%20plan.md)
- [Week 2 Data Pipeline](../week_2/README.md)

## Notes

- This implementation prioritizes reproducibility and benchmark consistency over experimentation
- The same target definition and split reference will be reused for Week 4 LLM benchmarking
- All scripts use memory-efficient chunked processing to handle the 20M+ row dataset
- Visualizations use non-interactive matplotlib backend for headless execution
- Week 3 generated artifacts are stored in this folder structure, not in `/workspace/data`
- The devcontainer mount at `/workspace/data` remains available for external raw dataset access
