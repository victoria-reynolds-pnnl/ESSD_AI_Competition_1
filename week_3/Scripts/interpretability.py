import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


# =========================
# CONFIG
# =========================
DATA_PATH = "Data/week2_clean.csv"
SPLIT_PATH = "Data/splits_indexed.csv"
LABELED_PATH = "Data/week3_labeled_with_tiers.csv"

MODEL_1_PATH = "Models/model_1.pkl"
MODEL_2_PATH = "Models/model_2.pkl"

HDBSCAN_PREPROCESSOR_PATH = "Models/hdbscan_preprocessor.pkl"
TIER_MAPPER_PATH = "Models/tier_mapper.pkl"

VIS_DIR = "Visualizations"
OUT_DIR = "Interpretability"

DATE_COL = "centroid_date"
TARGET_COL = "high_risk"

FEATURES = [
    "lowest_temperature_k",
    "duration_days",
    "nerc_id",
    "spatial_coverage",
    "yearly_max_heat_wave_intensity",
    "yearly_max_heat_wave_duration",
    "yearly_max_heat_wave_intensity_trend",
    "yearly_max_heat_wave_duration_trend",
]


def ensure_dirs():
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def apply_pickle_compatibility_shim():
    """Allow older sklearn pickles to load on newer sklearn releases."""
    try:
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct._RemainderColsList = _RemainderColsList
    except Exception:
        # If sklearn internals changed again, we still attempt standard loading.
        pass


def patch_simple_imputer_state(obj, seen=None):
    """Recursively patch old SimpleImputer pickle state for newer sklearn runtime."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    cls_name = obj.__class__.__name__ if hasattr(obj, "__class__") else ""
    if cls_name == "SimpleImputer":
        if hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = obj._fit_dtype

    if isinstance(obj, dict):
        for v in obj.values():
            patch_simple_imputer_state(v, seen)
        return

    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            patch_simple_imputer_state(v, seen)
        return

    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            patch_simple_imputer_state(v, seen)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def attach_splits(df):
    splits = pd.read_csv(SPLIT_PATH)
    if "row_id" not in splits.columns or "split" not in splits.columns:
        raise ValueError("splits_indexed.csv must include 'row_id' and 'split' columns.")
    return df.merge(splits[["row_id", "split"]], on="row_id", how="inner")


def assign_tiers(X_raw, unsup_preprocessor, tier_mapper):
    X_transformed = unsup_preprocessor.transform(X_raw)
    return tier_mapper.predict(X_transformed)


def add_binary_target(df):
    df = df.copy()
    df[TARGET_COL] = (df["risk_tier_3"] == 2).astype(int)
    return df


def load_labeled_data_or_reconstruct():
    if os.path.exists(LABELED_PATH):
        print(f"Using existing labeled data: {LABELED_PATH}")
        df = pd.read_csv(LABELED_PATH)
        if TARGET_COL not in df.columns and "risk_tier_3" in df.columns:
            df = add_binary_target(df)
        return df

    print("Labeled file not found. Reconstructing labels from split and tier artifacts...")
    df = load_data()
    df = attach_splits(df)

    unsup_preprocessor = joblib.load(HDBSCAN_PREPROCESSOR_PATH)
    tier_mapper = joblib.load(TIER_MAPPER_PATH)
    df["risk_tier_3"] = assign_tiers(df[FEATURES].copy(), unsup_preprocessor, tier_mapper)
    df = add_binary_target(df)
    return df


def get_feature_names_from_preprocessor(preprocessor):
    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return FEATURES


def ensure_dense_array(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def safe_predict(model, X):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else np.nan,
    }
    return metrics


def save_confusion_matrix_plot(y_true, y_pred, model_label):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    disp.ax_.set_title(f"{model_label} - Test Confusion Matrix")
    fig.tight_layout()
    out_path = os.path.join(VIS_DIR, f"{model_label}_confusion_matrix.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_feature_importance_plot(model, X_ref, model_label):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_.ravel())
        title = f"{model_label} - |Coefficient| Importance"
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        title = f"{model_label} - Feature Importance"
    else:
        transformed = preprocessor.transform(X_ref)
        importances = np.var(np.asarray(transformed), axis=0)
        title = f"{model_label} - Variance Proxy Importance"

    n = min(len(importances), len(feature_names))
    feat_df = pd.DataFrame({
        "feature": feature_names[:n],
        "importance": np.asarray(importances)[:n],
    }).sort_values("importance", ascending=False)

    feat_df.to_csv(os.path.join(OUT_DIR, f"{model_label}_feature_importance.csv"), index=False)

    top_k = feat_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_k["feature"], top_k["importance"], color="#2A6F9E")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()

    out_path = os.path.join(VIS_DIR, f"{model_label}_feature_importance.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_error_tables(test_df, y_pred, y_prob, model_label):
    results = test_df.copy().reset_index(drop=True)
    results["predicted"] = y_pred
    results["predicted_probability"] = y_prob
    results["error_type"] = np.where(
        results[TARGET_COL] == results["predicted"],
        "correct",
        np.where(results[TARGET_COL] == 1, "false_negative", "false_positive"),
    )

    base_cols = [
        "row_id",
        "split",
        TARGET_COL,
        "predicted",
        "predicted_probability",
        "error_type",
        "nerc_id",
        "source_file",
        "lowest_temperature_k",
        "duration_days",
        "spatial_coverage",
        "yearly_max_heat_wave_intensity",
        "yearly_max_heat_wave_duration",
        "yearly_max_heat_wave_intensity_trend",
        "yearly_max_heat_wave_duration_trend",
    ]
    keep_cols = [col for col in base_cols if col in results.columns]

    predictions_path = os.path.join(OUT_DIR, f"{model_label}_test_predictions.csv")
    results[keep_cols].to_csv(predictions_path, index=False)

    fp_df = results[results["error_type"] == "false_positive"][keep_cols].copy()
    fn_df = results[results["error_type"] == "false_negative"][keep_cols].copy()

    fp_path = os.path.join(OUT_DIR, f"{model_label}_false_positives.csv")
    fn_path = os.path.join(OUT_DIR, f"{model_label}_false_negatives.csv")
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)

    return results, [predictions_path, fp_path, fn_path]


def save_precision_recall_comparison_plot(test_metrics_df):
    plot_df = test_metrics_df[["model", "precision", "recall"]].copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(plot_df))
    width = 0.34
    ax.bar(x - width / 2, plot_df["precision"], width=width, label="Precision", color="#255F85")
    ax.bar(x + width / 2, plot_df["recall"], width=width, label="Recall", color="#D17A22")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test Precision vs Recall by Model")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(VIS_DIR, "precision_recall_comparison.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_error_by_region_plot(all_results):
    region_rows = []
    for model_name, results in all_results.items():
        grouped = (
            results.assign(
                false_negative=(results["error_type"] == "false_negative").astype(int),
                false_positive=(results["error_type"] == "false_positive").astype(int),
            )
            .groupby("nerc_id", dropna=False)[["false_negative", "false_positive"]]
            .sum()
            .reset_index()
        )
        grouped["model"] = model_name
        region_rows.append(grouped)

    region_df = pd.concat(region_rows, ignore_index=True)
    region_df.to_csv(os.path.join(OUT_DIR, "error_by_region_summary.csv"), index=False)

    pivot_df = region_df.copy()
    pivot_df["nerc_id"] = pivot_df["nerc_id"].fillna("missing")
    labels = pivot_df["model"] + " | " + pivot_df["nerc_id"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot_df))
    width = 0.34
    ax.bar(x - width / 2, pivot_df["false_negative"], width=width, label="False negatives", color="#B33A3A")
    ax.bar(x + width / 2, pivot_df["false_positive"], width=width, label="False positives", color="#447C69")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Error Counts by Model and Region")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(VIS_DIR, "error_by_region.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path, os.path.join(OUT_DIR, "error_by_region_summary.csv")


def save_false_negative_rate_by_region_outputs(all_results):
    rate_rows = []
    for model_name, results in all_results.items():
        grouped = (
            results.assign(
                positive_actual=(results[TARGET_COL] == 1).astype(int),
                false_negative=(results["error_type"] == "false_negative").astype(int),
            )
            .groupby("nerc_id", dropna=False)[["positive_actual", "false_negative"]]
            .sum()
            .reset_index()
        )
        grouped["false_negative_rate"] = np.where(
            grouped["positive_actual"] > 0,
            grouped["false_negative"] / grouped["positive_actual"],
            np.nan,
        )
        grouped["model"] = model_name
        rate_rows.append(grouped)

    rate_df = pd.concat(rate_rows, ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "false_negative_rate_by_region.csv")
    rate_df.to_csv(csv_path, index=False)

    plot_df = rate_df.copy()
    plot_df["nerc_id"] = plot_df["nerc_id"].fillna("missing")
    labels = plot_df["model"] + " | " + plot_df["nerc_id"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["false_negative_rate"], color="#A23B72")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("False negative rate")
    ax.set_title("False Negative Rate by Model and Region")
    fig.tight_layout()

    plot_path = os.path.join(VIS_DIR, "false_negative_rate_by_region.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return plot_path, csv_path


def build_error_analysis_note(test_metrics_df, all_results):
    lr_metrics = test_metrics_df.loc[test_metrics_df["model"] == "logistic_regression"].iloc[0]
    xgb_metrics = test_metrics_df.loc[test_metrics_df["model"] == "xgboost"].iloc[0]

    lr_results = all_results["logistic_regression"]
    xgb_results = all_results["xgboost"]

    def mean_or_nan(frame, column):
        return float(frame[column].mean()) if len(frame) else float("nan")

    lr_fn = lr_results[lr_results["error_type"] == "false_negative"]
    xgb_fn = xgb_results[xgb_results["error_type"] == "false_negative"]
    lr_tp = lr_results[(lr_results[TARGET_COL] == 1) & (lr_results["predicted"] == 1)]
    xgb_tp = xgb_results[(xgb_results[TARGET_COL] == 1) & (xgb_results["predicted"] == 1)]

    lines = []
    lines.append("# Error Analysis Summary")
    lines.append("")
    lines.append(
        f"On the test set, logistic regression is the less conservative model: it reaches recall {lr_metrics['recall']:.3f} with precision {lr_metrics['precision']:.3f}, while XGBoost reaches recall {xgb_metrics['recall']:.3f} with much higher precision {xgb_metrics['precision']:.3f}. This means logistic regression catches more true high-risk cases but also produces many more false positives, whereas XGBoost is more selective and misses more positives.")
    lines.append("")
    lines.append(
        f"The false-negative burden is materially larger for XGBoost ({len(xgb_fn)} missed high-risk cases) than for logistic regression ({len(lr_fn)} missed high-risk cases). Relative to their correctly identified positives, the missed cases in both models tend to show weaker heat-wave intensity and shorter duration signals on average, which suggests the models struggle most on borderline high-risk events rather than on the most extreme events.")
    lines.append("")
    unique_regions = sorted(set(lr_results['nerc_id'].dropna().astype(str)).union(set(xgb_results['nerc_id'].dropna().astype(str))))
    if len(unique_regions) <= 1:
        lines.append("Region-level error analysis is limited in this dataset because the test records only expose a single `nerc_id` category, so there is no meaningful cross-region contrast to evaluate. The grouped error plot still documents the imbalance in false positives versus false negatives by model.")
    else:
        lines.append("Region-level error counts vary across `nerc_id`, so the grouped error plot should be used to check whether false negatives cluster geographically.")
    lines.append("")
    lines.append("Average feature profile of missed high-risk cases:")
    lines.append(
        f"- Logistic regression false negatives: mean intensity {mean_or_nan(lr_fn, 'yearly_max_heat_wave_intensity'):.2f}, mean annual duration {mean_or_nan(lr_fn, 'yearly_max_heat_wave_duration'):.2f}, mean event duration {mean_or_nan(lr_fn, 'duration_days'):.2f}")
    lines.append(
        f"- Logistic regression true positives: mean intensity {mean_or_nan(lr_tp, 'yearly_max_heat_wave_intensity'):.2f}, mean annual duration {mean_or_nan(lr_tp, 'yearly_max_heat_wave_duration'):.2f}, mean event duration {mean_or_nan(lr_tp, 'duration_days'):.2f}")
    lines.append(
        f"- XGBoost false negatives: mean intensity {mean_or_nan(xgb_fn, 'yearly_max_heat_wave_intensity'):.2f}, mean annual duration {mean_or_nan(xgb_fn, 'yearly_max_heat_wave_duration'):.2f}, mean event duration {mean_or_nan(xgb_fn, 'duration_days'):.2f}")
    lines.append(
        f"- XGBoost true positives: mean intensity {mean_or_nan(xgb_tp, 'yearly_max_heat_wave_intensity'):.2f}, mean annual duration {mean_or_nan(xgb_tp, 'yearly_max_heat_wave_duration'):.2f}, mean event duration {mean_or_nan(xgb_tp, 'duration_days'):.2f}")

    note_path = os.path.join(OUT_DIR, "error_analysis_notes.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return note_path


def save_calibration_outputs(test_results_by_model):
    brier_rows = []
    curve_points = []

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    for model_name, results in test_results_by_model.items():
        y_true = results[TARGET_COL].to_numpy()
        y_prob = results["predicted_probability"].to_numpy()

        brier = brier_score_loss(y_true, y_prob)
        brier_rows.append({"model": model_name, "brier_score": float(brier)})

        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        for mp, fp in zip(mean_pred, frac_pos):
            curve_points.append({
                "model": model_name,
                "mean_predicted_probability": float(mp),
                "fraction_positives": float(fp),
            })

        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=f"{model_name}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Calibration Curve (Test Set)")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(VIS_DIR, "calibration_curve.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    brier_df = pd.DataFrame(brier_rows).sort_values("brier_score")
    brier_path = os.path.join(OUT_DIR, "calibration_brier_scores.csv")
    brier_df.to_csv(brier_path, index=False)

    curve_df = pd.DataFrame(curve_points)
    curve_path = os.path.join(OUT_DIR, "calibration_curve_points.csv")
    curve_df.to_csv(curve_path, index=False)

    best_model = brier_df.iloc[0]["model"]
    note_lines = [
        "# Calibration Note",
        "",
        f"Brier score compares probability calibration quality, with lower values indicating better calibrated probabilities.",
        f"On the test set, {best_model} has the lower Brier score and therefore better overall calibration under this metric.",
        "Calibration curves are included to show where each model is over-confident or under-confident across probability ranges.",
    ]
    note_path = os.path.join(OUT_DIR, "calibration_notes.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(note_lines) + "\n")

    return plot_path, [brier_path, curve_path, note_path]


def summarize_shap_contributions(shap_row, feature_names, top_n=5):
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_row,
        "abs_shap": np.abs(shap_row),
    }).sort_values("abs_shap", ascending=False)

    pushed_up = contributions[contributions["shap_value"] > 0].head(top_n)[["feature", "shap_value"]]
    pushed_down = contributions[contributions["shap_value"] < 0].head(top_n)[["feature", "shap_value"]]

    return {
        "pushed_up": pushed_up.to_dict(orient="records"),
        "pushed_down": pushed_down.to_dict(orient="records"),
    }


def infer_physical_sense(case_summary):
    dominant_features = [item["feature"] for item in case_summary["pushed_up"] + case_summary["pushed_down"]]
    weather_drivers = [
        feature for feature in dominant_features
        if any(token in feature for token in ["temperature", "duration", "intensity", "trend"])
    ]
    region_drivers = [feature for feature in dominant_features if "nerc_id" in feature]

    if weather_drivers and not region_drivers:
        return "The explanation is physically plausible because weather severity variables dominate the local prediction."
    if weather_drivers and region_drivers:
        return "The explanation is partly physically plausible because event severity features matter, but region proxy effects should be watched for over-reliance."
    if region_drivers:
        return "The explanation is less physically satisfying because location proxy features appear to dominate more than direct event severity measures."
    return "The explanation is mixed and should be reviewed manually against domain expectations."


def save_xgboost_shap_outputs(model, X_test, y_test):
    import shap

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(X_test)
    transformed_dense = ensure_dense_array(transformed)
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    feature_frame = pd.DataFrame(transformed_dense, columns=feature_names)
    explainer = shap.TreeExplainer(classifier)

    sample_size = min(2000, len(feature_frame))
    sample_frame = feature_frame.iloc[:sample_size].copy()
    shap_values_sample = explainer.shap_values(sample_frame)

    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values_sample, sample_frame, show=False)
    plt.title("XGBoost SHAP Summary")
    plt.tight_layout()
    summary_path = os.path.join(VIS_DIR, "xgboost_shap_summary.png")
    plt.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close()

    y_pred, y_prob = safe_predict(model, X_test)
    results = X_test.copy().reset_index(drop=True)
    results["actual"] = y_test.reset_index(drop=True).to_numpy()
    results["predicted"] = y_pred
    results["predicted_probability"] = y_prob

    high_idx = int(results["predicted_probability"].idxmax())
    low_idx = int(results["predicted_probability"].idxmin())

    wrong_mask = results["actual"] != results["predicted"]
    if wrong_mask.any():
        wrong_idx = int(
            results.loc[wrong_mask, "predicted_probability"]
            .sub(results.loc[wrong_mask, "actual"])
            .abs()
            .idxmax()
        )
    else:
        wrong_idx = high_idx

    case_map = {
        "high_risk_case": high_idx,
        "low_risk_case": low_idx,
        "incorrect_case": wrong_idx,
    }

    shap_values_cases = explainer.shap_values(feature_frame.iloc[list(case_map.values())])
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_scalar = float(np.ravel(expected_value)[0])
    else:
        expected_scalar = float(expected_value)

    notes = []
    saved_paths = [summary_path]

    for case_name, idx in case_map.items():
        row_frame = feature_frame.iloc[[idx]]
        row_shap = np.asarray(shap_values_cases[list(case_map.keys()).index(case_name)]).ravel()
        row_data = row_frame.iloc[0]

        explanation = shap.Explanation(
            values=row_shap,
            base_values=expected_scalar,
            data=row_data.to_numpy(),
            feature_names=feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        case_path = os.path.join(VIS_DIR, f"xgboost_shap_{case_name}.png")
        plt.savefig(case_path, dpi=180, bbox_inches="tight")
        plt.close()
        saved_paths.append(case_path)

        contribution_summary = summarize_shap_contributions(row_shap, feature_names)
        notes.append({
            "case": case_name,
            "row_index": int(idx),
            "actual": int(results.loc[idx, "actual"]),
            "predicted": int(results.loc[idx, "predicted"]),
            "predicted_probability": float(results.loc[idx, "predicted_probability"]),
            "pushed_up": contribution_summary["pushed_up"],
            "pushed_down": contribution_summary["pushed_down"],
            "physical_sense_note": infer_physical_sense(contribution_summary),
        })

    notes_path = os.path.join(OUT_DIR, "xgboost_shap_case_notes.json")
    with open(notes_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    return saved_paths, notes_path


def main():
    warnings.filterwarnings("ignore")
    ensure_dirs()
    apply_pickle_compatibility_shim()

    required_paths = [
        MODEL_1_PATH,
        MODEL_2_PATH,
    ]
    for p in required_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    if not os.path.exists(LABELED_PATH):
        for p in [DATA_PATH, SPLIT_PATH, HDBSCAN_PREPROCESSOR_PATH, TIER_MAPPER_PATH]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required file for label reconstruction: {p}")

    print("Loading labeled data...")
    df = load_labeled_data_or_reconstruct()

    print("Loading trained supervised models...")
    model_1 = joblib.load(MODEL_1_PATH)
    model_2 = joblib.load(MODEL_2_PATH)
    patch_simple_imputer_state(model_1)
    patch_simple_imputer_state(model_2)

    model_info = {
        "model_1": "logistic_regression",
        "model_2": "xgboost",
    }
    models = {
        "logistic_regression": model_1,
        "xgboost": model_2,
    }

    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    summary_rows = []
    saved_figures = []
    extra_outputs = []
    test_result_tables = {}

    for model_name, model in models.items():
        print(f"Processing {model_name}...")

        X_val = val_df[FEATURES].copy()
        y_val = val_df[TARGET_COL].copy()
        X_test = test_df[FEATURES].copy()
        y_test = test_df[TARGET_COL].copy()

        val_pred, val_prob = safe_predict(model, X_val)
        test_pred, test_prob = safe_predict(model, X_test)

        val_metrics = compute_metrics(y_val, val_pred, val_prob)
        test_metrics = compute_metrics(y_test, test_pred, test_prob)

        summary_rows.append({
            "model": model_name,
            "split": "val",
            **val_metrics,
        })
        summary_rows.append({
            "model": model_name,
            "split": "test",
            **test_metrics,
        })

        test_results, error_paths = save_error_tables(test_df, test_pred, test_prob, model_name)
        test_result_tables[model_name] = test_results
        extra_outputs.extend(error_paths)

        saved_figures.append(save_confusion_matrix_plot(y_test, test_pred, model_name))
        saved_figures.append(save_feature_importance_plot(model, X_test, model_name))

        if model_name == "xgboost":
            shap_paths, notes_path = save_xgboost_shap_outputs(model, X_test, y_test)
            saved_figures.extend(shap_paths)
            extra_outputs.append(notes_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "interpretability_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    test_metrics_df = summary_df[summary_df["split"] == "test"].copy()
    saved_figures.append(save_precision_recall_comparison_plot(test_metrics_df))
    region_plot_path, region_summary_path = save_error_by_region_plot(test_result_tables)
    saved_figures.append(region_plot_path)
    extra_outputs.append(region_summary_path)
    fn_rate_plot_path, fn_rate_csv_path = save_false_negative_rate_by_region_outputs(test_result_tables)
    saved_figures.append(fn_rate_plot_path)
    extra_outputs.append(fn_rate_csv_path)
    extra_outputs.append(build_error_analysis_note(test_metrics_df, test_result_tables))
    calibration_plot_path, calibration_outputs = save_calibration_outputs(test_result_tables)
    saved_figures.append(calibration_plot_path)
    extra_outputs.extend(calibration_outputs)

    manifest = {
        "model_mapping": model_info,
        "summary_metrics_file": summary_path,
        "saved_figures": saved_figures,
        "extra_outputs": extra_outputs,
    }
    manifest_path = os.path.join(OUT_DIR, "interpretability_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Interpretability workflow complete.")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved manifest: {manifest_path}")
    print("Saved figures:")
    for fig_path in saved_figures:
        print(f"- {fig_path}")


if __name__ == "__main__":
    main()
