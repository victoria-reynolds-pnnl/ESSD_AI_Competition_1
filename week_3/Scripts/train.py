import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestCentroid

from xgboost import XGBClassifier
import hdbscan


# =========================
# CONFIG
# =========================
DATA_PATH = "Data/week2_clean.csv"
SPLIT_OUTPUT_PATH = "Data/splits_indexed.csv"
MODEL_DIR = "Models"
ARTIFACT_DIR = "Models"
RANDOM_STATE = 42

DATE_COL = "centroid_date"
RISK_SCORE_COL = "cumulative_heat_stress_index"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

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

EXCLUDED_COLUMNS = [
    "start_date",
    "end_date",
    "centroid_date",
    "inter_event_recovery_interval_days",
    "cumulative_heat_stress_index",
    "risk_tier_3",
    "high_risk",
]

assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-8


# =========================
# UTILITIES
# =========================
def make_dirs():
    os.makedirs("Data", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data(path):
    return pd.read_csv(path)


def sort_by_time(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def create_time_splits(df):
    df = df.copy()
    n = len(df)

    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    split_labels = np.array([""] * n, dtype=object)
    split_labels[:train_end] = "train"
    split_labels[train_end:val_end] = "val"
    split_labels[val_end:] = "test"

    df["row_id"] = np.arange(n)
    df["split"] = split_labels
    return df


def save_split_index(df):
    split_df = df[["row_id", DATE_COL, "split"]].copy()
    split_df.to_csv(SPLIT_OUTPUT_PATH, index=False)


def infer_feature_types(df, feature_cols):
    categorical_features = [col for col in feature_cols if df[col].dtype == "object"]
    numerical_features = [col for col in feature_cols if col not in categorical_features]
    return numerical_features, categorical_features


def build_unsupervised_preprocessor(numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_supervised_preprocessor(numerical_features, categorical_features):
    return build_unsupervised_preprocessor(numerical_features, categorical_features)


# =========================
# HDBSCAN TIERING
# =========================
def fit_hdbscan_and_create_3_tiers(X_train_raw, train_df, feature_cols):
    """
    1. Preprocess training features
    2. Fit HDBSCAN on training only
    3. Rank discovered clusters by average cumulative_heat_stress_index
    4. Map them into 3 tiers: low / medium / high
    5. Train a nearest-centroid classifier on the transformed training space
       so val/test can be assigned to those tiers
    """
    numerical_features, categorical_features = infer_feature_types(X_train_raw, feature_cols)
    unsup_preprocessor = build_unsupervised_preprocessor(numerical_features, categorical_features)

    X_train_transformed = unsup_preprocessor.fit_transform(X_train_raw)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=25,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    cluster_labels = clusterer.fit_predict(X_train_transformed)
    train_df = train_df.copy()
    train_df["hdbscan_cluster"] = cluster_labels

    # If HDBSCAN marks noise as -1, keep it, but assign it later based on severity
    cluster_summary = (
        train_df.groupby("hdbscan_cluster")[RISK_SCORE_COL]
        .mean()
        .reset_index()
        .rename(columns={RISK_SCORE_COL: "avg_risk_score"})
        .sort_values("avg_risk_score")
        .reset_index(drop=True)
    )

    # If too few clusters, fallback to quantiles of risk score
    unique_clusters = cluster_summary["hdbscan_cluster"].nunique()
    if unique_clusters < 3:
        train_df["risk_tier_3"] = pd.qcut(
            train_df[RISK_SCORE_COL].rank(method="first"),
            q=3,
            labels=[0, 1, 2]
        ).astype(int)

        # Fit nearest centroid on transformed training data to reproduce tier labels
        tier_mapper = NearestCentroid()
        tier_mapper.fit(X_train_transformed, train_df["risk_tier_3"])

        artifacts = {
            "fallback_used": True,
            "cluster_summary": cluster_summary.to_dict(orient="records"),
            "n_clusters_found": int(unique_clusters),
        }
        return train_df, unsup_preprocessor, clusterer, tier_mapper, artifacts

    # Assign cluster ranks to 3 tiers using quantiles of avg cluster severity
    cluster_summary["severity_rank"] = np.arange(len(cluster_summary))
    cluster_summary["risk_tier_3"] = pd.qcut(
        cluster_summary["severity_rank"],
        q=3,
        labels=[0, 1, 2],
        duplicates="drop"
    ).astype(int)

    cluster_to_tier = dict(zip(cluster_summary["hdbscan_cluster"], cluster_summary["risk_tier_3"]))
    train_df["risk_tier_3"] = train_df["hdbscan_cluster"].map(cluster_to_tier)

    # Train nearest-centroid mapper from transformed features -> 3-tier labels
    tier_mapper = NearestCentroid()
    tier_mapper.fit(X_train_transformed, train_df["risk_tier_3"])

    artifacts = {
        "fallback_used": False,
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "n_clusters_found": int(unique_clusters),
        "cluster_to_tier": {str(k): int(v) for k, v in cluster_to_tier.items()}
    }

    return train_df, unsup_preprocessor, clusterer, tier_mapper, artifacts


def assign_tiers_to_new_data(X_raw, unsup_preprocessor, tier_mapper):
    X_transformed = unsup_preprocessor.transform(X_raw)
    tiers = tier_mapper.predict(X_transformed)
    return tiers


def collapse_to_binary_high_risk(df):
    df = df.copy()
    df["high_risk"] = (df["risk_tier_3"] == 2).astype(int)
    return df


# =========================
# SUPERVISED MODELING
# =========================
def build_supervised_pipeline(preprocessor, model):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def tune_logistic_regression(X_train, y_train, X_val, y_val, preprocessor):
    candidate_params = [
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ]

    best_model = None
    best_params = None
    best_f1 = -np.inf
    best_auc = -np.inf
    tuning_rows = []

    for params in candidate_params:
        model = build_supervised_pipeline(
            preprocessor,
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
                C=params["C"]
            )
        )

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)

        tuning_rows.append({
            "model": "logistic_regression",
            "params": str(params),
            "val_f1": f1,
            "val_auc_roc": auc
        })

        if f1 > best_f1:
            best_f1 = f1
            best_auc = auc
            best_model = model
            best_params = params

    return best_model, best_params, best_f1, best_auc, tuning_rows


def tune_xgboost(X_train, y_train, X_val, y_val, preprocessor):
    candidate_params = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1},
    ]

    best_model = None
    best_params = None
    best_f1 = -np.inf
    best_auc = -np.inf
    tuning_rows = []

    for params in candidate_params:
        model = build_supervised_pipeline(
            preprocessor,
            XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                objective="binary:logistic",
                eval_metric="logloss"
            )
        )

        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)

        tuning_rows.append({
            "model": "xgboost",
            "params": str(params),
            "val_f1": f1,
            "val_auc_roc": auc
        })

        if f1 > best_f1:
            best_f1 = f1
            best_auc = auc
            best_model = model
            best_params = params

    return best_model, best_params, best_f1, best_auc, tuning_rows


def save_model(model, filename):
    joblib.dump(model, os.path.join(MODEL_DIR, filename))


# =========================
# MAIN
# =========================
def main():
    print("Starting train.py workflow...")
    make_dirs()

    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)

    print("Sorting chronologically...")
    df = sort_by_time(df)

    print("Creating chronological train/val/test split...")
    df = create_time_splits(df)
    save_split_index(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train_raw = train_df[FEATURES].copy()
    X_val_raw = val_df[FEATURES].copy()
    X_test_raw = test_df[FEATURES].copy()

    print("Fitting HDBSCAN on training set only and deriving 3 tiers...")
    train_df, unsup_preprocessor, clusterer, tier_mapper, hdbscan_artifacts = fit_hdbscan_and_create_3_tiers(
        X_train_raw, train_df, FEATURES
    )

    print("Assigning 3-tier labels to validation and test using training-derived mapper...")
    val_df["risk_tier_3"] = assign_tiers_to_new_data(X_val_raw, unsup_preprocessor, tier_mapper)
    test_df["risk_tier_3"] = assign_tiers_to_new_data(X_test_raw, unsup_preprocessor, tier_mapper)

    print("Collapsing 3 tiers to binary high_risk...")
    train_df = collapse_to_binary_high_risk(train_df)
    val_df = collapse_to_binary_high_risk(val_df)
    test_df = collapse_to_binary_high_risk(test_df)

    # Save labeled data snapshot for reproducibility/debugging
    labeled_df = pd.concat([train_df, val_df, test_df], axis=0).sort_values("row_id")
    labeled_df.to_csv("Data/week3_labeled_with_tiers.csv", index=False)

    X_train = train_df[FEATURES].copy()
    y_train = train_df["high_risk"].copy()

    X_val = val_df[FEATURES].copy()
    y_val = val_df["high_risk"].copy()

    numerical_features, categorical_features = infer_feature_types(X_train, FEATURES)
    supervised_preprocessor = build_supervised_preprocessor(numerical_features, categorical_features)

    print("Tuning Logistic Regression on validation set only...")
    lr_model, lr_params, lr_val_f1, lr_val_auc, lr_rows = tune_logistic_regression(
        X_train, y_train, X_val, y_val, supervised_preprocessor
    )

    print("Tuning XGBoost on validation set only...")
    xgb_model, xgb_params, xgb_val_f1, xgb_val_auc, xgb_rows = tune_xgboost(
        X_train, y_train, X_val, y_val, supervised_preprocessor
    )

    print("Saving trained supervised models...")
    save_model(lr_model, "model_1.pkl")
    save_model(xgb_model, "model_2.pkl")

    print("Saving HDBSCAN/tiering artifacts...")
    joblib.dump(unsup_preprocessor, os.path.join(ARTIFACT_DIR, "hdbscan_preprocessor.pkl"))
    joblib.dump(clusterer, os.path.join(ARTIFACT_DIR, "hdbscan_clusterer.pkl"))
    joblib.dump(tier_mapper, os.path.join(ARTIFACT_DIR, "tier_mapper.pkl"))

    tuning_df = pd.DataFrame(lr_rows + xgb_rows)
    tuning_df.to_csv("training_tuning_results.csv", index=False)

    metadata = {
        "data_path": DATA_PATH,
        "date_column": DATE_COL,
        "risk_score_column_for_cluster_ranking": RISK_SCORE_COL,
        "features": FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "train_fraction": TRAIN_FRAC,
        "val_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,
        "hdbscan_artifacts": hdbscan_artifacts,
        "binary_high_risk_definition": "risk_tier_3 == 2",
        "logistic_regression_best_params": lr_params,
        "logistic_regression_val_f1": lr_val_f1,
        "logistic_regression_val_auc_roc": lr_val_auc,
        "xgboost_best_params": xgb_params,
        "xgboost_val_f1": xgb_val_f1,
        "xgboost_val_auc_roc": xgb_val_auc
    }

    with open("training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print("Saved outputs:")
    print("- Data/splits_indexed.csv")
    print("- Data/week3_labeled_with_tiers.csv")
    print("- Models/model_1.pkl")
    print("- Models/model_2.pkl")
    print("- Models/hdbscan_preprocessor.pkl")
    print("- Models/hdbscan_clusterer.pkl")
    print("- Models/tier_mapper.pkl")
    print("- training_tuning_results.csv")
    print("- training_metadata.json")


if __name__ == "__main__":
    main()