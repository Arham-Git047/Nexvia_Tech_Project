"""Train a leakage-aware IEEE-CIS XGBoost baseline.

Run from the project root:
    python src/training/train_xgboost_baseline.py

Expected files:
    data/raw/train_transaction.csv
    data/raw/train_identity.csv

The split is chronological (70% train, 15% validation, 15% test). The decision
threshold is selected on validation data only. The test set is used once for
final reporting.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "baseline"
REPORT_DIR = ROOT / "reports" / "models"
TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
RANDOM_STATE = 42

# These are identifiers/categories, not continuous numeric measurements.
CATEGORICAL_ID_FIELDS = {
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
}
CATEGORICAL_ID_FIELDS.update(f"id_{i:02d}" for i in range(12, 39))


def load_dataset() -> pd.DataFrame:
    if not TRANSACTION_FILE.exists() or not IDENTITY_FILE.exists():
        raise FileNotFoundError(
            "Dataset files not found. Place train_transaction.csv and "
            f"train_identity.csv in: {RAW_DIR}"
        )

    print("[1/7] Loading IEEE-CIS transaction and identity CSVs...", flush=True)
    transactions = pd.read_csv(TRANSACTION_FILE, low_memory=False)
    identities = pd.read_csv(IDENTITY_FILE, low_memory=False)

    required = {"TransactionID", "TransactionDT", "isFraud"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Transaction file is missing required columns: {sorted(missing)}")
    if identities["TransactionID"].duplicated().any():
        raise ValueError("Identity file has duplicate TransactionID values; refusing an ambiguous merge.")

    print("[2/7] Joining identity features (missing identity is retained as missing)...", flush=True)
    data = transactions.merge(
        identities,
        how="left",
        on="TransactionID",
        sort=False,
        validate="one_to_one",
    )
    del transactions, identities

    data["isFraud"] = pd.to_numeric(data["isFraud"], errors="raise").astype("int8")
    data["TransactionDT"] = pd.to_numeric(data["TransactionDT"], errors="coerce")
    data = data.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    return data


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Choose an F1-maximizing threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "average_precision_pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "flagged_count": int(predictions.sum()),
        "actual_fraud_count": int(np.sum(y_true == 1)),
    }


def main() -> None:
    warnings.filterwarnings("default", category=UserWarning)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    row_count = len(data)
    train_end = int(row_count * 0.70)
    val_end = int(row_count * 0.85)
    if min(train_end, val_end - train_end, row_count - val_end) <= 0:
        raise ValueError("Not enough rows to create train/validation/test splits.")

    # Keep test labels separate from the feature matrix. TransactionID is used
    # only to identify predictions in the report, never as a model feature.
    labels = data["isFraud"].to_numpy(dtype="int8")
    transaction_ids = data["TransactionID"].to_numpy()
    transaction_dt = data["TransactionDT"].to_numpy()
    excluded = {"isFraud", "TransactionID"}
    feature_columns = [c for c in data.columns if c not in excluded]
    X = data[feature_columns].copy()
    del data

    categorical_columns = [
        c for c in feature_columns
        if c in CATEGORICAL_ID_FIELDS or not pd.api.types.is_numeric_dtype(X[c].dtype)
    ]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    # Normalize categories to strings and non-finite numerics to missing values.
    print("[3/7] Preparing features without fitting on validation/test rows...", flush=True)
    for col in categorical_columns:
        X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)
    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)

    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = labels[:train_end], labels[train_end:val_end], labels[val_end:]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, dtype=np.float32)),
    ])
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    del X, X_train, X_val, X_test

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    if positives == 0:
        raise ValueError("Training split contains no fraud cases; check the time split and dataset.")
    scale_pos_weight = negatives / positives

    print("[4/7] Training class-weighted XGBoost (early stopping uses validation only)...", flush=True)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=450,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=35,
    )
    model.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=25)

    print("[5/7] Selecting a decision threshold on the validation set...", flush=True)
    val_prob = model.predict_proba(X_val_t)[:, 1]
    threshold, val_best_f1 = choose_threshold(y_val, val_prob)
    val_metrics = evaluate(y_val, val_prob, threshold)
    val_metrics["f1_at_threshold_search"] = val_best_f1

    print("[6/7] Evaluating the untouched chronological test set...", flush=True)
    test_prob = model.predict_proba(X_test_t)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    top_features = pd.DataFrame({"feature": feature_names, "importance": importances})
    top_features.sort_values("importance", ascending=False).head(50).to_csv(
        REPORT_DIR / "xgboost_top_features.csv", index=False
    )

    predictions = (test_prob >= threshold).astype("int8")
    pd.DataFrame({
        "TransactionID": transaction_ids[val_end:],
        "TransactionDT": transaction_dt[val_end:],
        "isFraud_actual": y_test,
        "fraud_probability": test_prob,
        "isFraud_predicted": predictions,
    }).to_csv(REPORT_DIR / "xgboost_test_predictions.csv", index=False)

    precision, recall, _ = precision_recall_curve(y_test, test_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("IEEE-CIS XGBoost Baseline — Chronological Test Set")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "xgboost_test_precision_recall_curve.png", dpi=160)
    plt.close()

    metrics = {
        "dataset": "IEEE-CIS Fraud Detection",
        "split_strategy": "chronological by TransactionDT; 70% train / 15% validation / 15% test",
        "rows": row_count,
        "fraud_rate_percent_full_dataset": round(float(labels.mean() * 100), 4),
        "split_rows": {"train": train_end, "validation": val_end - train_end, "test": row_count - val_end},
        "split_fraud_counts": {
            "train": int(y_train.sum()),
            "validation": int(y_val.sum()),
            "test": int(y_test.sum()),
        },
        "original_feature_columns": len(feature_columns),
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "encoded_feature_columns": int(len(feature_names)),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators - 1)),
        "validation_metrics_threshold_selected_here": val_metrics,
        "test_metrics_threshold_fixed_from_validation": test_metrics,
        "notes": [
            "TransactionID is excluded as a predictor.",
            "Identity rows are left-joined; missing identity attributes are imputed/encoded.",
            "Preprocessing is fit only on the training split.",
            "Decision threshold is selected on validation data, not the test set.",
            "This is a tabular baseline; causal online velocity features and the GNN are later phases.",
        ],
    }
    with open(REPORT_DIR / "xgboost_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)

    joblib.dump({
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "decision_threshold": threshold,
    }, MODEL_DIR / "xgboost_baseline.joblib", compress=3)

    print("\n=== BASELINE RESULTS ===")
    print(json.dumps({
        "validation_pr_auc_average_precision": round(val_metrics["average_precision_pr_auc"], 5),
        "test_pr_auc_average_precision": round(test_metrics["average_precision_pr_auc"], 5),
        "test_roc_auc": round(test_metrics["roc_auc"], 5),
        "validation_selected_threshold": round(threshold, 5),
        "test_precision": round(test_metrics["precision"], 5),
        "test_recall": round(test_metrics["recall"], 5),
        "test_f1": round(test_metrics["f1"], 5),
        "test_confusion_matrix_labels_0_1": test_metrics["confusion_matrix_labels_0_1"],
        "best_iteration": metrics["best_iteration"],
        "reports_folder": str(REPORT_DIR),
        "saved_model": str(MODEL_DIR / "xgboost_baseline.joblib"),
    }, indent=2))
    print("\nBaseline complete. Do not report metrics as achieved until this script finishes successfully.")


if __name__ == "__main__":
    main()
