"""Evidently report plus a transparent PSI-based retraining-request trigger."""
from pathlib import Path
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/project"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/baseline"))
DATA_DIR = Path(os.getenv("DATA_PROCESSED", "/data/processed"))
REPORT_DIR = Path(os.getenv("REPORT_MODELS", "/reports/models"))
RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", "/runtime"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = DATA_DIR / "baseline_reference.csv"
EVENT_PATH = DATA_DIR / "stream_replay.jsonl"
SCHEMA_PATH = MODEL_DIR / "xgboost_feature_schema.joblib"


def psi(ref, cur, bins=10):
    ref = pd.Series(ref).dropna()
    cur = pd.Series(cur).dropna()
    if len(ref) < 10 or len(cur) < 10:
        return 0.0
    eps = 1e-6
    if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
        try:
            cuts = np.unique(np.quantile(ref.astype(float), np.linspace(0, 1, bins + 1)))
            if len(cuts) < 3:
                cuts = np.array([-np.inf, float(ref.min()) - 1e-9, float(ref.max()) + 1e-9, np.inf])
            else:
                cuts[0], cuts[-1] = -np.inf, np.inf
            r = pd.cut(ref.astype(float), cuts, include_lowest=True).value_counts(normalize=True, sort=False).to_numpy()
            c = pd.cut(cur.astype(float), cuts, include_lowest=True).value_counts(normalize=True, sort=False).to_numpy()
        except Exception:
            return 0.0
    else:
        top = ref.astype(str).value_counts().head(25).index.tolist()
        rser = ref.astype(str).where(ref.astype(str).isin(top), "__OTHER__")
        cser = cur.astype(str).where(cur.astype(str).isin(top), "__OTHER__")
        labels = list(dict.fromkeys(top + ["__OTHER__"]))
        r = rser.value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
        c = cser.value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main():
    if not REFERENCE_PATH.exists() or not EVENT_PATH.exists():
        raise FileNotFoundError(
            f"Expected {REFERENCE_PATH} and {EVENT_PATH}. Run the Colab artifact setup cell first."
        )

    reference = pd.read_csv(REFERENCE_PATH, low_memory=False)
    events = []
    with EVENT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                events.append(obj.get("features", {}))
    current = pd.DataFrame(events)

    if SCHEMA_PATH.exists():
        schema = joblib.load(SCHEMA_PATH)
        categorical = set(schema.get("categorical_columns", []))
    else:
        categorical = set()

    common = [c for c in reference.columns if c in current.columns]
    reference = reference[common].copy()
    current = current[common].copy()
    for col in categorical.intersection(common):
        reference[col] = reference[col].astype("string")
        current[col] = current[col].astype("string")

    drift_scores = []
    for col in common:
        drift_scores.append({"feature": col, "psi": psi(reference[col], current[col])})
    drift_table = pd.DataFrame(drift_scores).sort_values("psi", ascending=False)
    drift_table.to_csv(REPORT_DIR / "feature_drift_psi.csv", index=False)

    # Use Evidently to produce the required detailed HTML report.
    evidently_ok = False
    evidently_error = None
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        snapshot.save_html(str(REPORT_DIR / "evidently_data_drift_report.html"))
        evidently_ok = True
    except Exception as exc:
        evidently_error = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Evidently report generation failed: {evidently_error}")

    # Trigger only requests a labelled retraining workflow; it never trains
    # on unlabelled stream data or silently replaces the approved model.
    drifted = drift_table[drift_table["psi"] >= 0.25]
    drift_share = len(drifted) / max(len(common), 1)
    requested = drift_share >= float(os.getenv("DRIFTED_FEATURE_SHARE_THRESHOLD", "0.10"))

    result = {
        "monitor": "Evidently DataDriftPreset plus PSI trigger",
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_compared": int(len(common)),
        "features_with_psi_ge_0_25": int(len(drifted)),
        "drifted_feature_share": float(drift_share),
        "trigger_threshold_share": float(os.getenv("DRIFTED_FEATURE_SHARE_THRESHOLD", "0.10")),
        "retraining_requested": bool(requested),
        "evidently_report_generated": evidently_ok,
        "evidently_error": evidently_error,
        "note": "Retraining requires newly reviewed labels and offline validation before any model replacement.",
    }
    out = REPORT_DIR / "drift_trigger_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    trigger_file = RUNTIME_DIR / "retrain_trigger.json"
    trigger_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("PSI report:", REPORT_DIR / "feature_drift_psi.csv")
    if evidently_ok:
        print("Evidently HTML report:", REPORT_DIR / "evidently_data_drift_report.html")
    print("Trigger record:", trigger_file)

if __name__ == "__main__":
    main()
