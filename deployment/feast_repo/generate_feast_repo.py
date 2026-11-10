"""Run this once from Colab after extracting the bundle into the project.
Creates a Feast file source, entity/view definitions, and a label-free Parquet
snapshot for the 1,000-transaction replay. Online features are then materialized
by deployment/feast_repo/bootstrap.py before the API starts.
"""
from pathlib import Path
import json
import os
import joblib
import pandas as pd
import numpy as np

PROJECT_DIR = Path(os.getenv(
    "PROJECT_DIR", "/content/drive/MyDrive/AIML-01/fraud-detection-platform"
))
MODEL_DIR = PROJECT_DIR / "models" / "baseline"
DATA_DIR = PROJECT_DIR / "data" / "processed"
FEAST_DIR = PROJECT_DIR / "deployment" / "feast_repo"
FEAST_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

schema = joblib.load(MODEL_DIR / "xgboost_feature_schema.joblib")
feature_columns = list(schema["feature_columns"])
categorical = set(schema.get("categorical_columns", []))
events_path = DATA_DIR / "stream_replay.jsonl"
if not events_path.exists():
    raise FileNotFoundError(f"Run Cell 13 first; missing {events_path}")

rows = []
with events_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        event = json.loads(line)
        features = event["features"]
        row = {
            "transaction_id": int(event["transaction_id"]),
            "event_timestamp": pd.Timestamp("2017-01-01", tz="UTC") + pd.to_timedelta(float(event["transaction_dt"]), unit="s"),
        }
        for col in feature_columns:
            value = features.get(col)
            if col in categorical:
                row[col] = None if value is None else str(value)
            else:
                try:
                    row[col] = np.float32(value) if value is not None else np.nan
                except (TypeError, ValueError):
                    row[col] = np.nan
        rows.append(row)

frame = pd.DataFrame(rows)
frame = frame.sort_values("event_timestamp")
parquet_path = DATA_DIR / "feast_features.parquet"
frame.to_parquet(parquet_path, index=False)

# Generate a repository definition matching the model schema.
field_lines = []
for col in feature_columns:
    dtype = "String" if col in categorical else "Float32"
    field_lines.append(f"    Field(name={col!r}, dtype={dtype}),")
fields = "\n".join(field_lines)

repo_text = f'''import os\nfrom datetime import timedelta\nfrom feast import Entity, FeatureView, Field, FileSource\nfrom feast.types import Float32, String\n\ntransaction = Entity(name="transaction_id", join_keys=["transaction_id"])\n\nsource = FileSource(\n    name="transaction_feature_source",\n    path=os.getenv("FEAST_SOURCE_PATH", "/data/processed/feast_features.parquet"),\n    timestamp_field="event_timestamp",\n)\n\ntransaction_features = FeatureView(\n    name="transaction_features",\n    entities=[transaction],\n    ttl=timedelta(days=3650),\n    schema=[\n{fields}\n    ],\n    source=source,\n    online=True,\n)\n'''
(FEAST_DIR / "feature_repo.py").write_text(repo_text, encoding="utf-8")

store_yaml = '''project: fraud_features\nprovider: local\nregistry: /runtime/feast_registry.db\nonline_store:\n  type: sqlite\n  path: /runtime/feast_online.db\n'''
(FEAST_DIR / "feature_store.yaml").write_text(store_yaml, encoding="utf-8")

schema_json = {
    "entity": "transaction_id",
    "feature_view": "transaction_features",
    "features": feature_columns,
    "categorical_features": sorted(categorical),
    "entity_rows": len(frame),
    "source_path": str(parquet_path),
}
(FEAST_DIR / "feast_schema_summary.json").write_text(
    json.dumps(schema_json, indent=2), encoding="utf-8"
)

print("Feast repo generated:", FEAST_DIR)
print("Offline source:", parquet_path)
print("Rows:", len(frame), "features:", len(feature_columns))
