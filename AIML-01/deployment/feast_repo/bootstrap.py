"""Apply Feast definitions and materialize the replay feature rows online."""
from pathlib import Path
from datetime import timedelta
import os
import sys
import pandas as pd
from feast import FeatureStore

REPO = Path(os.getenv("FEAST_REPO_PATH", "/feast_repo"))
SOURCE = Path(os.getenv("FEAST_SOURCE_PATH", "/data/processed/feast_features.parquet"))
sys.path.insert(0, str(REPO))

from feature_repo import transaction, transaction_features  # noqa: E402

if not SOURCE.exists():
    raise FileNotFoundError(f"Feast source Parquet file missing: {SOURCE}")

data = pd.read_parquet(SOURCE, columns=["event_timestamp"])
if data.empty:
    raise RuntimeError("Feast source has no feature rows")

start_date = pd.to_datetime(data["event_timestamp"], utc=True).min().to_pydatetime()
end_date = pd.to_datetime(data["event_timestamp"], utc=True).max().to_pydatetime() + timedelta(seconds=2)

store = FeatureStore(repo_path=str(REPO))
store.apply([transaction, transaction_features])
store.materialize(start_date=start_date, end_date=end_date)
print(f"Feast bootstrap complete. Materialized {len(data)} transaction rows.")
print("Registry:", REPO / "feature_store.yaml")
