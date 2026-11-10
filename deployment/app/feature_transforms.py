from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/baseline"))
REPORT_DIR = Path(os.getenv("REPORT_MODELS", "/reports/models"))

class FeatureTransform:
    def __init__(self):
        self.xgb_schema = joblib.load(MODEL_DIR / "xgboost_feature_schema.joblib")
        self.graph_schema = joblib.load(MODEL_DIR / "graph_feature_schema.joblib")
        self.preprocess = joblib.load(MODEL_DIR / "graph_preprocessing.joblib")
        self.xgb_columns = list(self.xgb_schema["feature_columns"])
        self.cat_columns = set(self.xgb_schema.get("categorical_columns", []))
        self.cat_levels = self.xgb_schema.get("categorical_levels", {})
        self.graph_columns = list(self.graph_schema["selected_features"])
        self.graph_specs = self.graph_schema["feature_specs"]
        self.medians = np.asarray(self.graph_schema["training_medians"], dtype=np.float32)
        self.scaler = self.graph_schema["scaler"]

    @staticmethod
    def _clean(value: Any):
        if value is None:
            return None
        if isinstance(value, (dict, list, tuple)):
            return str(value)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, np.generic):
            return value.item()
        return value

    def xgb_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        values = {}
        for col in self.xgb_columns:
            value = self._clean(features.get(col))
            if col in self.cat_columns:
                levels = self.cat_levels.get(col, [])
                # Training schema used pandas String/category values; normalize
                # incoming JSON values to matching category labels.
                if value is not None:
                    value = str(value)
                    normalized_levels = [str(x) for x in levels]
                    values[col] = pd.Categorical(
                        [value], categories=normalized_levels
                    )
                else:
                    values[col] = pd.Categorical(
                        [None], categories=[str(x) for x in levels]
                    )
            else:
                if value is None:
                    values[col] = [np.nan]
                else:
                    try:
                        values[col] = [float(value)]
                    except (TypeError, ValueError):
                        values[col] = [np.nan]
        frame = pd.DataFrame(values, columns=self.xgb_columns)
        return frame

    def graph_vector(self, features: dict[str, Any]) -> np.ndarray:
        out = []
        for col in self.graph_columns:
            value = self._clean(features.get(col))
            spec = self.graph_specs[col]
            mode = spec["mode"]

            if value is None:
                number = np.nan
            elif mode == "numeric":
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = np.nan
            else:
                categories = spec.get("categories", [])
                # Match on string-normalized labels to survive JSON conversion.
                mapping = {str(cat): idx for idx, cat in enumerate(categories)}
                number = mapping.get(str(value), np.nan)

            out.append(number)

        vector = np.asarray(out, dtype=np.float32)
        missing = ~np.isfinite(vector)
        vector[missing] = self.medians[missing]
        scaled = self.scaler.transform(vector.reshape(1, -1))[0]
        return np.clip(scaled, -10, 10).astype(np.float32)


def load_serving_config():
    path = REPORT_DIR / "serving_config.json"
    if not path.exists():
        path = REPORT_DIR / "final_hybrid_config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
