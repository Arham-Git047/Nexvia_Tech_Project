from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
import json
import os
import time

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.db import init_db, get_case, upsert_case, list_cases, update_case, summary
from app.feature_transforms import FeatureTransform, load_serving_config
from app.graph_runtime import OnlineGraphScorer

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/baseline"))
REPORT_MODELS = Path(os.getenv("REPORT_MODELS", "/reports/models"))

state: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    state["config"] = load_serving_config()
    state["features"] = FeatureTransform()
    xgb_model = xgb.XGBClassifier(enable_categorical=True)
    xgb_model.load_model(str(MODEL_DIR / "xgboost_ieee_cis_baseline.json"))
    state["xgb"] = xgb_model
    state["graph"] = OnlineGraphScorer()
    state["feast"] = None
    state["started_at"] = time.time()
    feast_repo = Path(os.getenv("FEAST_REPO_PATH", "/feast_repo"))
    if (feast_repo / "feature_store.yaml").exists():
        try:
            from feast import FeatureStore
            state["feast"] = FeatureStore(repo_path=str(feast_repo))
            print("Feast online feature store client initialized")
        except Exception as exc:
            print(f"Feast client unavailable; event payload fallback enabled: {type(exc).__name__}: {exc}")
    yield
    state.clear()

app = FastAPI(
    title="IEEE-CIS Real-Time Fraud Detection API",
    version="1.0.0",
    description="Hybrid XGBoost + GraphSAGE risk scoring, case management, and analyst feedback.",
    lifespan=lifespan,
)

class TransactionEvent(BaseModel):
    transaction_id: int
    transaction_dt: float | None = None
    features: dict[str, Any]
    entities: dict[str, Any] = Field(default_factory=dict)

class CaseUpdate(BaseModel):
    status: Literal["NEW", "REVIEWING", "CONFIRMED_FRAUD", "LEGITIMATE", "CLOSED"]
    analyst_label: Literal["UNREVIEWED", "FRAUD", "LEGITIMATE"]
    notes: str = ""

@app.get("/health")
def health():
    ready = all(key in state for key in ("config", "features", "xgb", "graph"))
    return {
        "status": "ok" if ready else "starting",
        "models_loaded": ready,
        "runtime_seconds": round(time.time() - state.get("started_at", time.time()), 2),
    }

@app.post("/score")
def score_transaction(event: TransactionEvent):
    if not all(key in state for key in ("config", "features", "xgb", "graph")):
        raise HTTPException(status_code=503, detail="Models are still loading")

    existing = get_case(event.transaction_id)
    if existing:
        return {
            "transaction_id": existing["transaction_id"],
            "xgboost_score": existing["xgb_score"],
            "graph_score": existing["graph_score"],
            "risk_score": existing["risk_score"],
            "threshold": existing["threshold"],
            "decision": existing["decision"],
            "status": existing["status"],
            "feature_source": existing["feature_source"],
            "duplicate_event": True,
        }

    started = time.perf_counter()
    try:
        feature_transform: FeatureTransform = state["features"]
        scoring_features = event.features
        feature_source = "event_payload"
        feast_store = state.get("feast")
        if feast_store is not None:
            try:
                feature_refs = [
                    f"transaction_features:{col}"
                    for col in feature_transform.xgb_columns
                ]
                online = feast_store.get_online_features(
                    features=feature_refs,
                    entity_rows=[{"transaction_id": int(event.transaction_id)}],
                ).to_dict()
                resolved = {}
                complete = True
                for col in feature_transform.xgb_columns:
                    candidate_keys = [
                        f"transaction_features__{col}",
                        f"transaction_features:{col}",
                        col,
                    ]
                    found = next((key for key in candidate_keys if key in online), None)
                    if found is None:
                        complete = False
                        break
                    value_list = online[found]
                    resolved[col] = value_list[0] if isinstance(value_list, (list, tuple, np.ndarray)) else value_list
                if complete:
                    scoring_features = resolved
                    feature_source = "feast_online"
            except Exception as feast_exc:
                print(f"Feast lookup failed for {event.transaction_id}; using event payload: {type(feast_exc).__name__}: {feast_exc}")

        features = feature_transform.xgb_frame(scoring_features)
        xgb_score = float(state["xgb"].predict_proba(features)[0, 1])
        graph_vector = feature_transform.graph_vector(scoring_features)
        graph_score = float(
            state["graph"].score_and_update(event.model_dump(), graph_vector)
        )
        config = state["config"]
        xgb_weight = float(config["hybrid_xgboost_weight"])
        graph_weight = float(config["hybrid_graphsage_weight"])
        threshold = float(config["hybrid_threshold"])
        risk_score = xgb_weight * xgb_score + graph_weight * graph_score
        decision = "REVIEW" if risk_score >= threshold else "ALLOW"
        latency_ms = (time.perf_counter() - started) * 1000

        record = {
            "transaction_id": event.transaction_id,
            "transaction_dt": event.transaction_dt,
            "xgb_score": xgb_score,
            "graph_score": graph_score,
            "risk_score": float(risk_score),
            "threshold": threshold,
            "decision": decision,
            "feature_source": feature_source,
            "payload": {
                "entities": event.entities,
                "feature_count": len(scoring_features),
            },
        }
        saved = upsert_case(record)
        return {
            "transaction_id": event.transaction_id,
            "xgboost_score": xgb_score,
            "graph_score": graph_score,
            "risk_score": float(risk_score),
            "threshold": threshold,
            "decision": decision,
            "status": saved["status"],
            "feature_source": feature_source,
            "latency_ms": round(latency_ms, 3),
            "duplicate_event": False,
        }
    except Exception as exc:
        # Avoid hiding model/feature failures. They are exposed as API errors
        # so the consumer can retry and the issue can be observed in logs.
        raise HTTPException(status_code=500, detail=f"Scoring failed: {type(exc).__name__}: {exc}") from exc

@app.get("/cases")
def get_cases(
    limit: int = Query(default=200, ge=1, le=5000),
    status: str | None = None,
):
    return {"cases": list_cases(limit, status)}

@app.get("/summary")
def get_summary():
    return summary()

@app.patch("/cases/{transaction_id}")
def patch_case(transaction_id: int, body: CaseUpdate):
    updated = update_case(
        transaction_id,
        body.status,
        body.analyst_label,
        body.notes,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"case": updated, "feedback_saved": True}
