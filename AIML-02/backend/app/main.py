from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from .audit import append_audit
from .model_service import ChestXrayService, ModelUnavailable

APP_VERSION = "0.1.0-research-prototype"
DATA_DIR = Path(os.getenv("CASE_STORAGE_DIR", "/data/cases"))
REGISTRY_PATH = Path(os.getenv("CASE_REGISTRY_PATH", "/data/cases.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg"}

app = FastAPI(
    title="AI-Assisted Medical Imaging Review API",
    version=APP_VERSION,
    description=(
        "Educational research prototype. It is not a medical device and must not be used "
        "for diagnosis, triage, or treatment decisions. Model output requires independent review."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type"],
)
model_service = ChestXrayService(os.getenv("MODEL_PATH", "/models/densenet121_nih.pth"))
_CASES: dict[str, dict] = {}
_CASE_LOCK = Lock()


class ReviewBody(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer_id: str = Field(min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=1000)


def _load_registry() -> None:
    try:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            _CASES.update(raw)
    except (OSError, json.JSONDecodeError):
        pass


def _persist_registry() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = REGISTRY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(_CASES, indent=2), encoding="utf-8")
    temp_path.replace(REGISTRY_PATH)


_load_registry()


def _public_case(case: dict) -> dict:
    # Do not expose absolute storage paths in API responses.
    return {k: v for k, v in case.items() if not k.endswith("_path")}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "model_available": model_service.available,
        "model_version": model_service.version,
        "explainability_available": model_service.available,
        "clinical_use": False,
    }


@app.get("/model-status")
def model_status():
    return {
        "available": model_service.available,
        "model_version": model_service.version,
        "status": "ready" if model_service.available else "not_configured",
        "message": (
            "A trained research checkpoint is loaded." if model_service.available
            else "Inference is disabled until a validated training checkpoint is supplied. No prediction will be fabricated."
        ),
        "clinical_use": False,
    }


@app.get("/cases")
def list_cases():
    with _CASE_LOCK:
        items = sorted(_CASES.values(), key=lambda x: x.get("created_at_utc", ""), reverse=True)
        return {"items": [_public_case(x) for x in items], "count": len(items)}


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    with _CASE_LOCK:
        case = _CASES.get(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return _public_case(case)


@app.post("/analyze", status_code=201)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG chest X-ray export")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="File is not a valid PNG/JPEG image") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        output = model_service.predict(raw)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid leaking model internals or patient data into response messages.
        raise HTTPException(status_code=500, detail=f"Inference failed safely ({type(exc).__name__})") from exc

    case_id = str(uuid.uuid4())
    case_dir = DATA_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    input_path = case_dir / "input.png"
    Image.open(io.BytesIO(raw)).convert("RGB").save(input_path, format="PNG")
    heatmap_path = case_dir / "gradcam.png"
    heatmap_path.write_bytes(output.pop("heatmap_png"))
    case = {
        "case_id": case_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_sha256": digest,
        "status": "awaiting_review",
        "source": "manual_upload",
        "findings": output["findings"],
        "top_finding": output["top_finding"],
        "top_score": output["top_score"],
        "model_version": output["model_version"],
        "model_kind": output["model_kind"],
        "clinical_use": False,
        "image_path": str(input_path),
        "heatmap_path": str(heatmap_path),
        "heatmap_data_url": "data:image/png;base64," + base64.b64encode(heatmap_path.read_bytes()).decode("ascii"),
        "review": None,
    }
    with _CASE_LOCK:
        _CASES[case_id] = case
        _persist_registry()
    append_audit("case_analyzed", case_id, {"source": "manual_upload", "model_version": output["model_version"]})
    return _public_case(case)


@app.patch("/cases/{case_id}/review")
def review_case(case_id: str, body: ReviewBody):
    with _CASE_LOCK:
        case = _CASES.get(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        if case.get("status") != "awaiting_review":
            raise HTTPException(status_code=409, detail="Case has already been reviewed")
        reviewed_at = datetime.now(timezone.utc).isoformat()
        case["status"] = body.decision
        case["review"] = {
            "decision": body.decision,
            "reviewer_id": body.reviewer_id.strip(),
            "note": body.note,
            "reviewed_at_utc": reviewed_at,
        }
        _persist_registry()
    append_audit("case_reviewed", case_id, {
        "decision": body.decision,
        "reviewer_id": body.reviewer_id.strip(),
        "note_present": bool(body.note),
    })
    return _public_case(case)
