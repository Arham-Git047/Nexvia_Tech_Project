import io
import os
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

# Isolate local test artefacts from any default /data locations.
os.environ.setdefault("CASE_STORAGE_DIR", "/tmp/medical_imaging_test/cases")
os.environ.setdefault("CASE_REGISTRY_PATH", "/tmp/medical_imaging_test/cases.json")
os.environ.setdefault("AUDIT_LOG_PATH", "/tmp/medical_imaging_test/audit.jsonl")
os.environ.setdefault("MODEL_PATH", "/tmp/no-medical-checkpoint-expected.pth")

from app import main

client = TestClient(main.app)


def tiny_png() -> bytes:
    image = Image.new("RGB", (64, 64), color=(20, 20, 20))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def test_health_is_available_without_checkpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] is False
    assert body["clinical_use"] is False


def test_model_status_explains_fail_closed_behavior():
    response = client.get("/model-status")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "disabled" in body["message"].lower() or "not configured" in body["message"].lower()


def test_upload_fails_closed_without_a_trained_model():
    response = client.post("/analyze", files={"file": ("tiny.png", tiny_png(), "image/png")})
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].lower() or "model" in response.json()["detail"].lower()


def test_rejects_non_image_content_type():
    response = client.post("/analyze", files={"file": ("scan.dcm", b"not a dicom", "application/dicom")})
    assert response.status_code == 415


def test_review_route_returns_404_for_unknown_case():
    response = client.patch("/cases/not-a-case/review", json={"decision": "approved", "reviewer_id": "test-reviewer"})
    assert response.status_code == 404


def test_review_route_records_decision_and_prevents_double_review(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REGISTRY_PATH", tmp_path / "cases.json")
    main._CASES["test-case-123"] = {
        "case_id": "test-case-123", "status": "awaiting_review", "created_at_utc": "2026-10-23T00:00:00+00:00",
        "clinical_use": False, "findings": [], "review": None,
    }
    response = client.patch("/cases/test-case-123/review", json={"decision": "approved", "reviewer_id": "reviewer-1", "note": "Prototype workflow test"})
    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert response.json()["review"]["reviewer_id"] == "reviewer-1"
    second = client.patch("/cases/test-case-123/review", json={"decision": "rejected", "reviewer_id": "reviewer-1"})
    assert second.status_code == 409
    main._CASES.pop("test-case-123", None)
