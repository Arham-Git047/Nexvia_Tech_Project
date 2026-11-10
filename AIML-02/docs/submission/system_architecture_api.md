# System Architecture and API Documentation

**Project:** MedVision Review — AI-Assisted Medical Imaging Review Prototype  
**Developer:** Arham Kelkar  
**Institution:** Dr. D.Y. Patil School of Science and Technology  
**Status date:** 10 November 2026

> Educational/research prototype only. Not for clinical use.

## 1. Purpose and scope

The local system accepts a de-identified PNG/JPEG chest X-ray export, validates it, runs a DenseNet121 14-label classifier when a loadable checkpoint exists, generates a Grad-CAM activation overlay, creates a review case, and records reviewer decisions in a local audit log. A separate optional Orthanc worker can poll DICOM study changes and forward rendered instance previews. A MONAI 3D U-Net training scaffold is included but was not trained in the verified run.

## 2. Components

| Component | Responsibility | Verified status |
|---|---|---|
| React/Vite frontend | Image upload, image preview, findings, heatmap and review actions | Built and tested locally at localhost:8080 |
| FastAPI | Upload validation, model state, inference, case and review endpoints | Docker container healthy; API workflow verified |
| DenseNet121 service | Fourteen-label output scores | Rapid ChestMNIST checkpoint loaded successfully |
| Grad-CAM | Activation overlay from DenseNet denseblock4 | Generation verified; overlay shown in UI |
| Review registry | Local JSON case state | Review action completed in UI |
| Audit helper | Appends UTC event entries to JSONL | Source included; review workflow action completed |
| Orthanc + worker | Optional DICOM store and preview polling | Configuration included; end-to-end not verified |
| MONAI 3D U-Net | NIfTI segmentation training path | Scaffold only; not trained/evaluated |

## 3. Data flow

1. User selects a de-identified PNG/JPEG file in the React UI.
2. Frontend posts multipart form data to `POST /analyze`.
3. API checks MIME type, file size and image decoding.
4. `ChestXrayService` loads `models/densenet121_nih.pth`; missing or invalid checkpoint fails closed.
5. DenseNet121 returns 14 sigmoid scores; Grad-CAM generates an activation overlay for the highest-scoring label.
6. The case and image/heatmap paths are stored under `runtime/`; a JSON registry tracks review status.
7. Reviewer submits approved/rejected, reviewer identifier and optional note.
8. An audit JSONL record captures the case-created and case-reviewed events.

## 4. API contract

Base URL: `http://localhost:8000` (loopback-only local demo)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | App status, checkpoint and explainability availability, clinical-use flag |
| GET | `/model-status` | Detailed checkpoint loading status |
| POST | `/analyze` | PNG/JPEG upload, model inference and Grad-CAM |
| GET | `/cases` | List case queue |
| GET | `/cases/{case_id}` | Retrieve one case |
| PATCH | `/cases/{case_id}/review` | Record reviewer decision |

### Analyze upload

`POST /analyze` accepts multipart field `file`; accepted MIME types are `image/png`, `image/jpeg` and `image/jpg`. Default upload cap: 15 MiB. Responses include a case ID, 14 findings, top finding/score, model metadata, heatmap data URL and `clinical_use: false`. Errors: 400 invalid/empty image, 413 too large, 415 unsupported content type, 503 unavailable checkpoint, 500 safely handled inference error.

### Review request

```json
{
  "decision": "approved",
  "reviewer_id": "reviewer-01",
  "note": "Research prototype workflow test only"
}
```

Only `approved` or `rejected` is accepted. A case can be reviewed once; duplicate review returns 409. Reviewer ID is an audit field, not an authenticated identity.

## 5. Deployment

`compose.yaml` runs `api` and `frontend` by default. Ports bind to `127.0.0.1`. Optional `dicom` profile adds Orthanc and the polling worker. Local deployment was verified for API and frontend. Orthanc end-to-end ingestion is not yet verified. Do not expose the current compose configuration publicly: no production authentication, TLS, RBAC or hardened secrets management is configured.

## 6. Known limits

- Current checkpoint was trained on a small 800/200/200 ChestMNIST subset, not full-resolution NIH data.
- Scores are not calibrated clinical probabilities.
- Grad-CAM is activation visualization, not a lesion mask or causal explanation.
- The review field is not authentication.
- Orthanc workflow and MONAI segmentation were not verified.
- No patient-level, external or clinical validation was performed.
