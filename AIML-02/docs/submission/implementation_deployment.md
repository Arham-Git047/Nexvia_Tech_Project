# Implementation and Local Deployment Guide — AIML-02

## Prerequisites

- Windows 10/11 with Docker Desktop and Docker Engine running.
- At least enough free disk/RAM for Python/PyTorch container image layers; avoid concurrent heavy model training on a resource-constrained laptop.
- Project folder contains `compose.yaml`, `backend/`, `frontend/`, and trained model checkpoint at `models/densenet121_nih.pth`.

## Start the verified local application

From the project root in PowerShell:

```powershell
docker compose config
docker compose up --build -d api frontend
docker compose ps
```

The user verified API as healthy and frontend as running. Open `http://localhost:8080`; check `http://localhost:8000/health`, `http://localhost:8000/model-status` and `http://localhost:8000/docs`.

## Train the rapid classifier (already completed in the recorded run)

See `fast_training_10_15_min.md`. It used 800/200/200 ChestMNIST samples and two CPU epochs. Do not retrain unless needed. Preserve `models/densenet121_nih.pth` and `reports/classification_metrics.json`.

## Test inference and review

1. Upload only a de-identified PNG/JPEG sample in the React UI.
2. Click Analyze image.
3. Confirm that findings and the Grad-CAM activation overlay render.
4. Enter a reviewer ID and research-only review note.
5. Choose an approve/reject workflow action and confirm the case status / audit record is updated.

The current user verified one image result and a successful review action. The observed top finding was Pneumothorax with score 0.686565 for one test image; do not call this a diagnosis or calibrated probability.

## Optional Orthanc profile

`compose.yaml` defines Orthanc and an Orthanc polling worker in the `dicom` profile:

```powershell
docker compose --profile dicom up -d --build
```

This integration was not verified end-to-end. Use only de-identified test DICOM. Do not expose the local Orthanc service publicly.

## Shutdown

```powershell
docker compose stop
```

## GitHub packaging

Include source, docs, test results, metrics, architecture diagram and the trained `models/densenet121_nih.pth` checkpoint (approximately 28.4 MB in the recorded local copy). Exclude full datasets, `.npz`, DICOM, NIfTI, runtime databases, local audit/case state, `.env`, virtual environments, `node_modules`, and local caches. This pack includes a `.gitignore` exception for `models/densenet121_nih.pth`.

## Not yet completed

MONAI segmentation training and evaluation, Orthanc end-to-end validation, external/clinical validation, security hardening and public cloud deployment are not claimed as complete.
