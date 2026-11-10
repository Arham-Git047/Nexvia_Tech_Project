# MedVision Review — AI-Assisted Medical Imaging Review Prototype

**Student:** Arham Kelkar  
**Institution:** Dr. D.Y. Patil School of Science and Technology  
**Specialization:** AI / Machine Learning  
**Project window:** 9 October–10 November 2026

> **EDUCATIONAL / RESEARCH ONLY — NOT FOR CLINICAL USE.** This prototype does not provide medical advice or a diagnosis. Do not use it for patient care, triage, treatment, or other clinical decisions.

## 1. Project summary

MedVision Review is a local research prototype for a chest-X-ray image review workflow. It combines a React review console, FastAPI service, DenseNet121 14-label classifier, Grad-CAM activation overlay, human review controls, local case registry and JSONL audit trail. An optional Orthanc profile and worker provide a DICOM integration path. A separate MONAI 3D U-Net training entry point is included for a task-specific Medical Segmentation Decathlon dataset.

## 2. Verified implementation status (10 November 2026)

- Docker images for API and frontend built successfully on the target Windows machine.
- API container reported healthy; React/Nginx frontend ran on `http://localhost:8080`.
- DenseNet121 rapid classifier trained on CPU for 2 epochs using 800 train / 200 validation / 200 test samples from MedMNIST ChestMNIST official splits.
- Training time reported by the script: 2.637 minutes (data and pretrained-weight downloads are additional).
- Checkpoint `models/densenet121_nih.pth` was loaded by the API; `/health` reported model and explainability availability.
- One held-out sample prediction returned top finding `Pneumothorax` with score 0.686565; Grad-CAM PNG was generated (1,578 bytes in direct service smoke test).
- The React UI displayed the prediction list and Grad-CAM overlay. A clinician review action was completed for a demonstration case and recorded in the local review/audit workflow.
- The Matplotlib compatibility issue in the Grad-CAM overlay was fixed (`cm.jet`), and the patched source is included here.

These are small-subset prototype checks only. They are **not** clinical-performance evidence and the displayed scores are not calibrated diagnostic probabilities.

## 3. Dataset and rapid training choice

The rapid run used **MedMNIST ChestMNIST**, a lightweight 14-label chest-X-ray benchmark derived from NIH ChestX-ray14. ChestMNIST images are low resolution (28×28) and use official train/validation/test splits. The run sampled 800 / 200 / 200 items respectively and resized them to 128×128 for DenseNet121 input. The MedMNIST maintainers explicitly state that the datasets are not intended for clinical use; reduced resolution can remove clinically important detail. Cite MedMNIST v2 and the source dataset when reusing the data.

The original NIH full-resolution archive, ChestMNIST archive, and derived data are not committed to this repository. The fast-training script downloads ChestMNIST into the container data volume when needed.

## 4. Model training and results

Architecture: TorchVision DenseNet121, ImageNet-pretrained; feature extractor frozen; 14-output classifier head trained using weighted `BCEWithLogitsLoss` and AdamW. Two epochs completed on CPU.

| Metric | Observed result |
|---|---:|
| Macro average precision | 0.1185728 |
| Macro ROC-AUC | 0.6023526 |
| Micro precision @ 0.5 | 0.0872939 |
| Micro recall @ 0.5 | 0.6293706 |
| Micro F1 @ 0.5 | 0.1533220 |
| Train / validation / test samples | 800 / 200 / 200 |
| Epochs | 2 |
| Training time | 2.637 min (reported) |

The checkpoint selected by the script is the epoch with the best validation loss. These small-subset metrics are preliminary and should not be compared as if they were a full NIH benchmark or evidence of clinical utility.

## 5. Architecture

1. Researcher uploads a de-identified PNG/JPEG export in the React console.
2. FastAPI validates content type, size and image decoding.
3. If the DenseNet121 checkpoint loads, the service produces 14 label scores and a Grad-CAM activation overlay.
4. The result becomes a case in the local review queue.
5. A reviewer records `approved` or `rejected`, reviewer identifier and optional note.
6. Case state is stored in a local JSON registry and events are appended to a JSONL audit log.
7. The optional Orthanc profile and worker are provided as a DICOM integration path; end-to-end Orthanc workflow has not been verified.
8. MONAI 3D U-Net training scaffold is provided separately; it has not been trained/evaluated in this project run.

See `diagrams/system_architecture.png` and `docs/system_architecture_api.md`.

## 6. Technology stack

- Classification / explainability: PyTorch, TorchVision DenseNet121, Grad-CAM
- Segmentation scaffold: MONAI 3D U-Net
- API: FastAPI, Pydantic
- UI: React, Vite, Nginx
- DICOM integration scaffold: Orthanc REST API and polling worker
- Data/version pipeline: DVC
- Local services: Docker Compose

## 7. Repository layout

```text
AIML-02/
├── backend/                 # FastAPI, model service, audit and Orthanc worker
│   ├── app/
│   └── tests/
├── frontend/                # React/Vite reviewer UI
├── scripts/                 # fast ChestMNIST classifier and MONAI 3D U-Net scaffold
├── models/                  # trained DenseNet121 checkpoint (copy from local run)
├── reports/                 # classification metrics and test evidence
├── docs/                    # architecture, model card, testing, runbook, reflection
├── diagrams/                # architecture image
├── data/                    # README only; do not commit full datasets
├── runtime/                 # local case/audit state (not committed)
├── compose.yaml
└── README.md
```

## 8. Run with Docker (Windows PowerShell)

Put the trained checkpoint at `models/densenet121_nih.pth` before starting the API. This file is generated by the rapid training script and is not included in this source pack; copy it from your local trained project folder. Then, from the `AIML-02` directory:

```powershell
docker compose up --build -d api frontend
docker compose ps
Invoke-RestMethod http://localhost:8000/health | Format-List
Invoke-RestMethod http://localhost:8000/model-status | Format-List
```

Open:
- UI: `http://localhost:8080`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Optional Orthanc profile (not end-to-end verified):

```powershell
docker compose --profile dicom up -d --build
```

Stop:

```powershell
docker compose stop
```

Ports bind to loopback (`127.0.0.1`). This local development configuration is not hardened for public Internet exposure. Do not change the bind address or expose it publicly without authentication, TLS, access control, de-identification and security review.

## 9. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Process and model health |
| GET | `/model-status` | Check checkpoint availability |
| POST | `/analyze` | PNG/JPEG image validation, classification and Grad-CAM |
| GET | `/cases` | List review cases |
| GET | `/cases/{case_id}` | Retrieve a case |
| PATCH | `/cases/{case_id}/review` | Record approved/rejected decision, reviewer and note |

Swagger UI: `http://localhost:8000/docs`.

## 10. Testing and limitations

- Six API unit tests passed in the prepared build environment (`6 passed in 0.47s`) before the rapid checkpoint was added; these cover health, missing-checkpoint fail-closed behavior, content validation, unknown case and review state transitions.
- Target Windows Docker: API and frontend containers built and ran; API health reported `ok`; model status reported available; single-image prediction and Grad-CAM succeeded; review action was completed in the UI.
- Orthanc DICOM end-to-end ingestion has not been verified.
- MONAI 3D U-Net segmentation has not been trained or evaluated; no Dice/HD95 results are claimed.
- No external, patient-level or clinical validation has been performed.

## 11. Safety, privacy and bias

Grad-CAM indicates model activation, not a lesion mask or causal explanation. Scores are not calibrated clinical probabilities. ChestMNIST is low-resolution and for educational use only. The reviewer identifier is an audit label, not authenticated identity. Do not upload identifiable patient information. This prototype is not clinically validated or approved for patient care.

## 12. References

1. MedMNIST maintainers, dataset repository, official splits, formats, data link, license and non-clinical-use notice: https://github.com/MedMNIST/MedMNIST
2. Yang, J. et al. (2023). *MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific Data. https://doi.org/10.1038/s41597-022-01721-8
3. NIH ChestX-ray14 source dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
4. TorchVision DenseNet121: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html
5. MONAI documentation: https://docs.monai.io/
6. Orthanc REST API: https://orthanc.uclouvain.be/book/users/rest.html
