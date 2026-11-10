# Nexvia Tech Project — AI/ML Engineering Projects

**Developer:** Arham Kelkar  
**Institution:** Dr. D. Y. Patil School of Science and Technology  
**GitHub:** [Arham-Git047/Nexvia_Tech_Project](https://github.com/Arham-Git047/Nexvia_Tech_Project)

This repository contains two academic AI/ML research and engineering projects. Each project includes its source code, model artifacts where permitted, setup instructions, technical documentation, and test evidence.

> **Demo note:** The applications have been run locally with Docker Compose and screen-recorded for evaluation. Localhost addresses below are only available on the machine running the containers; they are not public deployment URLs.

## Repository layout

```text
Nexvia_Tech_Project/
├── AIML-01/                     # Real-Time Financial Fraud Detection
│   ├── data/                    # Local data; raw and derived datasets excluded from Git
│   ├── deployment/              # Kafka, FastAPI, Feast, Streamlit, Docker
│   ├── docs/                    # Experiment, architecture and submission documents
│   ├── models/baseline/          # XGBoost, GraphSAGE and feature artifacts
│   ├── reports/                 # EDA, model metrics and drift evidence
│   ├── src/                     # Data audit and training code
│   └── README.md
├── AIML-02/                     # AI-Assisted Medical Imaging Review
│   ├── backend/                 # FastAPI, model service, audit workflow
│   ├── frontend/                # React review console
│   ├── models/                  # DenseNet121 checkpoint
│   ├── reports/                 # Classification metrics and smoke-test evidence
│   ├── docs/                    # Architecture, model card, setup and test documents
│   ├── diagrams/                # System architecture
│   ├── scripts/                 # Training and utility scripts
│   ├── compose.yaml
│   └── README.md
└── README.md                    # This project index
```

## AIML-01 — Real-Time Financial Fraud Detection Platform

### Objective

Process transaction events, estimate fraud risk with machine-learning models, and present higher-risk cases to an analyst for review.

### Architecture and technology

Kafka transaction topic → consumer and feature transformation → Feast feature retrieval → FastAPI scoring → XGBoost/GraphSAGE hybrid score → case and feedback store → Streamlit dashboard. Evidently is used for feature-drift monitoring and MLflow for experiment tracking.

**Main stack:** Python, XGBoost, PyTorch/GraphSAGE, Apache Kafka, FastAPI, Feast, Streamlit, Evidently, MLflow, Docker Compose.

### Dataset audit

The IEEE-CIS audit used 590,540 transaction rows and 144,233 identity rows. It found 20,663 fraud records (approximately 3.50% of transaction rows); identity coverage was approximately 24.42%. Original dataset files and processed replay data are not included in this public repository. Obtain data only from its authorized source and follow its terms.

### Test-set model results

| Model | Average Precision | ROC-AUC |
|---|---:|---:|
| XGBoost | 0.2333 | 0.7763 |
| GraphSAGE | 0.1810 | 0.7970 |
| Hybrid (95% XGBoost + 5% GraphSAGE) | 0.2452 | 0.8155 |

At the validation-selected operating threshold, the hybrid model recorded precision 0.2325, recall 0.4093 and F1 0.2965. These are experimental offline metrics, not production guarantees. GraphSAGE uses proxy relationships from available profile features; it does not establish verified fraud-ring identities.

### Previously verified streaming demo

A recorded local replay scored 1,000 transactions and generated 33 cases for review. One analyst feedback record was saved. These are results from that recorded run and are not guaranteed for every replay.

### Run locally (Windows + Docker Desktop)

From the repository root:

```powershell
cd .\AIML-01
docker compose -f .\deployment\docker-compose.yml up -d kafka feast-init api consumer dashboard
docker compose -f .\deployment\docker-compose.yml ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

Open the Streamlit dashboard at `http://localhost:8501` and FastAPI docs at `http://localhost:8000/docs`.

To run the drift-monitor job:

```powershell
docker compose -f .\deployment\docker-compose.yml --profile tools run --rm drift-monitor
```

To stop AIML-01:

```powershell
docker compose -f .\deployment\docker-compose.yml stop
```

A transaction replay fixture is excluded from Git because it is derived from the dataset. Place the authorized `stream_replay.jsonl` at `AIML-01/data/processed/stream_replay.jsonl` before running the optional producer profile.

### Limitations

The Docker Compose local demo was tested. Kubernetes starter manifests are present, but Kubernetes/Minikube deployment was not verified. Risk scores are recommendations for human review, not final fraud determinations.

## AIML-02 — AI-Assisted Medical Imaging Review (MedVision Review)

### Objective

Provide a research prototype for image upload, multi-label chest X-ray classification, Grad-CAM activation visualization, and human review with an audit record.

### Architecture and technology

React review console → FastAPI backend → PyTorch DenseNet121 classifier → Grad-CAM overlay → clinician review and audit workflow. Docker Compose runs the API and frontend locally. The project also contains an Orthanc integration path and a MONAI segmentation training entry point.

**Main stack:** React, FastAPI, PyTorch, DenseNet121, Grad-CAM, MONAI training scaffold, Orthanc integration scaffold, Docker, DVC configuration.

### Rapid classification experiment

The rapid prototype used ChestMNIST, a small 14-label dataset derived from NIH ChestX-ray14. The run used 800 training samples, 200 validation samples and 200 test samples, with a frozen pretrained DenseNet121 backbone and two classifier-head training epochs on CPU. The reported training time was approximately 2.637 minutes.

| Metric | Recorded result |
|---|---:|
| Macro Average Precision | 0.1186 |
| Macro ROC-AUC | 0.6024 |
| Micro Precision at 0.5 | 0.0873 |
| Micro Recall at 0.5 | 0.6294 |
| Micro F1 at 0.5 | 0.1533 |

A sample inference returned Pneumothorax as the top finding with a score of 0.686565 and generated a Grad-CAM overlay. The clinician review action was exercised in the UI. Scores are not calibrated diagnostic probabilities and must not be interpreted as diagnoses.

### Run locally (Windows + Docker Desktop)

From the repository root:

```powershell
cd .\AIML-02
docker compose up --build -d api frontend
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

Open the React review console at `http://localhost:8080` and API documentation at `http://localhost:8000/docs`.

To stop AIML-02:

```powershell
docker compose stop
```

### Limitations and safety

This is for education and research only, not clinical use. The ChestMNIST rapid run is a small prototype experiment and is not clinically validated. Orthanc end-to-end DICOM ingestion was not verified. The MONAI 3D U-Net segmentation branch is present as a training scaffold but was not trained or evaluated in this run. Grad-CAM is an activation overlay, not a lesion segmentation mask.

## Reproducibility and data policy

- Do not commit raw datasets, patient-identifiable data, API credentials, `.env` files, runtime databases, temporary files, or local caches.
- Keep model artifacts only where their source/license and repository size are appropriate. The AIML-01 XGBoost model is larger than GitHub's recommended 50 MiB warning threshold but below the 100 MiB hard file limit; consider Git LFS or release assets if the repository grows.
- Re-run experiments only with the documented dataset split and configuration; report any differences from the recorded results.

## Deliverables to add before final portal submission

- Add the final 10-minute PowerPoint for each project under its respective `presentation/` folder.
- Add the final screen-recorded demo links under each project's `demo/README.md` or in the Drive folder linked by the submission portal.
- Personalize and finalize the individual reflection for each project.
- Verify the public GitHub repository and Google Drive video links in a signed-out/incognito browser.

**No live public deployment is claimed.** The locally tested applications are demonstrated through screen recordings. Only provide a live deployment URL if you later deploy and verify one.
