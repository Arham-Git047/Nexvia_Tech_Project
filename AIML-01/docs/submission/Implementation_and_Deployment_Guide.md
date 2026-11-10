# Implementation & Deployment Guide — Real-Time Financial Fraud Detection Platform

**Prepared by:** Arham Kelkar  
**Institution:** Dr. D.Y. Patil School of Science and Technology  
**Date:** 26 October 2026

> Scope: this is the previously verified Windows Docker Compose demonstration. Kubernetes/Minikube was not verified.

## 1. What the system demonstrates

- IEEE-CIS fraud-risk scoring using class-weighted XGBoost and GraphSAGE, blended at 95% XGBoost / 5% GraphSAGE.
- Label-free JSONL replay → Kafka `transactions.raw` → consumer → FastAPI score service → `transactions.scored` and case database.
- Streamlit analyst dashboard and feedback persistence.
- Evidently HTML and custom PSI monitoring; retraining marker only, not automatic model replacement.

## 2. Verified environment

- Windows / PowerShell.
- Docker Desktop 4.96.3; Docker Engine 29.9.2; Compose v5.6.1 recorded during verification.
- Project root: `C:\Users\admin\Documents\AIML-01\fraud-detection-platform`.
- Docker reported about 3.77 GiB memory available. First dependency build took a long time; do not rebuild needlessly.
- Separate source backup `fraud_detection_submission_backup.zip`: 78 files / about 103.7 MB, with critical entries checked. Retain it separately.

## 3. Start services

Open PowerShell:

```powershell
cd "C:\Users\admin\Documents\AIML-01\fraud-detection-platform\deployment"
docker version
docker compose version
docker compose config
```

First build (only if images have not been built or code/dependencies changed):

```powershell
docker compose up --build -d kafka feast-init api consumer dashboard
```

Subsequent starts without rebuilding:

```powershell
docker compose up -d kafka feast-init api consumer dashboard
docker compose ps -a
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 5
Invoke-RestMethod http://localhost:8000/summary | ConvertTo-Json -Depth 5
```

Expected verified health: `status=ok`, `models_loaded=true`. Feast initialization may show `exited (0)` after successful one-time setup. Kafka should be healthy; API, consumer, and dashboard should be running.

## 4. Replay 1,000 transactions

```powershell
docker compose --profile tools run --rm producer
docker compose logs --tail 60 consumer
Invoke-RestMethod http://localhost:8000/summary | ConvertTo-Json -Depth 5
```

Recorded producer output: `Published 1000 transactions to transactions.raw via kafka:9092`. The consumer logged scored transaction records. This is historical replay, not live traffic; do not place ground-truth labels in Kafka or scoring containers.

## 5. Dashboard and analyst feedback

1. Open [http://localhost:8501](http://localhost:8501).
2. Show 1,000 scored events and 33 review cases.
3. Open the highest-risk case list.
4. Use the review controls to save a status/label and note for a demo record.
5. Refresh the dashboard or call `/summary`; the recorded run retained one feedback record after service restart.

API docs: [http://localhost:8000/docs](http://localhost:8000/docs). Endpoints: `GET /health`, `GET /summary`, `POST /score`, `GET /cases?limit=500`, `PATCH /cases/{transaction_id}`, `GET /docs`. Inspect the exact scoring request schema in Swagger; do not guess a payload.

## 6. Drift monitor

```powershell
docker compose --profile tools run --rm drift-monitor
```

Recorded run: 3,000 reference rows, 1,000 current rows, 435 features. Evidently flagged 98/435 columns (22.529%), but dataset-level drift was not detected at threshold 0.5. Custom PSI flagged 37/435 features at PSI ≥ 0.25 (8.51%) against a 10% trigger; `retraining_requested=false`. Artifacts: `reports/models/evidently_data_drift_report.html`, `reports/models/feature_drift_psi.csv`, `deployment/runtime/retrain_trigger.json`. Evidently and PSI are different tests; report them separately. Non-fatal SciPy divide-by-zero warnings appeared, but `evidently_error=null` and the HTML report was generated.

## 7. Offline model results

| Model | Test AP | Test ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.2333 | 0.7763 | 0.3374 | 0.2413 | 0.2814 |
| GraphSAGE | 0.1810 | 0.7970 | 0.2136 | 0.3789 | 0.2732 |
| Hybrid (0.95 / 0.05) | 0.2452 | 0.8155 | 0.2325 | 0.4093 | 0.2965 |

Average Precision is the PR-AUC summary. Precision/recall/F1 use thresholds selected on validation data. Hybrid test AP improved about 5.1% over XGBoost alone. These are offline test results, not guaranteed future performance.

## 8. Video evidence checklist

- [ ] Service status (`docker compose ps -a`)
- [ ] `/health` and `/summary` responses
- [ ] Producer confirmation for 1,000 events and consumer logs
- [ ] Dashboard showing 1,000 transactions and 33 review cases
- [ ] Analyst review and feedback count
- [ ] Evidently report and PSI trigger output
- [ ] Model comparison and limitations slide

## 9. Troubleshooting and shutdown

- Kafka unhealthy: `docker compose logs --tail 100 kafka`, then re-check health.
- Consumer restarts: inspect `docker compose logs --tail 100 consumer`. A previously seen `consumer_timeout_ms=0` issue was fixed by removing that setting from `deployment/app/consumer.py` and rebuilding the consumer image; only apply if the same issue recurs.
- API model load issue: inspect API logs and artifact paths; do not claim scoring is ready until `/health` confirms model loading.
- Stop without deleting volumes: `docker compose stop`. Avoid `down -v` and prune commands until evidence and backups are safe.

## 10. Honest scope boundaries

- Historical replay, not live financial transactions.
- Alerts are review recommendations, not confirmed fraud determinations or calibrated probabilities.
- GraphSAGE uses card/device profile relationship proxies, not definitive fraud rings.
- Feast demo features were pre-materialized from label-free replay rows; production should write features at arrival.
- Retraining request does not promote a model; reviewed labels, validation, approval, and rollback are required.
- Kubernetes starter manifests were not deployed/verified. Production security, HA, load, and failover testing remain out of scope.
