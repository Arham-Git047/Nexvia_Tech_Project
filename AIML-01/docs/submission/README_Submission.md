# AIML-01 — Real-Time Financial Fraud Detection Platform

**Student:** Arham Kelkar  
**Institution:** Dr. D.Y. Patil School of Science and Technology  
**Submission date:** 26 October 2026

## What this project demonstrates

A research/demo fraud-risk scoring pipeline using the IEEE-CIS Fraud Detection dataset. The solution combines a class-weighted XGBoost tabular model with a GraphSAGE model built over card/device profile relationship proxies. A 95% XGBoost / 5% GraphSAGE blend was selected using validation Average Precision. The local demonstration streams label-free historical transaction events through Kafka, scores them through a FastAPI service, stores cases for analyst review, and presents them in Streamlit. Feast provides the demo feature lookup; MLflow tracks model runs; Evidently and a custom PSI rule produce monitoring outputs.

## Main reported offline metrics

| Model | Test Average Precision | Test ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.2333 | 0.7763 | 0.3374 | 0.2413 | 0.2814 |
| GraphSAGE | 0.1810 | 0.7970 | 0.2136 | 0.3789 | 0.2732 |
| Hybrid blend | 0.2452 | 0.8155 | 0.2325 | 0.4093 | 0.2965 |

The hybrid increased test Average Precision by about 5.1% over XGBoost alone in this experiment. Threshold-specific metrics were calculated using a threshold selected on validation data. These results do not establish production performance.

## Local demo status

The Windows Docker Compose demo processed 1,000 historical label-free events, queued 33 cases for review, saved one analyst feedback record, and generated both the Evidently HTML report and PSI output. The custom PSI monitor found 37/435 features with PSI >= 0.25 (8.51%), below the configured 10% trigger share, so `retraining_requested=false`. Evidently separately flagged 98/435 columns as drifted, while its dataset-level drift test did not cross its configured threshold. These are different statistical checks and should not be conflated.

## Start the local demo

The complete source/model backup is packaged separately as `fraud_detection_submission_backup.zip`; this pack contains selected report documents and evidence screenshots. Extract the full source backup, then from the `deployment` directory run:

```powershell
docker compose config
docker compose up --build -d kafka feast-init api consumer dashboard
docker compose ps -a
```

Verify `http://localhost:8000/health`, `http://localhost:8000/summary`, API docs at `http://localhost:8000/docs`, and the dashboard at `http://localhost:8501`. To replay the demonstration events:

```powershell
docker compose --profile tools run --rm producer
```

To generate a drift report:

```powershell
docker compose --profile tools run --rm drift-monitor
```

Stop without deleting persistent project data:

```powershell
docker compose stop
```

## Evidence files in this pack

- `01_Screenshots/01_dashboard_1000_transactions.png` — dashboard summary with 1,000 events and 33 review cases.
- `01_Screenshots/03_case_review_scores.png` — single-case review screen and model scores.
- `01_Screenshots/05_api_summary_after_restart.png` — API summary after restart and saved feedback count.
- `01_Screenshots/06_evidently_drift_report.png` — Evidently drift report snapshot.
- `04_Architecture/system_architecture.png` — architecture diagram.
- `02_Documentation/` — system/API documentation and test/monitoring report.
- `03_Presentation/` — 9-slide presentation.

## Important limits and reproducibility notes

- The replay is historical test data used for a local integration demonstration, not live banking traffic. Ground-truth labels are kept out of serving events.
- The dashboard’s risk score is for prioritizing human review, not a verified probability or a final fraud decision.
- Card/device profile graph edges are proxies, not verified fraud-ring links.
- The retraining monitor writes a request marker only. A reviewed-label dataset, offline validation, approval, and rollback are required before model promotion.
- Docker Compose was verified locally; Kubernetes/Minikube manifests are starter materials and were not confirmed as deployed.
- Production security, high availability, load testing, latency-at-scale, and failover are outside the verified scope.

## Final submission check

Place the completed individual reflection in this pack before handing it in. Include the separate full source/model backup alongside this pack (or wherever the course portal specifies). Do not include raw ground-truth labels in the online event stream or screenshots.


## Deployment guide and video plan

Additional files are in `05_Deployment_Guide_and_Video/`: implementation/deployment runbook and deployment video pitch with scene-by-scene narration. The guide documents the verified local Docker Compose demo. Kubernetes/Minikube was not verified.
