# Real-Time Financial Fraud Detection — Local Deployment Bundle

This bundle is designed for the IEEE-CIS artifacts produced in the Colab notebook. It provides FastAPI scoring, an online GraphSAGE scorer seeded with pre-test graph history, Kafka transaction replay/consumer services, a Streamlit analyst console, Feast online feature lookup, and Evidently drift reporting with a retraining-request marker.

## Important scope note

The demonstration replays the preprocessed, label-free `stream_replay.jsonl` sample. The separate `stream_replay_ground_truth.csv` is for offline evaluation only and is never mounted into the scoring containers. The Feast online store is materialized from the same label-free replay feature rows keyed by transaction ID so the scoring API can demonstrate online lookups. A production system would ingest and compute each transaction's features at arrival, then write them to Feast before scoring. Drift monitoring only requests a retraining workflow; it must not retrain or replace a model without reviewed labels and separate validation.

## 1. Files the Colab setup must create

The project root must contain:

- `models/baseline/xgboost_ieee_cis_baseline.json`
- `models/baseline/graphsage_ieee_cis.pt`
- `models/baseline/xgboost_feature_schema.joblib`
- `models/baseline/graph_feature_schema.joblib`
- `models/baseline/graph_preprocessing.joblib`
- `models/baseline/initial_graph_state.joblib`
- `reports/models/serving_config.json`
- `data/processed/stream_replay.jsonl`
- `data/processed/feast_features.parquet`
- `data/processed/baseline_reference.csv`
- `deployment/feast_repo/feature_repo.py`
- `deployment/feast_repo/feature_store.yaml`

Use the supplied `feast_repo/generate_feast_repo.py` script from Colab to generate the Feast files and offline Parquet. The same setup cell should save `baseline_reference.csv` using a sample of `X_train` without labels.

## 2. Local Docker Compose demo

Install Docker Desktop and ensure its Linux container engine is running. Open a terminal in the **`deployment` directory**.

Build and start Kafka, Feast bootstrap, FastAPI, the Kafka consumer, and Streamlit:

```bash
docker compose up --build -d kafka feast-init api consumer dashboard
```

Check health:

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/summary
```

Publish the 1,000 test events to Kafka:

```bash
docker compose --profile tools run --rm producer
```

Open the analyst dashboard at `http://localhost:8501` and the API docs at `http://localhost:8000/docs`.

Run the drift report after the replay sample is available:

```bash
docker compose --profile tools run --rm drift-monitor
```

Key output files are saved in `reports/models/` and the case database is stored under `deployment/runtime/fraud_cases.db`.

To inspect service logs:

```bash
docker compose logs -f api consumer
```

To stop the stack without deleting the Kafka volume:

```bash
docker compose down
```

## 3. Kafka topic behavior

- `transactions.raw`: transaction event input (does not contain fraud labels).
- `transactions.scored`: JSON response from the scoring API.
- Consumer group: `fraud-scoring-v1`.

For a fresh replay from the beginning after a prior replay, use a new `KAFKA_GROUP_ID` or reset/remove the single-node development Kafka volume. This Compose configuration is for local testing, not production security or availability.

## 4. Kubernetes/Minikube

The `k8s/` folder contains starter manifests. Copy/mount the model and data artifact directories into the Minikube node before applying the manifests. For a student demo, use Docker Compose first, then demonstrate the same API/dashboard containers on Minikube if time permits. The single-node Kafka KRaft setup is for local development only.

## 5. Evaluation integrity

Do not include `stream_replay_ground_truth.csv` in the scoring containers or Kafka messages. The test labels are retained only for offline evaluation. A live risk score is a model alert, not a confirmed fraud verdict.
