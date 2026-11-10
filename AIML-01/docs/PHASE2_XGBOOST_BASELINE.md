# Phase 2 — XGBoost baseline

## Run

From the `fraud-detection-platform` repository root, with `.venv` activated and the two Kaggle training CSVs in `data/raw/`:

```powershell
python src/training/train_xgboost_baseline.py
```

The run trains an initial class-weighted XGBoost model. It uses a chronological 70%/15%/15% train/validation/test split ordered by `TransactionDT`; fits imputation and one-hot encoding on the training rows only; selects a threshold using validation data; and reports final metrics on the later test rows.

## Outputs

- `reports/models/xgboost_baseline_metrics.json`
- `reports/models/xgboost_test_predictions.csv`
- `reports/models/xgboost_top_features.csv`
- `reports/models/xgboost_test_precision_recall_curve.png`
- `models/baseline/xgboost_baseline.joblib`

## Important interpretation notes

- The audit found a 3.499% fraud rate in this training file; do not assume all fraud datasets are below 1%.
- Identity coverage is approximately 24.424%. Missing identity values are expected and retained by the left join.
- `TransactionID` is excluded from predictor columns.
- PR-AUC in the script is reported as average precision.
- The decision threshold is selected on validation data, and then fixed for test evaluation.
- This is a tabular baseline, not yet the final platform. Streaming velocity features, graph features/GNN, Feast, Kafka, FastAPI, drift monitoring and dashboard will be added in later phases.
- If the run fails with memory pressure, stop the run and share the final traceback; do not silently switch to a random split or evaluate on training rows.
