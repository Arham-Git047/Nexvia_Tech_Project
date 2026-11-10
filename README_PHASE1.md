# IEEE-CIS Fraud Detection — Phase 1

## Dataset files
Download the IEEE-CIS competition data from https://www.kaggle.com/competitions/ieee-fraud-detection/data. Put these files in `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`

Do not rename the files. Do not commit raw data to Git.

## Run the audit
From the repository root, with the virtual environment activated:

```powershell
python src/data/audit_dataset.py
```

Outputs are written to `reports/eda/`:
- `data_summary.json`
- `transaction_missingness.csv`
- `identity_missingness.csv`
- `fraud_by_product.csv`

This is a data audit, not model evaluation. It deliberately does not calculate model metrics.
