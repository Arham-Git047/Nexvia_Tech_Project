"""Chunked EDA audit for the IEEE-CIS Fraud Detection dataset."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports" / "eda"
TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
CHUNK_SIZE = 50_000


def audit_csv(path: Path, count_target: str | None = None) -> dict:
    total_rows = 0
    missing_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    product_counts: Counter[tuple[str, str]] = Counter()
    time_min = time_max = None
    amount_min = amount_max = None
    duplicate_ids = 0
    seen_ids: set[int] = set()

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)
        missing_counts.update({str(k): int(v) for k, v in chunk.isna().sum().items()})

        if "TransactionID" in chunk:
            ids = pd.to_numeric(chunk["TransactionID"], errors="coerce").dropna().astype("int64")
            # Keep exact IDs for a duplicate audit without retaining whole rows.
            for tx_id in ids:
                i = int(tx_id)
                if i in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(i)

        if count_target and count_target in chunk:
            label_counts.update({str(k): int(v) for k, v in chunk[count_target].value_counts(dropna=False).items()})

        if {"ProductCD", "isFraud"}.issubset(chunk.columns):
            grouped = chunk.groupby(["ProductCD", "isFraud"], dropna=False).size()
            for (product, label), count in grouped.items():
                product_counts[(str(product), str(label))] += int(count)

        if "TransactionDT" in chunk:
            values = pd.to_numeric(chunk["TransactionDT"], errors="coerce").dropna()
            if not values.empty:
                chunk_min, chunk_max = float(values.min()), float(values.max())
                time_min = chunk_min if time_min is None else min(time_min, chunk_min)
                time_max = chunk_max if time_max is None else max(time_max, chunk_max)

        if "TransactionAmt" in chunk:
            values = pd.to_numeric(chunk["TransactionAmt"], errors="coerce").dropna()
            if not values.empty:
                chunk_min, chunk_max = float(values.min()), float(values.max())
                amount_min = chunk_min if amount_min is None else min(amount_min, chunk_min)
                amount_max = chunk_max if amount_max is None else max(amount_max, chunk_max)

    missing_df = pd.DataFrame(
        [{"column": col, "missing_count": count, "missing_percent": round(count / total_rows * 100, 4)}
         for col, count in missing_counts.items()]
    ).sort_values("missing_percent", ascending=False)

    return {
        "file": path.name,
        "rows": total_rows,
        "columns_missing_summary": missing_df,
        "label_counts": dict(label_counts),
        "product_label_counts": product_counts,
        "transaction_dt_min": time_min,
        "transaction_dt_max": time_max,
        "transaction_amount_min": amount_min,
        "transaction_amount_max": amount_max,
        "duplicate_transaction_ids": duplicate_ids if "TransactionID" in pd.read_csv(path, nrows=0).columns else None,
    }


def main() -> None:
    missing_files = [str(p) for p in (TRANSACTION_FILE, IDENTITY_FILE) if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(missing_files)
            + f"\nDownload the IEEE-CIS competition data and place both CSV files in {RAW_DIR}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Auditing transaction file in chunks...")
    tx = audit_csv(TRANSACTION_FILE, "isFraud")
    print("Auditing identity file in chunks...")
    identity = audit_csv(IDENTITY_FILE)

    # Match count for how many transactions have a companion identity row.
    identity_ids = set(pd.read_csv(IDENTITY_FILE, usecols=["TransactionID"])["TransactionID"].dropna().astype("int64"))
    matched = 0
    for chunk in pd.read_csv(TRANSACTION_FILE, usecols=["TransactionID"], chunksize=CHUNK_SIZE):
        matched += int(chunk["TransactionID"].isin(identity_ids).sum())

    rows = int(tx["rows"])
    fraud_rows = int(tx["label_counts"].get("1", 0))
    legitimate_rows = int(tx["label_counts"].get("0", 0))
    summary = {
        "dataset": "IEEE-CIS Fraud Detection",
        "transaction_rows": rows,
        "identity_rows": int(identity["rows"]),
        "transaction_columns": len(pd.read_csv(TRANSACTION_FILE, nrows=0).columns),
        "identity_columns": len(pd.read_csv(IDENTITY_FILE, nrows=0).columns),
        "fraud_rows": fraud_rows,
        "legitimate_rows": legitimate_rows,
        "fraud_rate_percent": round(fraud_rows / rows * 100, 5) if rows else None,
        "transactions_with_identity": matched,
        "identity_coverage_percent": round(matched / rows * 100, 3) if rows else None,
        "transaction_dt_min": tx["transaction_dt_min"],
        "transaction_dt_max": tx["transaction_dt_max"],
        "transaction_amount_min": tx["transaction_amount_min"],
        "transaction_amount_max": tx["transaction_amount_max"],
        "duplicate_transaction_ids": tx["duplicate_transaction_ids"],
    }
    (REPORT_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tx["columns_missing_summary"].to_csv(REPORT_DIR / "transaction_missingness.csv", index=False)
    identity["columns_missing_summary"].to_csv(REPORT_DIR / "identity_missingness.csv", index=False)

    product_rows = [
        {"ProductCD": product, "isFraud": label, "rows": count}
        for (product, label), count in tx["product_label_counts"].items()
    ]
    pd.DataFrame(product_rows).to_csv(REPORT_DIR / "fraud_by_product.csv", index=False)

    print("\n=== DATA AUDIT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nReports saved to: {REPORT_DIR}")
    print("Review missingness and fraud_by_product.csv before feature engineering.")


if __name__ == "__main__":
    main()
