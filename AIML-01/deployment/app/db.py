import json
import os
import sqlite3
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", "/runtime"))
DB_PATH = RUNTIME_DIR / "fraud_cases.db"


def connect():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                transaction_id INTEGER PRIMARY KEY,
                transaction_dt REAL,
                xgb_score REAL NOT NULL,
                graph_score REAL NOT NULL,
                risk_score REAL NOT NULL,
                threshold REAL NOT NULL,
                decision TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'NEW',
                feature_source TEXT NOT NULL DEFAULT 'event_payload',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                payload_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                analyst_label TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(transaction_id) REFERENCES cases(transaction_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_risk ON cases(risk_score DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)")


def get_case(transaction_id: int):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM cases WHERE transaction_id = ?", (int(transaction_id),)
        ).fetchone()
    return dict(row) if row else None


def upsert_case(record: dict[str, Any]):
    with connect() as conn:
        conn.execute("""
            INSERT INTO cases (
                transaction_id, transaction_dt, xgb_score, graph_score,
                risk_score, threshold, decision, status, feature_source, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'NEW', ?, ?)
            ON CONFLICT(transaction_id) DO NOTHING
        """, (
            int(record["transaction_id"]),
            record.get("transaction_dt"),
            float(record["xgb_score"]),
            float(record["graph_score"]),
            float(record["risk_score"]),
            float(record["threshold"]),
            str(record["decision"]),
            str(record.get("feature_source", "event_payload")),
            json.dumps(record.get("payload", {}), allow_nan=False),
        ))
    return get_case(int(record["transaction_id"]))


def list_cases(limit: int = 200, status: str | None = None):
    limit = max(1, min(int(limit), 5000))
    query = "SELECT * FROM cases"
    args: list[Any] = []
    if status and status != "ALL":
        query += " WHERE status = ?"
        args.append(status)
    query += " ORDER BY risk_score DESC, created_at DESC LIMIT ?"
    args.append(limit)
    with connect() as conn:
        rows = conn.execute(query, args).fetchall()
    return [dict(r) for r in rows]


def update_case(transaction_id: int, status: str, analyst_label: str, notes: str = ""):
    allowed_statuses = {"NEW", "REVIEWING", "CONFIRMED_FRAUD", "LEGITIMATE", "CLOSED"}
    allowed_labels = {"UNREVIEWED", "FRAUD", "LEGITIMATE"}
    if status not in allowed_statuses:
        raise ValueError(f"Invalid status: {status}")
    if analyst_label not in allowed_labels:
        raise ValueError(f"Invalid analyst label: {analyst_label}")

    with connect() as conn:
        cur = conn.execute(
            "UPDATE cases SET status = ? WHERE transaction_id = ?",
            (status, int(transaction_id)),
        )
        if cur.rowcount == 0:
            return None
        conn.execute(
            "INSERT INTO feedback (transaction_id, analyst_label, notes) VALUES (?, ?, ?)",
            (int(transaction_id), analyst_label, notes[:4000]),
        )
    return get_case(int(transaction_id))


def summary():
    with connect() as conn:
        totals = conn.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) AS review_count,
                   SUM(CASE WHEN status = 'CONFIRMED_FRAUD' THEN 1 ELSE 0 END) AS confirmed_fraud,
                   SUM(CASE WHEN status = 'LEGITIMATE' THEN 1 ELSE 0 END) AS marked_legitimate,
                   AVG(risk_score) AS avg_risk
            FROM cases
        """).fetchone()
        feedback_count = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
    result = dict(totals)
    result["feedback_count"] = int(feedback_count)
    for key in ("review_count", "confirmed_fraud", "marked_legitimate"):
        result[key] = int(result.get(key) or 0)
    result["total"] = int(result.get("total") or 0)
    result["avg_risk"] = float(result["avg_risk"] or 0.0)
    return result
