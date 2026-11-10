"""Append-only JSONL audit helper for local educational demos."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_LOCK = Lock()


def append_audit(event: str, case_id: str, details: dict | None = None) -> None:
    path = Path(os.getenv("AUDIT_LOG_PATH", "/data/audit.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "case_id": case_id,
        "details": details or {},
    }
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
