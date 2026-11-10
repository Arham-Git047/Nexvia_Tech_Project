"""Local-only Orthanc change poller that forwards rendered previews to the API."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import requests

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://orthanc:8042").rstrip("/")
API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "5"))
OFFSET_FILE = Path(os.getenv("ORTHANC_OFFSET_FILE", "/data/orthanc_offset.txt"))


def main():
    OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        offset = int(OFFSET_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        offset = 0
    session = requests.Session()
    while True:
        try:
            response = session.get(f"{ORTHANC_URL}/changes", params={"since": offset}, timeout=15)
            response.raise_for_status()
            payload = response.json()
            for change in payload.get("Changes", []):
                seq = int(change.get("Seq", offset))
                change_type = change.get("ChangeType")
                instance_id = change.get("ID")
                if change_type == "NewInstance" and instance_id:
                    image_response = session.get(f"{ORTHANC_URL}/instances/{instance_id}/preview", timeout=30)
                    image_response.raise_for_status()
                    try:
                        analyze_response = session.post(
                            f"{API_URL}/analyze",
                            files={"file": (f"orthanc_{instance_id}.png", image_response.content, "image/png")},
                            timeout=120,
                        )
                        if analyze_response.status_code == 503:
                            print("Model checkpoint unavailable; study preview not analyzed.", flush=True)
                        else:
                            analyze_response.raise_for_status()
                            print(f"Processed Orthanc instance {instance_id} into a review case.", flush=True)
                    except requests.RequestException as exc:
                        print(f"Inference request failed: {type(exc).__name__}", flush=True)
                offset = max(offset, seq)
                OFFSET_FILE.write_text(str(offset), encoding="utf-8")
            if payload.get("Done", True):
                time.sleep(POLL_INTERVAL)
        except requests.RequestException as exc:
            print(f"Orthanc polling temporarily failed: {type(exc).__name__}; retrying.", flush=True)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
