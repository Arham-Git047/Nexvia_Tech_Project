import argparse
import json
import os
import time
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError

DEFAULT_EVENTS = Path(os.getenv("EVENTS_PATH", "/data/processed/stream_replay.jsonl"))
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_RAW_TOPIC", "transactions.raw")


def main():
    parser = argparse.ArgumentParser(description="Replay prepared IEEE-CIS transactions to Kafka")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--limit", type=int, default=0, help="0 sends all events")
    parser.add_argument("--delay-ms", type=int, default=0)
    args = parser.parse_args()
    event_path = Path(args.events)
    if not event_path.exists():
        raise FileNotFoundError(f"Replay file missing: {event_path}")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda value: json.dumps(value, allow_nan=False).encode("utf-8"),
        retries=5,
        acks="all",
        max_block_ms=30000,
    )
    sent = 0
    try:
        with event_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                if "is_fraud" in event or "y_true" in event:
                    raise ValueError("Ground-truth labels must never enter the transaction topic")
                producer.send(TOPIC, key=str(event["transaction_id"]).encode(), value=event)
                sent += 1
                if args.delay_ms > 0:
                    time.sleep(args.delay_ms / 1000.0)
                if args.limit > 0 and sent >= args.limit:
                    break
        producer.flush(timeout=60)
        print(f"Published {sent} transactions to {TOPIC} via {BOOTSTRAP}")
    finally:
        producer.close(timeout=10)

if __name__ == "__main__":
    main()
