import json
import os
import time
import requests
from kafka import KafkaConsumer, KafkaProducer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
RAW_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "transactions.raw")
SCORED_TOPIC = os.getenv("KAFKA_SCORED_TOPIC", "transactions.scored")
API_URL = os.getenv("SCORING_API_URL", "http://api:8000/score")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud-scoring-v1")


def wait_for_api():
    health_url = API_URL.rsplit("/", 1)[0] + "/health"
    for attempt in range(60):
        try:
            response = requests.get(health_url, timeout=3)
            if response.ok and response.json().get("models_loaded"):
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("Scoring API did not become ready within 120 seconds")


def main():
    wait_for_api()
    consumer = KafkaConsumer(
        RAW_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        max_poll_interval_ms=300000,
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda value: json.dumps(value, allow_nan=False).encode("utf-8"),
        retries=5,
        acks="all",
    )
    print(f"Consuming {RAW_TOPIC}; scoring via {API_URL}")
    try:
        for message in consumer:
            event = message.value
            try:
                response = requests.post(API_URL, json=event, timeout=120)
                response.raise_for_status()
                result = response.json()
                producer.send(SCORED_TOPIC, key=str(event["transaction_id"]).encode(), value=result)
                producer.flush(timeout=30)
                consumer.commit()
                print(
                    f"scored transaction={result['transaction_id']} "
                    f"risk={result['risk_score']:.5f} decision={result['decision']}"
                )
            except Exception as exc:
                # Do not commit this offset when scoring fails; restart/retry is possible.
                print(f"ERROR scoring transaction {event.get('transaction_id')}: {exc}")
                time.sleep(2)
    finally:
        consumer.close()
        producer.close()

if __name__ == "__main__":
    main()
