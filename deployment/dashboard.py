import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("SCORING_API_URL", "http://api:8000").rstrip("/")
st.set_page_config(page_title="Fraud Analyst Console", page_icon="🛡️", layout="wide")
st.title("🛡️ Real-Time Financial Fraud Detection")
st.caption("IEEE-CIS research/demo environment — all alerts are model recommendations, not final determinations.")

@st.cache_data(ttl=3)
def fetch_json(path, params=None):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()

try:
    health = fetch_json("/health")
    metrics = fetch_json("/summary")
    cases_response = fetch_json("/cases", {"limit": 500})
except Exception as exc:
    st.error(f"API is not reachable at {API_URL}: {exc}")
    st.stop()

status_text = "Models ready" if health.get("models_loaded") else "Models starting"
st.info(f"API status: {health.get('status')} · {status_text}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Transactions scored", metrics.get("total", 0))
m2.metric("Cases to review", metrics.get("review_count", 0))
m3.metric("Confirmed fraud", metrics.get("confirmed_fraud", 0))
m4.metric("Marked legitimate", metrics.get("marked_legitimate", 0))
m5.metric("Feedback records", metrics.get("feedback_count", 0))

cases = cases_response.get("cases", [])
if not cases:
    st.warning("No scores have arrived yet. Start the Kafka replay producer to populate the dashboard.")
    st.stop()

frame = pd.DataFrame(cases)
if "payload_json" in frame.columns:
    frame = frame.drop(columns=["payload_json"])

st.subheader("Highest-risk transactions")
filter_status = st.selectbox(
    "Filter by case status",
    ["ALL", "NEW", "REVIEWING", "CONFIRMED_FRAUD", "LEGITIMATE", "CLOSED"],
)
visible = frame if filter_status == "ALL" else frame[frame["status"] == filter_status]
show_cols = [c for c in [
    "transaction_id", "transaction_dt", "risk_score", "xgb_score",
    "graph_score", "decision", "status", "feature_source", "created_at"
] if c in visible.columns]
st.dataframe(
    visible[show_cols].sort_values("risk_score", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Review a case")
ids = [int(v) for v in visible["transaction_id"].tolist()]
if not ids:
    st.info("No cases match the selected status.")
else:
    selected_id = st.selectbox("Transaction ID", ids)
    current = frame[frame["transaction_id"] == selected_id].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Hybrid risk score", f"{float(current['risk_score']):.4f}")
    col2.metric("XGBoost score", f"{float(current['xgb_score']):.4f}")
    col3.metric("GraphSAGE score", f"{float(current['graph_score']):.4f}")

    statuses = ["NEW", "REVIEWING", "CONFIRMED_FRAUD", "LEGITIMATE", "CLOSED"]
    default_status = current["status"] if current["status"] in statuses else "NEW"
    review_status = st.selectbox("Set case status", statuses, index=statuses.index(default_status))
    labels = ["UNREVIEWED", "FRAUD", "LEGITIMATE"]
    analyst_label = st.selectbox("Analyst label", labels)
    notes = st.text_area("Investigation notes", max_chars=4000)

    if st.button("Save analyst feedback", type="primary"):
        try:
            response = requests.patch(
                f"{API_URL}/cases/{selected_id}",
                json={"status": review_status, "analyst_label": analyst_label, "notes": notes},
                timeout=10,
            )
            response.raise_for_status()
            st.success("Case status and feedback saved.")
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save feedback: {exc}")

st.caption("This is a demonstration using a historical transaction replay. The final disposition should be based on human review and verified labels.")
