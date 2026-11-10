# Real-Time Financial Fraud Detection Platform

**Developer:** Arham Kelkar
**Institution:** Dr. D.Y. Patil School of Science and Technology

## Overview
A real-time financial fraud detection platform using Kafka, FastAPI, Feast, XGBoost, GraphSAGE and Streamlit.

## Architecture
Kafka -> Consumer -> Feast -> FastAPI -> Hybrid XGBoost and GraphSAGE -> Analyst Dashboard.

## Dataset
IEEE-CIS Fraud Detection. 590,540 transactions, 20,663 fraud records and approximately 3.5% fraud rate. Original dataset excluded.

## Model Results
XGBoost: AP 0.2333, ROC-AUC 0.7763.
GraphSAGE: AP 0.1810, ROC-AUC 0.7970.
Hybrid: AP 0.2452, ROC-AUC 0.8155.

## Verified Demo
1,000 transactions scored, 33 review cases and 1 analyst feedback record.

## Run
Use Docker Desktop, then run: docker compose -f .\\deployment\\docker-compose.yml up -d kafka feast-init api consumer dashboard

Dashboard: http://localhost:8501
API documentation: http://localhost:8000/docs

## Project Structure
src/ - training and data audit
deployment/ - Kafka, API, dashboard and Docker
models/ - trained models and schemas
docs/ - documentation
reports/ - metrics and drift reports

## Limitations
Academic research/demo only. Risk scores require analyst review. Kubernetes deployment is not verified. Graph relationships are proxies, not verified fraud rings.
