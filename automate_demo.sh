#!/bin/bash
set -e

echo "========================================================"
echo "          STARTING FRAUD DETECTION DEMO SETUP           "
echo "========================================================"

# 1. Clean up old runs
echo "[1/4] Cleaning up previous containers..."
docker compose down

# 2. Build and Start Containers
echo "[2/4] Building and starting services..."
docker compose up -d --build

echo "Waiting 15 seconds for services to initialize..."
sleep 15

# 3. Run Pipeline
echo "[3/4] Running MLOps Pipeline (Ingestion -> Monitoring)..."
docker exec fraud-detection-mlops-poc-fraud-detection-pipeline-1 ./run_pipeline.sh

# 4. Show Endpoints
echo ""
echo "========================================================"
echo "             DEMO ENVIRONMENT READY                     "
echo "========================================================"
echo "Available Endpoints:"
echo " -> API Root:          http://localhost:8000/"
echo " -> Drift Dashboard:   http://localhost:8000/dashboard/drift"
echo " -> Metrics:           http://localhost:8000/metrics"
echo " -> MLflow UI:         http://localhost:5000"
echo " -> Prometheus UI:     http://localhost:9090"
echo " -> Grafana UI:        http://localhost:3000"
echo "========================================================"
