#!/bin/bash
set -e

echo "Starting Fraud Detection Pipeline..."

echo "-----------------------------------"
echo "Step 1: Data Ingestion"
echo "-----------------------------------"
python src/data_ingestion.py

echo "-----------------------------------"
echo "Step 2: Data Validation"
echo "-----------------------------------"
python src/data_validation.py

echo "-----------------------------------"
echo "Step 3: Feature Engineering"
echo "-----------------------------------"
python src/feature_engineering.py

echo "-----------------------------------"
echo "Step 4: Data Splitting"
echo "-----------------------------------"
python src/data_splitting.py

echo "-----------------------------------"
echo "Step 5: Model Training"
echo "-----------------------------------"
python src/model_training.py

echo "-----------------------------------"
echo "Step 6: Model Monitoring"
echo "-----------------------------------"
python src/model_monitoring.py

echo "-----------------------------------"
echo "Pipeline Completed Successfully!"
echo "-----------------------------------"
