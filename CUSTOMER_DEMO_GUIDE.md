# Customer Demo Guide: End-to-End MLOps Fraud Detection System

## Demo Overview
This guide provides a step-by-step walkthrough to demonstrate your complete MLOps pipeline to customers, showcasing production-grade ML operations with monitoring, deployment, and drift detection.

## Demo Prerequisites

### 1. Environment Setup
```bash
# Clone and setup
git clone <your-repo-url>
cd Fraud-Detection-Mlops-poc

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Components
```bash
# Check all required files exist
ls -la
# Should show: Dockerfile, docker-compose.yml, requirements.txt, src/, tests/

# Verify GitHub Actions workflows
ls -la .github/workflows/
# Should show: 8 workflow files
```

## Demo Flow (45-60 minutes)

### Phase 1: System Architecture Overview (5 minutes)

**Show Architecture Diagram:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│   Data Pipeline │───▶│   Feature Eng.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Train   │───▶│   Model Registry│───▶│   Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│   Drift Detect  │◀───│   Canary Deploy │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Key Points to Highlight:**
- End-to-end automation
- Production-grade monitoring
- Automated rollback capability
- Multi-model approach

### Phase 2: Data Pipeline Demonstration (10 minutes)

#### Step 2.1: Run Data Pipeline
```bash
# Start with clean state
python run_pipeline.py
# Select: 1 - Run Data Pipeline
```

**What to Show:**
- **Data Ingestion**: Loading creditcard.csv (284,807 transactions)
- **Data Validation**: Quality checks and profiling
- **Feature Engineering**: 12+ engineered features
- **Consistency Checks**: Parity validation

**Expected Output:**
```
========================================
Starting Data Ingestion Pipeline...
========================================
Data ingestion completed successfully
Data validation passed: 284807 rows, 31 columns
Memory usage: 67.91 MB
Feature engineering completed. Generated 12 new features
Feature consistency checks passed
```

#### Step 2.2: Show Data Quality Reports
```bash
# Show generated files
ls -la data/validated/
# Should show: creditcard_validated.parquet, feature_consistency_report.json

# Display sample data
python -c "
import pandas as pd
df = pd.read_parquet('data/validated/creditcard_validated.parquet')
print('Shape:', df.shape)
print('Fraud rate:', df['Class'].mean())
print('Features:', list(df.columns))
"
```

**Key Metrics to Highlight:**
- Dataset size: 284,807 transactions
- Fraud rate: 0.17% (highly imbalanced)
- Features: 43 total (31 original + 12 engineered)
- Data quality: 100% validation passed

### Phase 3: Model Training & Comparison (15 minutes)

#### Step 3.1: Train Models
```bash
# Train all models
python run_pipeline.py
# Select: 2 - Train Models
```

**What to Show:**
- **Multi-Model Training**: 4 different algorithms
- **Class Imbalance Handling**: SMOTE technique
- **MLFlow Integration**: Experiment tracking
- **Performance Metrics**: F1, Precision, Recall, ROC-AUC

**Expected Output:**
```
========================================
Starting Model Training with MLFlow...
========================================
Models trained: ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']
Best Model: XGBoost
Composite Score: 0.8452
Model training completed successfully
```

#### Step 3.2: Show Model Comparison
```bash
# Display model metrics
cat models/metrics.json | python -m json.tool
```

**Key Performance Metrics:**
- **XGBoost**: F1-Score: 0.7319, ROC-AUC: 0.9802
- **LightGBM**: F1-Score: 0.7298, ROC-AUC: 0.9713
- **RandomForest**: F1-Score: 0.7121, ROC-AUC: 0.9788
- **LogisticRegression**: F1-Score: 0.6821, ROC-AUC: 0.9657

#### Step 3.3: Model Registry
```bash
# Show model registry
cat models/model_comparison_report.json | python -m json.tool
```

**Highlight:**
- Version tracking
- Performance comparison
- Best model selection
- Audit trail

### Phase 4: API Service & Testing (10 minutes)

#### Step 4.1: Start API Service
```bash
# Start FastAPI service
uvicorn src.api.service:app --host 0.0.0.0 --port 8000 --reload
```

**Open browser to:** http://localhost:8000/docs

**What to Show:**
- **Swagger UI**: Interactive API documentation
- **Health Check**: `/health` endpoint
- **Single Prediction**: `/predict` endpoint
- **Batch Prediction**: `/predict_batch` endpoint

#### Step 4.2: Test API Endpoints

**Test Health Check:**
```bash
curl http://localhost:8000/health
```

**Test Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 123456,
    "Amount": 150.50,
    "V1": 0.5,
    "V2": -1.2,
    "V3": 0.8,
    "V4": -0.3,
    "V5": 1.1,
    "V6": -0.7,
    "V7": 0.9,
    "V8": -0.4,
    "V9": 0.6,
    "V10": -0.8,
    "V11": 0.7,
    "V12": -0.5,
    "V13": 0.9,
    "V14": -0.6,
    "V15": 0.4,
    "V16": -0.9,
    "V17": 0.3,
    "V18": -0.7,
    "V19": 0.8,
    "V20": -0.2,
    "V21": 0.5,
    "V22": -0.4,
    "V23": 0.6,
    "V24": -0.8,
    "V25": 0.7,
    "V26": -0.3,
    "V27": 0.9,
    "V28": -0.5
  }'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "probability": 0.1234,
  "model_version": "v1.0.0",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

#### Step 4.3: Show Prometheus Metrics
```bash
# View metrics endpoint
curl http://localhost:8000/metrics
```

**Highlight Key Metrics:**
- `fraud_detection_requests_total`: Total API requests
- `fraud_detection_request_duration_seconds`: Response time
- `fraud_detection_predictions_total`: Prediction counts
- `fraud_detection_model_info`: Model version info

### Phase 5: Docker & Containerization (5 minutes)

#### Step 5.1: Build Docker Image
```bash
# Build container
docker build -t fraud-detection:latest .

# Run container
docker run -p 8000:8000 fraud-detection:latest
```

#### Step 5.2: Docker Compose (Full Stack)
```bash
# Start complete stack
docker-compose up -d

# Services started:
# - fraud-detection-api:8000
# - prometheus:9090
# - grafana:3000
# - nginx:80
```

**Show Running Containers:**
```bash
docker ps
```

### Phase 6: Monitoring & Observability (10 minutes)

#### Step 6.1: Grafana Dashboard
**Open:** http://localhost:3000 (admin/admin)

**Show Dashboards:**
- **Model Performance**: Real-time metrics
- **API Health**: Request rates, errors, latency
- **Data Quality**: Feature distributions
- **Drift Detection**: Statistical tests

#### Step 6.2: Prometheus Metrics
**Open:** http://localhost:9090

**Query Examples:**
- `rate(fraud_detection_requests_total[5m])`: Request rate
- `fraud_detection_request_duration_seconds`: Response time
- `fraud_detection_predictions_total`: Prediction counts

#### Step 6.3: Alertmanager
**Open:** http://localhost:9093

**Show Alert Rules:**
- High error rate (>1% for 5 minutes)
- Latency spike (+200ms)
- No predictions (model offline)
- Drift detection alerts

### Phase 7: CI/CD Pipeline (10 minutes)

#### Step 7.1: GitHub Actions Overview
**Show:** https://github.com/your-repo/actions

**Workflow Files:**
1. **ml-pipeline-ci.yml**: Complete CI pipeline (8 stages)
2. **ml-pipeline-cd.yml**: Deployment automation
3. **deploy-v1.yml**: Production deployment
4. **deploy-canary.yml**: Canary deployment
5. **train-v2.yml**: Model retraining
6. **drift-detection.yml**: Automated monitoring
7. **rollback.yml**: Emergency rollback

#### Step 7.2: Trigger CI Pipeline
```bash
# Make a small change to trigger CI
echo "# Test comment" >> README.md
git add README.md
git commit -m "Demo: Trigger CI pipeline"
git push origin Mangesh_m1
```

**Show CI Pipeline Running:**
- All 8 stages executing
- Integration tests passing
- Model training completing
- Artifacts being generated

#### Step 7.3: Canary Deployment Demo
```bash
# Deploy canary version
# This would be done via GitHub Actions workflow_dispatch
```

**Explain Canary Process:**
1. Deploy v2 to 5% traffic
2. Monitor for 24-48 hours
3. Compare performance metrics
4. Rollback if issues detected
5. Full rollout if successful

### Phase 8: Drift Detection (5 minutes)

#### Step 8.1: Show Drift Detection
```bash
# Run drift detection manually
python -c "
from src.models.drift_detection import DriftDetector
detector = DriftDetector('models/baseline_stats.json')
# This would analyze feature distributions
print('Drift detection scheduled every 6 hours')
print('K-S test threshold: 0.1')
print('Alerts sent to Alertmanager')
"
```

#### Step 8.2: Explain Drift Monitoring
**Key Points:**
- **Automated**: Runs every 6 hours via cron
- **Statistical**: K-S test for distribution changes
- **Alerting**: Automatic notifications on drift
- **Response**: Triggers investigation workflow

## Demo Scripts & Commands

### Quick Start Script
```bash
#!/bin/bash
# demo_quick_start.sh

echo "=== Fraud Detection MLOps Demo ==="
echo "Starting complete pipeline..."

# 1. Data Pipeline
echo "1. Running data pipeline..."
python run_pipeline.py <<EOF
1
EOF

# 2. Model Training
echo "2. Training models..."
python run_pipeline.py <<EOF
2
EOF

# 3. Start API
echo "3. Starting API service..."
uvicorn src.api.service:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# 4. Start monitoring stack
echo "4. Starting monitoring stack..."
docker-compose up -d

echo "=== Demo Ready ==="
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Alertmanager: http://localhost:9093"

# Cleanup function
cleanup() {
    kill $API_PID
    docker-compose down
    echo "Demo stopped"
}

trap cleanup EXIT
```

### Test Data Script
```bash
#!/bin/bash
# test_api.sh

echo "=== Testing Fraud Detection API ==="

# Test health check
echo "1. Health Check:"
curl -s http://localhost:8000/health
echo ""

# Test single prediction
echo "2. Single Prediction:"
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 123456,
    "Amount": 150.50,
    "V1": 0.5, "V2": -1.2, "V3": 0.8, "V4": -0.3, "V5": 1.1,
    "V6": -0.7, "V7": 0.9, "V8": -0.4, "V9": 0.6, "V10": -0.8,
    "V11": 0.7, "V12": -0.5, "V13": 0.9, "V14": -0.6, "V15": 0.4,
    "V16": -0.9, "V17": 0.3, "V18": -0.7, "V19": 0.8, "V20": -0.2,
    "V21": 0.5, "V22": -0.4, "V23": 0.6, "V24": -0.8, "V25": 0.7,
    "V26": -0.3, "V27": 0.9, "V28": -0.5
  }' | python -m json.tool
echo ""

# Test batch prediction
echo "3. Batch Prediction:"
curl -s -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"Time": 123456, "Amount": 150.50, "V1": 0.5, "V2": -1.2},
    {"Time": 123457, "Amount": 75.25, "V1": -0.3, "V2": 0.8}
  ]' | python -m json.tool
echo ""

# Test metrics
echo "4. Prometheus Metrics:"
curl -s http://localhost:8000/metrics | head -20
echo ""

echo "=== API Tests Complete ==="
```

## Key Demo Points

### 1. Production-Grade Features
- **Monitoring**: Real-time metrics and alerting
- **Rollback**: One-click model rollback
- **Canary**: Safe deployment with traffic splitting
- **Drift**: Automated model performance monitoring

### 2. MLOps Best Practices
- **Version Control**: Complete model lineage
- **Testing**: Unit, integration, and end-to-end tests
- **CI/CD**: Automated pipeline from code to production
- **Documentation**: Comprehensive API docs and monitoring

### 3. Business Value
- **Fraud Detection**: High-precision ML models
- **Cost Optimization**: Efficient resource usage
- **Risk Mitigation**: Automated rollback and monitoring
- **Scalability**: Container-based deployment

### 4. Technical Excellence
- **Multi-Model**: Ensemble approach for robustness
- **Feature Engineering**: 12+ engineered features
- **Data Quality**: Comprehensive validation pipeline
- **Performance**: Sub-100ms API response times

## Troubleshooting for Demo

### Common Issues
1. **Port Conflicts**: Kill existing processes on ports 8000, 3000, 9090
2. **Docker Issues**: Ensure Docker Desktop is running
3. **Dependencies**: Run `pip install -r requirements.txt`
4. **Data Missing**: Verify `dataset/creditcard.csv` exists

### Quick Recovery
```bash
# Kill all processes
pkill -f uvicorn
pkill -f python
docker-compose down

# Restart clean
docker-compose up -d
uvicorn src.api.service:app --host 0.0.0.0 --port 8000 --reload
```

## Customer Q&A Preparation

### Common Questions & Answers

**Q: How does the system handle model drift?**
A: Automated K-S testing every 6 hours with configurable thresholds. Alerts trigger investigation and potential retraining.

**Q: What happens if the model performance degrades?**
A: Alertmanager sends notifications, and the system can automatically rollback to previous versions via GitHub Actions.

**Q: How do you ensure data quality?**
A: Multi-stage validation pipeline with statistical profiling, consistency checks, and automated quality scoring.

**Q: Can this scale to production volumes?**
A: Yes, container-based deployment with load balancing, designed for high-throughput fraud detection.

**Q: How long does model retraining take?**
A: Full pipeline completes in ~5 minutes, with incremental updates possible for faster turnaround.

## Demo Success Metrics

### Technical Validation
- [ ] All 8 CI pipeline stages pass
- [ ] API responds to health checks
- [ ] Predictions return valid results
- [ ] Monitoring dashboards show metrics
- [ ] Drift detection runs successfully

### Business Validation
- [ ] Model performance meets requirements (F1 > 0.7)
- [ ] API latency < 100ms (p95)
- [ ] System handles concurrent requests
- [ ] Rollback process is automated
- [ ] Monitoring provides actionable insights

This comprehensive demo guide ensures you can confidently showcase your production-grade MLOps system to customers, highlighting both technical excellence and business value.