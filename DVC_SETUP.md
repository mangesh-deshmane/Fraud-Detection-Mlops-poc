# DVC Setup and Usage Guide

## What Changed

We've migrated from GitHub LFS to **DVC (Data Version Control)** to manage the 144MB credit card dataset. This resolves the GitHub LFS quota issues and provides a production-ready solution for Docker/EKS deployments.

## Quick Start

### 1. Install DVC
```bash
pip install dvc
```

### 2. Pull the Dataset
```bash
# Pull dataset from DVC remote
python -m dvc pull dataset/creditcard.csv.dvc

# Verify dataset
ls -lh dataset/creditcard.csv  # Should show 144MB
```

### 3. Run the Pipeline
```bash
python run_pipeline.py
```

## For New Team Members

When you clone the repository:

```bash
git clone https://github.com/mangesh-deshmane/Fraud-Detection-Mlops-poc.git
cd Fraud-Detection-Mlops-poc
pip install -r requirements.txt
python -m dvc pull dataset/creditcard.csv.dvc
```

## For CI/CD (GitHub Actions)

The workflow automatically handles dataset download with fallback strategies:
1. Try DVC pull (if remote configured)
2. Merge local chunks (if available in repo)
3. Use sample dataset (for quick CI testing)

No additional configuration needed for basic CI testing.

## For Docker/EKS Deployment

### Option 1: Use Sample Dataset (Testing)
```bash
docker build -t fraud-detection .
# Dockerfile will use sample dataset automatically
```

### Option 2: Configure S3 Remote (Production)

```bash
# One-time setup: Configure S3 remote
dvc remote add -d production s3://your-bucket/dvc-storage
dvc remote modify production region us-east-1

# Push dataset to S3
dvc push -r production

# Update Dockerfile to use production remote
# Add before building: ENV AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx
```

### Option 3: Use Azure Blob Storage

```bash
dvc remote add -d azure azure://mycontainer/path
dvc remote modify azure account_name 'myaccount'
dvc push -r azure
```

## DVC Commands Reference

```bash
# Check DVC status
dvc status

# Pull specific file
dvc pull dataset/creditcard.csv.dvc

# Pull all DVC-tracked files
dvc pull

# Push to remote
dvc push

# Check remote configuration
dvc remote list
```

## Troubleshooting

### Dataset not found after pull
```bash
# Check if DVC remote is configured
dvc remote list

# If no remote, use local chunks
python simple_merge.py dataset_chunks "creditcard_part_*.csv" "creditcard.csv"
```

### Docker build fails
```bash
# Use sample dataset for local testing
# The Dockerfile automatically falls back to sample dataset if DVC pull fails
docker build -t fraud-detection .
```

### CI/CD fails
- The workflow uses sample dataset by default for fast CI testing
- For full dataset in CI, configure DVC remote and add credentials as GitHub secrets

## Files Modified

- `requirements.txt` - Added `dvc==3.50.0`
- `.gitignore` - Updated to track DVC config, exclude dataset
- `Dockerfile` - Added DVC pull step with fallbacks
- `.github/workflows/ml-pipeline-ci.yml` - Simplified dataset download
- `dataset/creditcard.csv.dvc` - DVC tracking file (tracked in Git)
- `.dvc/` - DVC configuration directory (tracked in Git)

## Production Deployment Checklist

- [ ] Configure DVC remote (S3/Azure/GCS)
- [ ] Push dataset to remote: `dvc push`
- [ ] Add cloud credentials to deployment environment
- [ ] Update Dockerfile with remote credentials
- [ ] Test Docker build with DVC pull
- [ ] Deploy to EKS/EC2

## Cost Considerations

- **Local Remote**: Free (uses local filesystem)
- **S3**: ~$0.023/GB/month (~$3.30/month for 144MB)
- **Azure Blob**: ~$0.018/GB/month (~$2.60/month for 144MB)
- **Google Cloud Storage**: ~$0.020/GB/month (~$2.88/month for 144MB)

All options are significantly cheaper than GitHub LFS and work seamlessly with Docker/Kubernetes.
