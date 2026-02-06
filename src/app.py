from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import uvicorn
from pydantic import BaseModel
import logging
import os

# --- Setup ---
# --- Setup ---
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="Fraud Detection API", version="1.0")

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

# Paths (Assuming local artifacts for this POC)
SCALER_PATH = "artifacts/scaler.pkl"
MODEL_PATH = "artifacts/models/XGBoost.pkl"
DRIFT_REPORT_PATH = "artifacts/drift_report.html"

# Load Artifacts
logging.info("Loading artifacts...")
try:
    if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH):
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        logging.info("Artifacts loaded successfully.")
    else:
        logging.warning("Artifacts not found! API will not work until training is complete.")
        scaler = None
        model = None
except Exception as e:
    logging.error(f"Error loading artifacts: {e}")
    scaler = None
    model = None

# --- Input Validation ---
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.get("/dashboard/drift", response_class=HTMLResponse)
async def drift_dashboard():
    if os.path.exists(DRIFT_REPORT_PATH):
        with open(DRIFT_REPORT_PATH, "r") as f:
            return f.read()
    return HTMLResponse(content="<h1>Drift Report Not Found</h1><p>Run the pipeline to generate it.</p>", status_code=404)

@app.post("/predict")
def predict(transaction: Transaction):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        # Convert input to dataframe
        data_dict = transaction.dict()
        df = pd.DataFrame([data_dict])
        
        # 1. Feature Engineering (Scaling Amount)
        # Reshape Amount because scaler expects 2D array, but we are replacing 1 column
        # Note: In our pipeline we trained scaler on the 'Amount' column.
        # We must apply it exactly the same.
        
        # Ensure column order matches training
        expected_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]
        
        # Reorder to match training
        df = df[expected_columns]
        
        # Scale Amount
        df['Amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))

        # 2. Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "is_fraud": bool(prediction == 1),
            "probability": float(probability)
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
