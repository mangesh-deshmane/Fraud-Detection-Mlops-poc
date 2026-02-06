"""FastAPI Prediction API for fraud detection"""
import logging
from typing import List, Dict, Any
from pathlib import Path
import pickle
import json

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn
import time

from src.features.engineering import FeatureEngineer
from src.utils.logger import setup_logging
from src.api.metrics import (
    track_prediction, track_api_request, get_metrics,
    prediction_counter, prediction_latency, api_requests_total, api_request_latency
)

# Setup logging
logger = setup_logging()


# Pydantic models for request/response
class TransactionFeatures(BaseModel):
    """Input features for a single transaction"""
    Time: int = Field(..., description="Transaction time")
    Amount: float = Field(..., description="Transaction amount")
    V1: float = Field(..., description="Feature V1")
    V2: float = Field(..., description="Feature V2")
    V3: float = Field(..., description="Feature V3")
    V4: float = Field(..., description="Feature V4")
    V5: float = Field(..., description="Feature V5")
    V6: float = Field(..., description="Feature V6")
    V7: float = Field(..., description="Feature V7")
    V8: float = Field(..., description="Feature V8")
    V9: float = Field(..., description="Feature V9")
    V10: float = Field(..., description="Feature V10")
    V11: float = Field(..., description="Feature V11")
    V12: float = Field(..., description="Feature V12")
    V13: float = Field(..., description="Feature V13")
    V14: float = Field(..., description="Feature V14")
    V15: float = Field(..., description="Feature V15")
    V16: float = Field(..., description="Feature V16")
    V17: float = Field(..., description="Feature V17")
    V18: float = Field(..., description="Feature V18")
    V19: float = Field(..., description="Feature V19")
    V20: float = Field(..., description="Feature V20")
    V21: float = Field(..., description="Feature V21")
    V22: float = Field(..., description="Feature V22")
    V23: float = Field(..., description="Feature V23")
    V24: float = Field(..., description="Feature V24")
    V25: float = Field(..., description="Feature V25")
    V26: float = Field(..., description="Feature V26")
    V27: float = Field(..., description="Feature V27")
    V28: float = Field(..., description="Feature V28")


class BatchTransactionFeatures(BaseModel):
    """Batch input for multiple transactions"""
    transactions: List[TransactionFeatures]


class PredictionResponse(BaseModel):
    """Response for a single prediction"""
    prediction: int = Field(..., description="0=Normal, 1=Fraud")
    confidence: float = Field(..., description="Confidence score (0-1)")
    risk_level: str = Field(..., description="Low/Medium/High")
    model_name: str = Field(..., description="Model used for prediction")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_name: str
    model_version: str


class PredictionAPI:
    """FastAPI application for fraud detection predictions"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize the prediction API
        
        Args:
            config_path: Path to configuration file
        """
        import yaml
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.app = FastAPI(
            title="Fraud Detection API",
            description="Real-time fraud detection predictions",
            version="1.0.0"
        )
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Load model
        self.model = None
        self.model_name = None
        self.load_model()
        
        # Setup routes
        self.setup_routes()
        
        logger.info("Prediction API initialized successfully")
    
    def load_model(self):
        """Load the trained model"""
        model_path = Path("models/best_model.pkl")
        
        if not model_path.exists():
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Using placeholder model - ensure to train and save model first")
            return
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Try to get model name
            model_info_path = Path("models/model_info.json")
            if model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    info = json.load(f)
                    self.model_name = info.get('model_name', 'Unknown')
            
            # Load scaler
            scaler_path = Path("models/scaler.pkl")
            if scaler_path.exists():
                self.feature_engineer.load_scaler(str(scaler_path))
            else:
                logger.warning("Scaler not found. Feature scaling might fail.")

            logger.info(f"Model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                model_name=self.model_name or "Not loaded",
                model_version="1.0"
            )
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return PlainTextResponse(get_metrics().decode('utf-8'))
        
        @self.app.post("/predict", response_model=PredictionResponse)
        @track_api_request(endpoint="/predict", method="POST")
        async def predict(transaction: TransactionFeatures = Body(...)):
            """Predict fraud for a single transaction"""
            try:
                if self.model is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                # Convert to DataFrame
                df = pd.DataFrame([transaction.dict()])
                
                # Apply feature engineering
                df_engineered = self.feature_engineer.engineer_features(df)
                
                # Scale features
                df_scaled = self.feature_engineer.scale_features(df_engineered, fit=False)
                
                # Make prediction
                prediction = self.model.predict(df_scaled)[0]
                confidence = self.model.predict_proba(df_scaled)[0][prediction]
                
                # Determine risk level
                if confidence > 0.8:
                    risk_level = "High"
                elif confidence > 0.6:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                return PredictionResponse(
                    prediction=int(prediction),
                    confidence=float(confidence),
                    risk_level=risk_level,
                    model_name=self.model_name or "Unknown"
                )
            
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict_batch", response_model=BatchPredictionResponse)
        async def predict_batch(batch: BatchTransactionFeatures = Body(...)):
            """Predict fraud for multiple transactions"""
            import time
            
            try:
                if self.model is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                start_time = time.time()
                
                # Convert to DataFrame
                transactions_list = [t.dict() for t in batch.transactions]
                df = pd.DataFrame(transactions_list)
                
                # Apply feature engineering
                df_engineered = self.feature_engineer.engineer_features(df)
                
                # Scale features
                df_scaled = self.feature_engineer.scale_features(df_engineered, fit=False)
                
                # Make predictions
                predictions = self.model.predict(df_scaled)
                confidences = self.model.predict_proba(df_scaled)[:, predictions]
                
                # Build responses
                responses = []
                for pred, conf in zip(predictions, confidences):
                    if conf > 0.8:
                        risk_level = "High"
                    elif conf > 0.6:
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"
                    
                    responses.append(PredictionResponse(
                        prediction=int(pred),
                        confidence=float(conf),
                        risk_level=risk_level,
                        model_name=self.model_name or "Unknown"
                    ))
                
                processing_time = (time.time() - start_time) * 1000
                
                return BatchPredictionResponse(
                    predictions=responses,
                    processing_time_ms=processing_time
                )
            
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model_info")
        async def model_info():
            """Get model information"""
            return {
                "model_name": self.model_name or "Unknown",
                "model_version": "1.0",
                "features": self.feature_engineer.get_feature_info(
                    pd.DataFrame([[0]*29], columns=[f"V{i}" for i in range(1, 29)] + ["Time", "Amount"])
                )
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the API server
        
        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Whether to enable auto-reload
        """
        logger.info(f"Starting Fraud Detection API on {host}:{port}")
        logger.info("Documentation available at http://localhost:8000/docs")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


def create_app(config_path: str = "src/config/config.yaml") -> FastAPI:
    """Create FastAPI application
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI application instance
    """
    api = PredictionAPI(config_path)
    return api.app


if __name__ == "__main__":
    # Create and run API
    api = PredictionAPI()
    api.run()
