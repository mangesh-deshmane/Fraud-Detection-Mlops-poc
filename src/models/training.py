"""Model training module with MLFlow integration"""
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pickle
import traceback

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate fraud detection models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.mlflow_config = config.get('mlflow', {})
        self.models = {}
        self.metrics = {}
        
        # Setup MLFlow - use local file-based backend for CI/CD
        mlflow_uri = self.mlflow_config.get('tracking_uri', 'file:./mlruns')
        if mlflow_uri.startswith('http'):
            # For CI/CD environments, fall back to local backend
            mlflow_uri = 'file:./mlruns'
        
        mlflow.set_tracking_uri(mlflow_uri)
        self.experiment_name = self.mlflow_config.get('experiment_name', 'fraud_detection_v1')
        
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLFlow configured: {mlflow_uri}")
        except Exception as e:
            logger.warning(f"âš ï¸  MLFlow setup warning: {e}")
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for class imbalance handling
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Balanced X_train, y_train
        """
        logger.info("Handling class imbalance with SMOTE...")
        
        # Determine appropriate k_neighbors for SMOTE based on sample size
        minority_count = y_train.sum()
        k_neighbors = 5
        
        if minority_count <= 1:
            logger.warning(f"Minority class has only {minority_count} sample(s). Skipping SMOTE.")
            return X_train, y_train
            
        if minority_count <= 6:
            k_neighbors = max(1, int(minority_count - 1))
            logger.info(f"Adjusting SMOTE k_neighbors to {k_neighbors} due to small sample size ({minority_count} fraud samples)")
            
        smote = SMOTE(random_state=self.config['features'].get('random_state', 42), k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"  Original distribution: Fraud={y_train.sum()}, Normal={(1-y_train).sum()}")
        logger.info(f"  After SMOTE: Fraud={y_train_resampled.sum()}, Normal={(1-y_train_resampled).sum()}")
        
        return X_train_resampled, y_train_resampled
    
    def train_model(self, 
                   model_name: str, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series) -> Any:
        """Train a single model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model object
        """
        logger.info(f"Training {model_name}...")
        
        hyperparams = self.model_config.get('hyperparameters', {}).get(model_name, {})
        
        if model_name == 'LogisticRegression':
            model = LogisticRegression(**hyperparams)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**hyperparams)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(**hyperparams)
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(**hyperparams)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.fit(X_train, y_train)
        logger.info(f"{model_name} trained successfully")
        
        return model
    
    def evaluate_model(self, 
                      model: Any, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Calculate PR AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)
        
        logger.info(f"\n  {model_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"    {metric}: {value:.4f}")
        
        return metrics
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        apply_smote: bool = True) -> Dict[str, Any]:
        """Train and evaluate all configured models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            apply_smote: Whether to apply SMOTE for class imbalance
            
        Returns:
            Dictionary with models and metrics
        """
        logger.info("=" * 70)
        logger.info("STARTING MODEL TRAINING WITH MLFLOW")
        logger.info("=" * 70)
        
        # Handle class imbalance
        if apply_smote:
            X_train_processed, y_train_processed = self.handle_class_imbalance(X_train, y_train)
        else:
            X_train_processed, y_train_processed = X_train, y_train
        
        results = {
            'models': {},
            'metrics': {},
            'run_ids': {}
        }
        
        models_to_train = self.model_config.get('models_to_train', ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM'])
        
        for model_name in models_to_train:
            try:
                with mlflow.start_run(run_name=f"{model_name}_run"):
                    logger.info(f"\n{'='*70}")
                    logger.info(f"Training: {model_name}")
                    logger.info(f"{'='*70}")
                    
                    # Log hyperparameters
                    hyperparams = self.model_config.get('hyperparameters', {}).get(model_name, {})
                    for param_name, param_value in hyperparams.items():
                        mlflow.log_param(f"{model_name}_{param_name}", param_value)
                    
                    # Train model
                    model = self.train_model(model_name, X_train_processed, y_train_processed)
                    
                    # Evaluate model
                    metrics = self.evaluate_model(model, X_test, y_test, model_name)
                    
                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
                    
                    # Store results
                    self.models[model_name] = model
                    self.metrics[model_name] = metrics
                    results['models'][model_name] = model
                    results['metrics'][model_name] = metrics
                    results['run_ids'][model_name] = mlflow.active_run().info.run_id
                    
                    # Log model to MLFlow
                    if model_name == 'XGBoost':
                        mlflow.xgboost.log_model(model, f"{model_name}_model")
                    else:
                        mlflow.sklearn.log_model(model, f"{model_name}_model")
                    
                    logger.info(f"{model_name} training completed")
                    logger.info(f"   MLFlow Run ID: {mlflow.active_run().info.run_id}")
                    
            except Exception as e:
                logger.error(f"âŒ Failed to train {model_name}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info("\n" + "=" * 70)
        logger.info("MODEL TRAINING COMPLETED")
        logger.info("=" * 70)
        
        return results
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models to disk
        
        Args:
            output_dir: Directory to save models
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = Path(output_dir) / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} to {model_path}")
    
    def save_metrics(self, output_path: str) -> None:
        """Save metrics to JSON file
        
        Args:
            output_path: Path to save metrics JSON
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def get_best_model(self) -> Tuple[str, Any, Dict[str, float]]:
        """Get the best model based on F1 score
        
        Returns:
            Tuple of (model_name, model, metrics)
        """
        best_model_name = None
        best_f1_score = 0
        best_metrics = None
        
        for model_name, metrics in self.metrics.items():
            f1 = metrics.get('f1_score', 0)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_name = model_name
                best_metrics = metrics
        
        if best_model_name:
            return best_model_name, self.models[best_model_name], best_metrics
        
        raise ValueError("No models trained successfully")
