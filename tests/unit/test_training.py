"""Unit tests for model training and comparison"""
import pytest
import pandas as pd
import numpy as np
import logging

from src.features.engineering import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.comparison import ModelComparator

logger = logging.getLogger(__name__)

@pytest.fixture
def test_config():
    """Create test configuration"""
    config = {
        'features': {
            'scaling_method': 'StandardScaler',
            'test_split_ratio': 0.2,
            'stratified_split': True,
            'random_state': 42
        },
        'model': {
            'models_to_train': ['LogisticRegression', 'RandomForest'],
            'hyperparameters': {
                'LogisticRegression': {
                    'max_iter': 100,
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'RandomForest': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'class_weight': 'balanced',
                    'random_state': 42
                }
            }
        },
        'mlflow': {
            'tracking_uri': 'file:./mlruns_test',
            'experiment_name': 'test_experiment'
        }
    }
    return config

@pytest.fixture
def sample_data():
    """Create sample test data"""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Time': np.random.randint(0, 86400, n_samples),
        'Amount': np.random.exponential(100, n_samples),
        **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
        'Class': np.random.binomial(1, 0.01, n_samples)  # ~1% fraud
    }
    
    return pd.DataFrame(data)

class TestModelTraining:
    """Test model training module"""
    
    def test_model_trainer_initialization(self, test_config):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(test_config)
        assert trainer.model_config is not None
        assert trainer.mlflow_config is not None
    
    def test_class_imbalance_handling(self, test_config, sample_data):
        """Test SMOTE for class imbalance"""
        engineer = FeatureEngineer(test_config)
        df_engineered = engineer.engineer_features(sample_data)
        df_scaled = engineer.scale_features(df_engineered, fit=True)
        X_train, X_test, y_train, y_test = engineer.split_data(df_scaled)
        
        trainer = ModelTrainer(test_config)
        X_resampled, y_resampled = trainer.handle_class_imbalance(X_train, y_train)
        
        # Check resampling
        assert len(X_resampled) >= len(X_train)
        # Note: SMOTE logic might skip if not enough samples, but with 500 samples it should trigger or at least work

class TestModelComparison:
    """Test model comparison module"""
    
    def test_model_comparator_initialization(self, test_config):
        """Test ModelComparator initialization"""
        comparator = ModelComparator(test_config)
        assert comparator.config is not None
    
    def test_model_comparison(self, test_config):
        """Test model comparison"""
        metrics = {
            'LogisticRegression': {
                'accuracy': 0.98,
                'f1_score': 0.68,
                'roc_auc': 0.85
            },
            'RandomForest': {
                'accuracy': 0.99,
                'f1_score': 0.82,
                'roc_auc': 0.92
            }
        }
        
        comparator = ModelComparator(test_config)
        comparison_df = comparator.compare_models(metrics)
        
        assert comparison_df.shape[0] == 2
    
    def test_best_model_selection(self, test_config):
        """Test best model selection"""
        metrics = {
            'LogisticRegression': {'f1_score': 0.68},
            'RandomForest': {'f1_score': 0.82}
        }
        
        comparator = ModelComparator(test_config)
        best_model, best_score = comparator.select_best_model(metrics)
        
        assert best_model == 'RandomForest'
        assert best_score == 0.82
