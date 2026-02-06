"""Integration tests combining multiple components"""
import pytest
import pandas as pd
import numpy as np
import logging

from src.features.engineering import FeatureEngineer
from src.features.consistency import FeatureConsistencyChecker
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
            'models_to_train': ['LogisticRegression'],
            'hyperparameters': {
                'LogisticRegression': {
                    'max_iter': 100,
                    'class_weight': 'balanced',
                    'random_state': 42
                }
            }
        },
        'mlflow': {
            'tracking_uri': 'file:./mlruns_integration',
            'experiment_name': 'integration_test'
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
        'Class': np.random.binomial(1, 0.01, n_samples)
    }
    return pd.DataFrame(data)

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_end_to_end_pipeline(self, test_config, sample_data):
        """Test complete pipeline flow"""
        # Step 1: Feature Engineering
        engineer = FeatureEngineer(test_config)
        df_engineered = engineer.engineer_features(sample_data)
        df_scaled = engineer.scale_features(df_engineered, fit=True)
        X_train, X_test, y_train, y_test = engineer.split_data(df_scaled)
        
        # Step 2: Consistency Checks
        checker = FeatureConsistencyChecker(test_config)
        parity_valid, _ = checker.check_feature_parity(sample_data, df_engineered)
        consistency_valid, _ = checker.check_train_test_consistency(
            X_train, X_test, y_train, y_test
        )
        
        # Step 3: Model Training
        trainer = ModelTrainer(test_config)
        
        # Train simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=100, random_state=42)
        X_train_resampled, y_train_resampled = trainer.handle_class_imbalance(X_train, y_train)
        model.fit(X_train_resampled, y_train_resampled)
        
        metrics = trainer.evaluate_model(model, X_test, y_test, 'LogisticRegression')
        
        # Step 4: Model Comparison
        comparator = ModelComparator(test_config)
        best_model_name, best_score = comparator.select_best_model({
            'LogisticRegression': metrics
        })
        
        # Verify end-to-end
        assert parity_valid
        assert consistency_valid
        assert best_model_name == 'LogisticRegression'
        assert best_score >= 0.0
