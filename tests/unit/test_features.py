"""Unit tests for feature engineering and consistency"""
import pytest
import pandas as pd
import numpy as np
import logging

from src.features.engineering import FeatureEngineer
from src.features.consistency import FeatureConsistencyChecker

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

class TestFeatureEngineering:
    """Test feature engineering module"""
    
    def test_feature_engineer_initialization(self, test_config):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer(test_config)
        assert engineer.scaling_method == 'StandardScaler'
    
    def test_feature_engineering(self, test_config, sample_data):
        """Test feature engineering transformations"""
        engineer = FeatureEngineer(test_config)
        df_engineered = engineer.engineer_features(sample_data)
        
        # Check new features were created
        assert len(df_engineered.columns) > len(sample_data.columns)
        assert 'Amount_log' in df_engineered.columns
        assert 'Time_hour' in df_engineered.columns
        assert 'V_mean' in df_engineered.columns
        
        # Check no NaN introduced
        assert df_engineered.isna().sum().sum() < len(df_engineered) * 0.01
    
    def test_feature_scaling(self, test_config, sample_data):
        """Test feature scaling"""
        engineer = FeatureEngineer(test_config)
        df_engineered = engineer.engineer_features(sample_data)
        df_scaled = engineer.scale_features(df_engineered, fit=True)
        
        # Check scaling applied
        assert df_scaled.shape == df_engineered.shape
        
        # Check mean and std for scaled features (should be ~0 and ~1)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Time', 'Class']:
                mean_val = abs(df_scaled[col].mean())
                std_val = df_scaled[col].std()
                # Allow some tolerance
                assert mean_val < 1.0
                assert 0.5 < std_val < 1.5
    
    def test_data_splitting(self, test_config, sample_data):
        """Test train/test splitting"""
        engineer = FeatureEngineer(test_config)
        df_engineered = engineer.engineer_features(sample_data)
        df_scaled = engineer.scale_features(df_engineered, fit=True)
        
        X_train, X_test, y_train, y_test = engineer.split_data(df_scaled)
        
        # Check split ratios
        assert len(X_test) / len(df_scaled) == pytest.approx(0.2, abs=0.05)
        
        # Check no leakage
        assert len(set(X_train.index) & set(X_test.index)) == 0

class TestFeatureConsistency:
    """Test feature consistency checks"""
    
    def test_consistency_checker_initialization(self, test_config):
        """Test FeatureConsistencyChecker initialization"""
        checker = FeatureConsistencyChecker(test_config)
        assert checker.config is not None
    
    def test_feature_parity_check(self, test_config, sample_data):
        """Test feature parity verification"""
        engineer = FeatureEngineer(test_config)
        df_engineered = engineer.engineer_features(sample_data)
        
        checker = FeatureConsistencyChecker(test_config)
        parity_valid, report = checker.check_feature_parity(sample_data, df_engineered)
        
        assert parity_valid
        assert report['checks']['row_count_preserved']
        assert report['checks']['original_columns_intact']
    
    def test_train_test_consistency(self, test_config, sample_data):
        """Test train/test consistency checks"""
        engineer = FeatureEngineer(test_config)
        df_engineered = engineer.engineer_features(sample_data)
        df_scaled = engineer.scale_features(df_engineered, fit=True)
        
        X_train, X_test, y_train, y_test = engineer.split_data(df_scaled)
        
        checker = FeatureConsistencyChecker(test_config)
        consistent, report = checker.check_train_test_consistency(
            X_train, X_test, y_train, y_test
        )
        
        assert consistent
        assert report['checks']['same_columns']
        assert report['checks']['no_data_leakage']
