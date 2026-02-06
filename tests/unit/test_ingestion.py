"""Unit tests for data ingestion pipeline"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.ingestion.loader import DataLoader
from src.ingestion.validator import SchemaValidator
from src.ingestion.profiler import DataProfiler

# Create test config
@pytest.fixture
def test_config():
    return {
        'data': {
            'raw_path': 'dataset/creditcard.csv',
            'validated_path': 'data/validated/creditcard_validated.parquet',
            'profile_path': 'data/validated/profile.json'
        },
        'schema': {
            'columns': {
                'Time': 'int64',
                'V1': 'float64',
                'Amount': 'float64',
                'Class': 'int64'
            }
        },
        'quality': {
            'min_completeness': 0.95,
            'max_duplicates': 0.01,
            'outlier_threshold': 3.0,
            'min_rows': 1000
        }
    }

@pytest.fixture
def sample_df():
    """Create sample dataframe"""
    return pd.DataFrame({
        'Time': [0, 1, 2, 3, 4],
        'V1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'Amount': [100.0, 200.0, 300.0, 400.0, 500.0],
        'Class': [0, 0, 0, 0, 1]
    })

def test_schema_validator(test_config, sample_df):
    """Test schema validation"""
    validator = SchemaValidator(test_config)
    is_valid, errors = validator.validate_schema(sample_df)
    assert is_valid
    assert len(errors) == 0

def test_quality_validation(test_config, sample_df):
    """Test quality checks"""
    validator = SchemaValidator(test_config)
    is_valid, report = validator.validate_quality(sample_df)
    assert 'completeness' in report['checks']
    assert 'duplicates' in report['checks']
    assert report['checks']['completeness']['passed']

def test_data_profiler(test_config, sample_df):
    """Test data profiling"""
    profiler = DataProfiler(test_config)
    profile = profiler.profile(sample_df)
    assert profile['metadata']['rows'] == 5
    assert profile['metadata']['columns'] == 4
    assert 'Time' in profile['columns']
    assert 'V1' in profile['columns']

def test_data_loader_parquet(test_config, sample_df):
    """Test parquet save/load"""
    loader = DataLoader(test_config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/test.parquet"
        loader.save_parquet(sample_df, output_path)
        
        loaded_df = loader.load_parquet(output_path)
        assert len(loaded_df) == len(sample_df)
        assert list(loaded_df.columns) == list(sample_df.columns)
        assert loaded_df['Class'].tolist() == sample_df['Class'].tolist()

def test_duplicates_detection(test_config):
    """Test duplicate detection"""
    df = pd.DataFrame({
        'Time': [0, 1, 1, 2],
        'V1': [1.0, 2.0, 2.0, 3.0],
        'Amount': [100.0, 200.0, 200.0, 300.0],
        'Class': [0, 0, 0, 1]
    })
    
    validator = SchemaValidator(test_config)
    is_valid, report = validator.validate_quality(df)
    
    # Should have duplicates
    assert report['checks']['duplicates']['value'] > 0

def test_class_distribution(test_config):
    """Test class distribution reporting"""
    df = pd.DataFrame({
        'Time': list(range(100)),
        'V1': np.random.randn(100),
        'Amount': np.random.uniform(0, 1000, 100),
        'Class': [0] * 95 + [1] * 5  # 95% legitimate, 5% fraud
    })
    
    validator = SchemaValidator(test_config)
    is_valid, report = validator.validate_quality(df)
    
    assert 'class_distribution' in report['checks']
    assert 'class_imbalance_ratio' in report['checks']
