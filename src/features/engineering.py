"""Feature engineering module for fraud detection"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering and preprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('features', {})
        self.scaler = None
        self.scaling_method = self.feature_config.get('scaling_method', 'StandardScaler')
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        df_engineered = df.copy()
        
        # Log-transform Amount (positive skew)
        logger.info("  - Applying log transformation to Amount column")
        df_engineered['Amount_log'] = np.log1p(df_engineered['Amount'])
        
        # Time-based features
        logger.info("  - Creating time-based features")
        df_engineered['Time_hour'] = (df_engineered['Time'] // 3600) % 24
        df_engineered['Time_day_period'] = pd.cut(
            df_engineered['Time_hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        ).cat.codes
        
        # Interaction features
        logger.info("  - Creating interaction features")
        # Amount with V1-V5 (top features typically)
        for col in ['V1', 'V2', 'V3', 'V4', 'V5']:
            df_engineered[f'{col}_Amount_interaction'] = df_engineered[col] * df_engineered['Amount']
        
        # Statistical features
        logger.info("  - Creating statistical features")
        v_cols = [col for col in df_engineered.columns if col.startswith('V')]
        df_engineered['V_mean'] = df_engineered[v_cols].mean(axis=1)
        df_engineered['V_std'] = df_engineered[v_cols].std(axis=1)
        df_engineered['V_max'] = df_engineered[v_cols].max(axis=1)
        df_engineered['V_min'] = df_engineered[v_cols].min(axis=1)
        
        logger.info(f"Feature engineering completed. Generated {len(df_engineered.columns) - len(df.columns)} new features")
        return df_engineered
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using configured method
        
        Args:
            df: Input dataframe
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled dataframe
        """
        logger.info(f"Scaling features using {self.scaling_method}...")
        
        # Select numeric columns to scale (exclude Time and Class)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in ['Class', 'Time']]
        
        if fit:
            # Initialize scaler
            if self.scaling_method == 'StandardScaler':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'RobustScaler':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            # Fit and transform
            df_scaled = df.copy()
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            logger.info(f"Scaler fitted and applied. Scaled {len(cols_to_scale)} features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call scale_features with fit=True first")
            
            df_scaled = df.copy()
            df_scaled[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            logger.info(f"Scaler applied. Scaled {len(cols_to_scale)} features")
        
        return df_scaled
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets
        
        Args:
            df: Input dataframe
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data into train and test sets...")
        
        test_split = self.feature_config.get('test_split_ratio', 0.2)
        stratified = self.feature_config.get('stratified_split', True)
        random_state = self.feature_config.get('random_state', 42)
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        if stratified:
            logger.info(f"  Using stratified split (test_size={test_split})")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_split,
                stratify=y,
                random_state=random_state
            )
        else:
            logger.info(f"  Using random split (test_size={test_split})")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_split,
                random_state=random_state
            )
        
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        logger.info(f"   Class distribution (Train): Fraud={y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%), Normal={(1-y_train).sum()} ({(1-y_train).sum()/len(y_train)*100:.2f}%)")
        logger.info(f"   Class distribution (Test): Fraud={y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%), Normal={(1-y_test).sum()} ({(1-y_test).sum()/len(y_test)*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get feature information for documentation
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with feature information
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_info = {
            'total_features': len(df.columns),
            'numeric_features': len(numeric_cols),
            'feature_names': df.columns.tolist(),
            'numeric_feature_names': numeric_cols,
        }
        
        return feature_info
    
    def save_scaler(self, path: str) -> None:
        """Save scaler to disk
        
        Args:
            path: Output path
        """
        if self.scaler is None:
            logger.warning("No scaler to save")
            return
            
        import pickle
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {path}")
        
    def load_scaler(self, path: str) -> None:
        """Load scaler from disk
        
        Args:
            path: Input path
        """
        import pickle
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler not found at {path}")
            
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {path}")
