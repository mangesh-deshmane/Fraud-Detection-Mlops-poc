"""End-to-end ML training pipeline"""
import logging
import yaml
from pathlib import Path
from src.ingestion.pipeline import DataIngestionPipeline
from src.features.engineering import FeatureEngineer
from src.features.consistency import FeatureConsistencyChecker
from src.models.training import ModelTrainer
from src.models.comparison import ModelComparator

logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """Complete ML training pipeline orchestrator"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize ML pipeline
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_model_name = None
    
    def run(self) -> dict:
        """Execute complete ML pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("\n" + "=" * 70)
        logger.info("STARTING END-TO-END ML TRAINING PIPELINE")
        logger.info("=" * 70)
        
        try:
            # Step 1: Data Ingestion
            logger.info("\n[STEP 1/6] Data Ingestion")
            self._run_data_ingestion()
            
            # Step 2: Feature Engineering
            logger.info("\n[STEP 2/6] Feature Engineering")
            self._run_feature_engineering()
            
            # Step 3: Feature Consistency Checks
            logger.info("\n[STEP 3/6] Feature Consistency Checks (Parity Verification)")
            self._run_consistency_checks()
            
            # Step 4: Model Training with MLFlow
            logger.info("\n[STEP 4/6] Model Training with MLFlow")
            self._run_model_training()
            
            # Step 5: Model Comparison and Selection
            logger.info("\n[STEP 5/6] Model Comparison and Selection")
            self._run_model_comparison()
            
            # Step 6: Save Results
            logger.info("\n[STEP 6/6] Saving Results")
            self._save_results()
            
            logger.info("\n" + "=" * 70)
            logger.info("ML TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            
            return {
                'status': 'success',
                'best_model': self.best_model_name
            }
        
        except Exception as e:
            logger.error(f"\nâŒ Pipeline failed: {e}")
            raise
    
    def _run_data_ingestion(self):
        """Run data ingestion pipeline"""
        logger.info("  Loading and validating data...")
        pipeline = DataIngestionPipeline(self.config_path)
        self.data = pipeline.run()
        logger.info(f"  Data loaded: {self.data.shape}")
    
    def _run_feature_engineering(self):
        """Run feature engineering"""
        logger.info("  Applying feature engineering...")
        
        engineer = FeatureEngineer(self.config)
        self.engineer = engineer
        df_engineered = engineer.engineer_features(self.data)
        df_scaled = engineer.scale_features(df_engineered)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = engineer.split_data(df_scaled)
        logger.info(f"  Features engineered and data split")
    
    def _run_consistency_checks(self):
        """Run feature consistency checks"""
        logger.info("  Checking feature consistency and parity...")
        
        checker = FeatureConsistencyChecker(self.config)
        
        # Check parity
        engineer = FeatureEngineer(self.config)
        df_engineered = engineer.engineer_features(self.data)
        
        parity_valid, parity_report = checker.check_feature_parity(self.data, df_engineered)
        if not parity_valid:
            logger.warning("  âš ï¸  Parity check failed - review issues")
        
        # Check train/test consistency
        consistency_valid, consistency_report = checker.check_train_test_consistency(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        # Check distributions
        dist_valid, dist_report = checker.check_feature_distributions(
            self.X_train, self.X_test
        )
        
        # Save report
        report = {
            'parity': parity_report,
            'consistency': consistency_report,
            'distributions': dist_report
        }
        
        checker.save_consistency_report(
            report,
            "data/validated/feature_consistency_report.json"
        )
        
        logger.info(f"  Consistency checks completed (parity={parity_valid}, consistency={consistency_valid})")
    
    def _run_model_training(self):
        """Run model training with MLFlow"""
        logger.info("  Training models with MLFlow...")
        
        trainer = ModelTrainer(self.config)
        results = trainer.train_all_models(
            self.X_train, self.X_test, self.y_train, self.y_test,
            apply_smote=True
        )
        
        self.metrics = results['metrics']
        self.trainer = trainer
        
        logger.info(f"  {len(results['models'])} models trained successfully")
    
    def _run_model_comparison(self):
        """Run model comparison and selection"""
        logger.info("  Comparing models and selecting best...")
        
        comparator = ModelComparator(self.config)
        
        # Get rankings
        rankings = comparator.get_model_rankings(self.metrics)
        
        # Get best model
        best_model_info = comparator.get_best_model_info(rankings)
        self.best_model_name = best_model_info['name']
        
        # Register model
        best_model_obj = self.trainer.models[self.best_model_name]
        model_registry_name = comparator.register_best_model(
            self.best_model_name,
            best_model_obj,
            best_model_info['metrics']
        )
        
        # Save comparison report
        comparator.save_comparison_report(
            "models/model_comparison_report.json"
        )
        
        logger.info(f"  Best model selected: {self.best_model_name}")
        logger.info(f"     Registered as: {model_registry_name}")
    
    def _save_results(self):
        """Save training results"""
        logger.info("  Saving models and metadata...")
        
        # Save models
        Path("models").mkdir(exist_ok=True)
        self.trainer.save_models("models")
        if hasattr(self, 'engineer'):
            self.engineer.save_scaler("models/scaler.pkl")
        self.trainer.save_metrics("models/metrics.json")
        
        # Save model info
        import json
        model_info = {
            'best_model': self.best_model_name,
            'metrics': self.metrics.get(self.best_model_name, {}),
            'pipeline_version': '1.0'
        }
        
        with open("models/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"  Results saved to models/")


if __name__ == "__main__":
    # Setup logging
    import logging.config
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = MLTrainingPipeline()
    pipeline.run()
