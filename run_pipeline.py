"""
Quick runner script to execute the complete ML pipeline
Usage: python run_pipeline.py
"""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete ML pipeline"""
    
    print("\n" + "=" * 70)
    print(" FRAUD DETECTION - ML PIPELINE RUNNER")
    print("=" * 70)
    
    print("\n[MENU] Select what to run:")
    print("  1. Run complete end-to-end pipeline")
    print("  2. Run data ingestion only")
    print("  3. Run feature engineering only")
    print("  4. Run feature consistency checks")
    print("  5. Run model training with MLFlow")
    print("  6. Run model comparison and selection")
    print("  7. Start FastAPI prediction service")
    print("  8. Run integration tests")
    print("  9. Exit")
    
    choice = input("\nEnter your choice (1-9): ").strip()
    
    try:
        if choice == "1":
            print("\n[RUNNING] Complete end-to-end ML pipeline...")
            from src.pipelines.ml_pipeline import MLTrainingPipeline
            pipeline = MLTrainingPipeline()
            results = pipeline.run()
            print(f"\nPipeline completed. Best model: {results['best_model']}")
        
        elif choice == "2":
            print("\n[RUNNING] Data ingestion...")
            from src.ingestion.pipeline import DataIngestionPipeline
            pipeline = DataIngestionPipeline()
            df = pipeline.run()
            print(f"Data ingestion completed. Loaded {len(df)} records")
        
        elif choice == "3":
            print("\n[RUNNING] Feature engineering...")
            import pandas as pd
            import yaml
            
            with open('src/config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            df = pd.read_parquet(config['data']['validated_path'])
            
            from src.features.engineering import FeatureEngineer
            engineer = FeatureEngineer(config)
            df_eng = engineer.engineer_features(df)
            df_scaled = engineer.scale_features(df_eng)
            X_train, X_test, y_train, y_test = engineer.split_data(df_scaled)
            
            print(f"Feature engineering completed")
            print(f"   Training set: {X_train.shape}")
            print(f"   Test set: {X_test.shape}")
        
        elif choice == "4":
            print("\n[RUNNING] Feature consistency checks...")
            import pandas as pd
            import yaml
            
            with open('src/config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            df = pd.read_parquet(config['data']['validated_path'])
            
            from src.features.engineering import FeatureEngineer
            from src.features.consistency import FeatureConsistencyChecker
            
            engineer = FeatureEngineer(config)
            df_eng = engineer.engineer_features(df)
            df_scaled = engineer.scale_features(df_eng)
            X_train, X_test, y_train, y_test = engineer.split_data(df_scaled)
            
            checker = FeatureConsistencyChecker(config)
            parity_valid, parity_report = checker.check_feature_parity(df, df_eng)
            consistency_valid, consistency_report = checker.check_train_test_consistency(
                X_train, X_test, y_train, y_test
            )
            
            print(f"Consistency checks completed")
            print(f"   Parity valid: {parity_valid}")
            print(f"   Consistency valid: {consistency_valid}")
        
        elif choice == "5":
            print("\n[RUNNING] Model training with MLFlow...")
            import pandas as pd
            import yaml
            
            with open('src/config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            df = pd.read_parquet(config['data']['validated_path'])
            
            from src.features.engineering import FeatureEngineer
            from src.models.training import ModelTrainer
            
            engineer = FeatureEngineer(config)
            df_eng = engineer.engineer_features(df)
            df_scaled = engineer.scale_features(df_eng)
            X_train, X_test, y_train, y_test = engineer.split_data(df_scaled)
            
            trainer = ModelTrainer(config)
            results = trainer.train_all_models(X_train, X_test, y_train, y_test)
            trainer.save_models('models')
            engineer.save_scaler('models/scaler.pkl')
            trainer.save_metrics('models/metrics.json')
            
            print(f"Model training completed")
            print(f"   Models trained: {list(results['models'].keys())}")
            print(f"   MLFlow runs: {list(results['run_ids'].keys())}")
        
        elif choice == "6":
            print("\n[RUNNING] Model comparison and selection...")
            import json
            import yaml
            
            with open('src/config/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            with open('models/metrics.json', 'r') as f:
                metrics = json.load(f)
            
            from src.models.comparison import ModelComparator
            
            comparator = ModelComparator(config)
            rankings = comparator.get_model_rankings(metrics)
            best_info = comparator.get_best_model_info(rankings)
            
            print(f"Model comparison completed")
            print(f"   Best model: {best_info['name']}")
            print(f"   Composite score: {best_info['composite_score']:.4f}")
            
            comparator.save_comparison_report('models/model_comparison_report.json')
        
        elif choice == "7":
            print("\n[RUNNING] FastAPI Prediction Service...")
            print("   Starting server on http://0.0.0.0:8000")
            print("   API documentation: http://localhost:8000/docs")
            print("   Press Ctrl+C to stop\n")
            
            from src.api.service import PredictionAPI
            api = PredictionAPI()
            api.run(host="0.0.0.0", port=8000)
        
        elif choice == "8":
            print("\n[RUNNING] Integration tests...")
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ml_pipeline.py", "-v", "-s"])
            print(f"Tests completed with exit code: {result.returncode}")
        
        elif choice == "9":
            print("\nExiting...")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please select 1-9")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nDone!\n")


if __name__ == "__main__":
    main()
