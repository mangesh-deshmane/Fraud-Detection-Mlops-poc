import pandas as pd
import argparse
import logging
import sys
import os
import joblib
from sklearn.preprocessing import StandardScaler

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/feature_engineering.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # also log to stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def feature_engineering(input_path, output_path, scaler_path):
    try:
        logging.info(f"Starting feature engineering on {input_path}")
        
        # 1. Load Data
        if not os.path.exists(input_path):
             logging.error(f"Input file not found: {input_path}")
             sys.exit(1)
             
        df = pd.read_csv(input_path)
        logging.info(f"Loaded data with shape: {df.shape}")

        # 2. Scale 'Amount'
        # We need to reshape for fit_transform but assign back to Series
        scaler = StandardScaler()
        # Ensure Amount exists
        if 'Amount' not in df.columns:
            logging.error("'Amount' column missing from dataset")
            sys.exit(1)
            
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        logging.info("Scaled 'Amount' feature using StandardScaler")

        # 3. Save Scaler
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved scaler model to {scaler_path}")

        # 4. Ensure Column Order
        # Time, V1..V28, Amount, Class
        expected_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
        ]
        
        # Check if all columns exist
        missing_cols = [c for c in expected_columns if c not in df.columns]
        if missing_cols:
            logging.error(f"Missing columns: {missing_cols}")
            sys.exit(1)
            
        df_processed = df[expected_columns]
        
        # 5. Save Processed Data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")

        # Log to MLflow
        try:
            import mlflow
            mlflow.set_experiment("Fraud_Detection_Experiment")
            with mlflow.start_run(run_name="Feature_Engineering"):
                mlflow.log_param("scaler", "StandardScaler")
                mlflow.log_metric("num_features", df_processed.shape[1])
                mlflow.log_metric("num_rows", df_processed.shape[0])
                mlflow.log_artifact(scaler_path)
                mlflow.log_artifact(output_path)
                logging.info("Feature engineering metrics logged to MLflow")
        except Exception as e:
            logging.warning(f"Failed to log to MLflow: {e}")

    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Script")
    parser.add_argument('--input', type=str, default='artifacts/raw.csv', help='Path to raw data')
    parser.add_argument('--output', type=str, default='artifacts/processed.csv', help='Path to save processed data')
    parser.add_argument('--scaler', type=str, default='artifacts/scaler.pkl', help='Path to save scaler model')
    args = parser.parse_args()

    setup_logging()
    feature_engineering(args.input, args.output, args.scaler)

if __name__ == "__main__":
    main()
