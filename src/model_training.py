import pandas as pd
import argparse
import logging
import sys
import os
import joblib
import json
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/model_training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error("Train or Test file not found.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test, models_dir, config):
    os.makedirs(models_dir, exist_ok=True)
    results = {}
    
    # Load params from config
    lr_params = config['model_training']['logistic_regression']
    xgb_params = config['model_training']['xgboost']
    lgbm_params = config['model_training']['lightgbm']
    
    models = {
        "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=42, **lr_params),
        "XGBoost": XGBClassifier(scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1], 
                                 random_state=42, **xgb_params),
        "LightGBM": LGBMClassifier(scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1],
                                   random_state=42, **lgbm_params)
    }
    
    # Set MLflow Tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logging.info(f"Using MLflow Tracking URI: {tracking_uri}")
    else:
        logging.info("Using local MLflow tracking.")

    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    logging.info(f"Training {len(models)} models...")
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        
        with mlflow.start_run(run_name=f"{name}_Run"):
            try:
                # Log Params
                mlflow.log_param("model_name", name)
                if name == "LogisticRegression":
                    mlflow.log_params(lr_params)
                elif name == "XGBoost":
                    mlflow.log_params(xgb_params)
                elif name == "LightGBM":
                    mlflow.log_params(lgbm_params)

                # Train
                model.fit(X_train, y_train)
                
                # Predict Probabilities (for AUC)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Predict Labels (for Classification Report)
                y_pred = model.predict(X_test)
                
                # Evaluate
                auc = roc_auc_score(y_test, y_pred_proba)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                results[name] = {
                    "AUC": auc,
                    "Precision_Fraud": report['1']['precision'],
                    "Recall_Fraud": report['1']['recall'],
                    "F1_Fraud": report['1']['f1-score']
                }
                
                logging.info(f"{name} AUC: {auc:.4f}")
                
                # Log Metrics
                mlflow.log_metric("AUC", auc)
                mlflow.log_metric("Precision_Fraud", results[name]["Precision_Fraud"])
                mlflow.log_metric("Recall_Fraud", results[name]["Recall_Fraud"])
                mlflow.log_metric("F1_Fraud", results[name]["F1_Fraud"])
                
                # Save Model (Local)
                joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))
                
                # Log Model (MLflow)
                if name == "XGBoost":
                    mlflow.xgboost.log_model(model, "model")
                elif name == "LightGBM":
                    mlflow.lightgbm.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                
            except Exception as e:
                logging.error(f"Failed to train {name}: {e}")
            
    # Save Results
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Training completed. Metrics saved to {models_dir}/metrics.json")

def main():
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument('--train_path', type=str, default='artifacts/train.csv', help='Path to train data')
    parser.add_argument('--test_path', type=str, default='artifacts/test.csv', help='Path to test data')
    parser.add_argument('--models_dir', type=str, default='artifacts/models', help='Directory to save models')
    parser.add_argument('--config', type=str, default='params.yaml', help='Path to config file')
    args = parser.parse_args()

    setup_logging()
    
    config = load_config(args.config)
    
    logging.info("Loading data...")
    X_train, y_train, X_test, y_test = load_data(args.train_path, args.test_path)
    
    train_and_evaluate(X_train, y_train, X_test, y_test, args.models_dir, config)

if __name__ == "__main__":
    main()
