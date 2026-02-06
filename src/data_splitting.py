import pandas as pd
import argparse
import logging
import sys
import os
from sklearn.model_selection import train_test_split

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/data_splitting.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def split_data(input_path, train_path, test_path, test_size=0.2, random_state=42):
    try:
        logging.info(f"Starting data splitting on {input_path}")
        
        if not os.path.exists(input_path):
             logging.error(f"Input file not found: {input_path}")
             sys.exit(1)
             
        df = pd.read_csv(input_path)
        logging.info(f"Loaded data with shape: {df.shape}")

        if 'Class' not in df.columns:
            logging.error("'Class' column missing, cannot stratify")
            sys.exit(1)

        # Stratified Split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['Class']
        )
        
        logging.info(f"Split data: Train shape {train_df.shape}, Test shape {test_df.shape}")
        
        # Verify stratification
        train_fraud_rate = train_df['Class'].mean()
        test_fraud_rate = test_df['Class'].mean()
        logging.info(f"Train Fraud Rate: {train_fraud_rate:.6f}")
        logging.info(f"Test Fraud Rate: {test_fraud_rate:.6f}")

        # Save splits
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logging.info(f"Saved train set to {train_path}")
        logging.info(f"Saved test set to {test_path}")

    except Exception as e:
        logging.error(f"Data splitting failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Data Splitting Script")
    parser.add_argument('--input', type=str, default='artifacts/processed.csv', help='Path to processed data')
    parser.add_argument('--train_output', type=str, default='artifacts/train.csv', help='Path to save train data')
    parser.add_argument('--test_output', type=str, default='artifacts/test.csv', help='Path to save test data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset to include in the test split')
    args = parser.parse_args()

    setup_logging()
    split_data(args.input, args.train_output, args.test_output, args.test_size)

if __name__ == "__main__":
    main()
