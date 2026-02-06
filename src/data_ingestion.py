import pandas as pd
import os
import logging
import argparse

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

# Ensure directories exist before configuring logging
setup_directories()

# Configure logging
logging.basicConfig(
    filename='logs/data_ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        logging.info(f"Attempting to load data from {file_path}")
        
        # Check if file is a Git LFS pointer
        if os.path.getsize(file_path) < 200:
            with open(file_path, 'r') as f:
                content = f.read()
                if "version https://git-lfs.github.com/spec/v1" in content:
                    error_msg = (
                        f"Input file '{file_path}' appears to be a Git LFS pointer (size < 200 bytes). "
                        "Please run 'git lfs pull' on your local machine and ensure the file is mounted correctly."
                    )
                    logging.error(error_msg)
                    raise ValueError(error_msg)

        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def save_data(df, output_path):
    """
    Save the DataFrame to a CSV file.
    """
    try:
        logging.info(f"Saving data to {output_path}")
        df.to_csv(output_path, index=False)
        logging.info("Data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Data Ingestion Script")
    parser.add_argument('--input', type=str, default='dataset/creditcard.csv', help='Path to source dataset')
    parser.add_argument('--output', type=str, default='artifacts/raw.csv', help='Path to save ingested artifact')
    args = parser.parse_args()

    setup_directories()

    try:
        df = load_data(args.input)
        
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]} ...") # Print first 5 cols to avoid clutter

        save_data(df, args.output)
        print(f"\nIngestion completed. Data saved to {args.output}")

        # Log to MLflow
        try:
            import mlflow
            mlflow.set_experiment("Fraud_Detection_Experiment")
            with mlflow.start_run(run_name="Data_Ingestion"):
                mlflow.log_param("input_file", args.input)
                mlflow.log_metric("num_rows", df.shape[0])
                mlflow.log_metric("num_cols", df.shape[1])
                mlflow.log_artifact(args.output)
                print("Ingestion metrics logged to MLflow")
        except Exception as e:
            print(f"Failed to log to MLflow: {e}")
        
    except Exception as e:
        import traceback
        print(f"Ingestion failed. Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
