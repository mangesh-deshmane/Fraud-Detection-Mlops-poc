import great_expectations as gx
import pandas as pd
import argparse
import logging
import sys
import os

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/data_validation.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Also log to stdout for immediate feedback
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def validate_data(file_path):
    try:
        logging.info(f"Starting validation for {file_path}")
        
        context = gx.get_context()
        
        datasource_name = "pandas_datasource"
        asset_name = "credit_card_data"
        
        # Use data_sources.add_pandas (if not already existing, handling error might be needed but for script it's fine)
        # For simplicity in this script, we re-add or get
        try:
             datasource = context.data_sources.get(datasource_name)
        except KeyError:
             datasource = context.data_sources.add_pandas(datasource_name)
        
        try:
            asset = datasource.get_asset(asset_name)
        except LookupError:
             asset = datasource.add_csv_asset(name=asset_name, filepath_or_buffer=file_path)
        
        # Build batch request
        # For CSV asset in fluent API, build_batch_request is available
        batch_request = asset.build_batch_request()
        
        # Define Expectation Suite
        suite_name = "credit_card_suite"
        
        validator = context.get_validator(
            batch_request=batch_request,
            create_expectation_suite_with_name=suite_name
        )
        
        # --- Define Expectations ---
        
        # 1. No nulls
        for col in validator.columns():
            validator.expect_column_values_to_not_be_null(column=col)

        # 2. Class in {0, 1}
        validator.expect_column_values_to_be_in_set(column="Class", value_set=[0, 1])

        # 3. Amount >= 0
        validator.expect_column_values_to_be_between(column="Amount", min_value=0)

        # 4. Schema consistency
        expected_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
        ]
        validator.expect_table_columns_to_match_ordered_list(column_list=expected_columns)

        # Run validation directly using the validator
        validation_result = validator.validate()
        
        if not validation_result["success"]:
            logging.error("Validation FAILED!")
            # Log specific failures (simplified)
            for res in validation_result["results"]:
                if not res["success"]:
                    logging.error(f"Failed expectation: {res['expectation_config']['expectation_type']} on {res['expectation_config']['kwargs']}")
            sys.exit(1)
        
        logging.info("Validation PASSED!")

        # Log to MLflow
        try:
            import mlflow
            mlflow.set_experiment("Fraud_Detection_Experiment")
            with mlflow.start_run(run_name="Data_Validation"):
                mlflow.log_param("validation_status", "Passed")
                logging.info("Validation metrics logged to MLflow")
        except Exception as e:
            logging.warning(f"Failed to log to MLflow: {e}")
        
    except Exception as e:
        logging.error(f"Validation process failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Data Validation Script")
    parser.add_argument('--input', type=str, default='artifacts/raw.csv', help='Path to dataset artifact')
    args = parser.parse_args()

    setup_logging()
    
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    validate_data(args.input)

if __name__ == "__main__":
    main()
