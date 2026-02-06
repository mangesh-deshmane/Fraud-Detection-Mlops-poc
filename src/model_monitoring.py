import pandas as pd
import argparse
import logging
import sys
import os
import json
import numpy as np

from evidently import Report
from evidently.presets import DataDriftPreset
try:
    from evidently import ColumnMapping
except ImportError:
    from evidently.legacy.pipeline.column_mapping import ColumnMapping

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/monitoring.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def load_data(train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error("Train or Test file not found.")
        sys.exit(1)
        
    reference_data = pd.read_csv(train_path)
    current_data = pd.read_csv(test_path) # In a real scenario, this would be new production data
    
    return reference_data, current_data

def simulate_drift(df):
    """
    Artificially drift the data to demonstrate Evidently capabilities.
    """
    logging.info("Simulating data drift on 'Amount' and 'V1' features...")
    df_drifted = df.copy()
    
    # Drift V1: Shift mean
    df_drifted['V1'] = df_drifted['V1'] + 2.5
    
    # Drift Amount: Multiply by logic
    df_drifted['Amount'] = df_drifted['Amount'].apply(lambda x: x * 5 if x > 0 else x)
    
    return df_drifted

def run_monitoring(reference_data, current_data, output_path, simulate=False):
    try:
        logging.info("Starting monitoring report generation...")
        
        if simulate:
            current_data = simulate_drift(current_data)
        
        # Evidently Column Mapping
        column_mapping = ColumnMapping()
        column_mapping.target = 'Class'
        column_mapping.numerical_features = [col for col in reference_data.columns if col not in ['Class', 'Time']]
        
        # Create Report
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        result = report.run(reference_data=reference_data, current_data=current_data)
        
        # Save HTML Report
        result.save_html(output_path)
        logging.info(f"Drift report saved to {output_path}")
        
        # Save JSON Metrics
        json_path = output_path.replace('.html', '.json')
        result.save_json(json_path)
        
        # Log basic result
        with open(json_path, 'r') as f:
            metrics_data = json.load(f)
            
            # Find DriftedColumnsCount metric
            drift_share = 0.0
            found = False
            for metric in metrics_data['metrics']:
                if 'DriftedColumnsCount' in metric['metric_name']:
                    drift_share = metric['value']['share']
                    found = True
                    break
            
            if not found:
                 logging.warning("DriftedColumnsCount metric not found in report.")
            
            if drift_share > 0.5:
                logging.warning(f"HIGH DRIFT DETECTED: {drift_share:.2%} of features are drifting!")
            else:
                logging.info(f"Drift check passed. Drift share: {drift_share:.2%}")

    except Exception as e:
        logging.error(f"Monitoring failed: {e}")
        # Not exiting with 1 to allow pipeline to possibly continue or just log an alert in real world
        # But for MLOps POC we want to know
        raise e

def main():
    parser = argparse.ArgumentParser(description="Model Monitoring Script")
    parser.add_argument('--reference', type=str, default='artifacts/train.csv', help='Reference data (Train)')
    parser.add_argument('--current', type=str, default='artifacts/test.csv', help='Current data (Test/Production)')
    parser.add_argument('--output', type=str, default='artifacts/drift_report.html', help='Path to save HTML report')
    parser.add_argument('--simulate_drift', action='store_true', help='Force simulated drift for demonstration')
    args = parser.parse_args()

    setup_logging()
    
    reference_data, current_data = load_data(args.reference, args.current)
    run_monitoring(reference_data, current_data, args.output, args.simulate_drift)

if __name__ == "__main__":
    main()
