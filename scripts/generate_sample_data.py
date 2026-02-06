import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(output_path: str, n_samples: int = 1100):
    """
    Generate synthetic credit card fraud data for CI/CD testing.
    Schema matches the Kaggle Credit Card Fraud Detection dataset.
    """
    print(f"Generating {n_samples} samples of synthetic fraud data...")
    
    np.random.seed(42)
    
    data = {
        'Time': np.sort(np.random.uniform(0, 172792, n_samples)),
        **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
        'Amount': np.random.exponential(scale=88, size=n_samples),
        'Class': np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
    }
    
    df = pd.DataFrame(data)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Sample data saved to {output_path}")

if __name__ == "__main__":
    generate_sample_data("dataset/creditcard.csv")
