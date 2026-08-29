import pandas as pd
import numpy as np
from pathlib import Path

from ml.schema import MODEL_FEATURES, METADATA_COLUMNS, TARGET_COLUMN, TARGET_CLASSES, FEATURE_RANGES

def generate():
    np.random.seed(42)
    n_rows = 100
    
    data = {}
    
    # Metadata
    data["segment_id"] = [f"URC_{str(i).zfill(5)}" for i in range(1, n_rows + 1)]
    data["latitude"] = np.random.uniform(12.8, 13.25, n_rows)
    data["longitude"] = np.random.uniform(80.05, 80.35, n_rows)
    
    # Target
    data[TARGET_COLUMN] = np.random.choice(TARGET_CLASSES, n_rows)
    
    # Features
    for feature in MODEL_FEATURES:
        if feature in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feature]
            data[feature] = np.random.uniform(min_val, max_val, n_rows)
        else:
            data[feature] = np.random.uniform(0, 1, n_rows)
            
    df = pd.DataFrame(data)
    
    out_dir = Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sample_dataset.csv", index=False)
    print(f"Generated tests/fixtures/sample_dataset.csv ({n_rows} rows)")

if __name__ == "__main__":
    generate()
