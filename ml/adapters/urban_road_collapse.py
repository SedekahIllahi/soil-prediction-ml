import pandas as pd

from .base import DatasetAdapter
from ml.schema import TARGET_COLUMN, MODEL_FEATURES, METADATA_COLUMNS

class UrbanRoadCollapseAdapter(DatasetAdapter):
    """
    Adapter for Dataset 2 (Urban Road Collapse Risk Assessment).
    The dataset is already very close to the canonical schema. This adapter
    simply drops the removed/leakage columns and splits features, target, and metadata.
    """
    
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Transforms the dataframe by separating X (features), y (target), and metadata.
        """
        # Ensure all required features are present (validation should have caught this if not, 
        # but we must extract safely)
        
        # 1. Extract Target
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
        y = df[TARGET_COLUMN].copy()
        
        # 2. Extract Features
        # We only take the exact columns listed in MODEL_FEATURES.
        # Missing features will raise a KeyError here, which is expected behavior
        # if validation was bypassed.
        X = df[list(MODEL_FEATURES)].copy()
        
        # 3. Extract Metadata
        # Metadata is optional, we take what we can find.
        available_metadata = [col for col in METADATA_COLUMNS if col in df.columns]
        if available_metadata:
            metadata = df[available_metadata].copy()
        else:
            metadata = pd.DataFrame(index=df.index)
            
        return X, y, metadata
