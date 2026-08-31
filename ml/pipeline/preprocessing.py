from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def build_linear_preprocessor(feature_names: tuple[str, ...]) -> Pipeline:
    """
    Builds an unfitted preprocessing pipeline for linear models.
    Linear models require both imputation and scaling.
    
    Args:
        feature_names: The exact list of features to process.
        
    Returns:
        An unfitted sklearn Pipeline.
    """
    # Pipeline for numerical features (all our features are numerical)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # ColumnTransformer to apply the numeric pipeline to the specified features
    # remainder='drop' ensures only these features are passed to the model
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, list(feature_names))
        ],
        remainder='drop'
    )
    
    # Wrap in an outer pipeline (can be useful for extending later)
    return Pipeline(steps=[('preprocessor', preprocessor)])

def build_tree_preprocessor(feature_names: tuple[str, ...]) -> Pipeline:
    """
    Builds an unfitted preprocessing pipeline for tree-based models.
    Tree models require imputation but NOT scaling.
    
    Args:
        feature_names: The exact list of features to process.
        
    Returns:
        An unfitted sklearn Pipeline.
    """
    # Pipeline for numerical features (imputation only)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, list(feature_names))
        ],
        remainder='drop'
    )
    
    return Pipeline(steps=[('preprocessor', preprocessor)])

import os
import joblib

def save_preprocessor(preprocessor: Pipeline | ColumnTransformer, filepath: str) -> str:
    """
    Serializes a fitted preprocessing pipeline to disk using joblib.

    Args:
        preprocessor: The fitted sklearn Pipeline or ColumnTransformer.
        filepath: Destination file path.

    Returns:
        The absolute path to the saved preprocessor artifact.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    return os.path.abspath(filepath)

def load_preprocessor(filepath: str) -> Pipeline | ColumnTransformer:
    """
    Deserializes a preprocessing pipeline from disk.

    Args:
        filepath: Path to the serialized joblib file.

    Returns:
        The loaded sklearn Pipeline or ColumnTransformer.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor artifact not found: {filepath}")
    return joblib.load(filepath)

