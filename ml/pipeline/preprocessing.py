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
