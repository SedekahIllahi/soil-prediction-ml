import time
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ml.models.registry import ModelConfig

@dataclass
class TrainedModel:
    """Wrapper for a model that has completed training."""
    config: ModelConfig
    model: any  # The fitted scikit-learn or xgboost estimator
    training_time_seconds: float

def train_model(config: ModelConfig, X_train: np.ndarray, y_train: np.ndarray) -> TrainedModel:
    """
    Trains a single model based on its configuration.
    
    Args:
        config: The ModelConfig specifying the algorithm and hyperparameters.
        X_train: The preprocessed feature matrix.
        y_train: The encoded target array.
        
    Returns:
        TrainedModel containing the fitted estimator.
    """
    estimator = config.build_estimator()
    
    start_time = time.time()
    estimator.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    return TrainedModel(
        config=config,
        model=estimator,
        training_time_seconds=training_time
    )

def run_baseline_training(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    linear_preprocessor,
    tree_preprocessor,
    target_encoder_linear,
    target_encoder_tree,
    model_configs: list[ModelConfig]
) -> tuple[list[TrainedModel], any, any, any, any]:
    """
    Trains a list of baseline models.
    Handles the dispatching of appropriate preprocessors and target encoders
    based on the model family (linear vs tree).
    
    Args:
        X_train: Raw training features.
        y_train: Raw training targets (strings).
        linear_preprocessor: Unfitted Pipeline for linear models.
        tree_preprocessor: Unfitted Pipeline for tree models.
        target_encoder_linear: TargetEncoderWrapper for linear models.
        target_encoder_tree: TargetEncoderWrapper for tree models.
        model_configs: List of configurations to train.
        
    Returns:
        Tuple containing:
        - List of TrainedModels
        - Fitted linear_preprocessor
        - Fitted tree_preprocessor
        - Fitted target_encoder_linear
        - Fitted target_encoder_tree
    """
    # 1. Fit Preprocessors and Encoders on training data
    # We fit both regardless of models, as we'll need them for evaluation
    X_train_linear = linear_preprocessor.fit_transform(X_train)
    X_train_tree = tree_preprocessor.fit_transform(X_train)
    
    y_train_linear = target_encoder_linear.fit_transform(y_train)
    y_train_tree = target_encoder_tree.fit_transform(y_train)
    
    trained_models = []
    
    # 2. Train each model with its appropriate preprocessed data
    for config in model_configs:
        if config.model_family == "linear":
            X_data = X_train_linear
            y_data = y_train_linear
        elif config.model_family == "tree":
            X_data = X_train_tree
            y_data = y_train_tree
        else:
            raise ValueError(f"Unknown model family: {config.model_family}")
            
        trained_model = train_model(config, X_data, y_data)
        trained_models.append(trained_model)
        
    return (
        trained_models, 
        linear_preprocessor, 
        tree_preprocessor, 
        target_encoder_linear, 
        target_encoder_tree
    )
