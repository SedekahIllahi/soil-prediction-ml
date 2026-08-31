import pytest
import numpy as np
import pandas as pd
from ml.models.registry import ModelConfig, get_baseline_models
from ml.pipeline.training import train_model, run_baseline_training
from ml.pipeline.preprocessing import build_linear_preprocessor, build_tree_preprocessor
from ml.pipeline.target_encoding import TargetEncoderWrapper
from ml.schema import MODEL_FEATURES
from sklearn.tree import DecisionTreeClassifier

def test_train_single_model():
    config = ModelConfig(
        name="decision_tree",
        estimator_class=DecisionTreeClassifier,
        hyperparameters={"max_depth": 3, "random_state": 42},
        model_family="tree"
    )
    X = np.random.randn(100, 34)
    y = np.random.randint(0, 4, size=100)

    trained = train_model(config, X, y)
    assert trained.config.name == "decision_tree"
    assert trained.training_time_seconds >= 0.0
    assert hasattr(trained.model, "predict")

def test_run_baseline_training_pipeline(sample_raw_df, adapter):
    X, y, _ = adapter.transform(sample_raw_df)
    
    linear_preproc = build_linear_preprocessor(MODEL_FEATURES)
    tree_preproc = build_tree_preprocessor(MODEL_FEATURES)
    target_enc_linear = TargetEncoderWrapper(is_linear=True)
    target_enc_tree = TargetEncoderWrapper(is_linear=False)

    # Test with subset of models
    configs = [
        ModelConfig(name="dt_test", estimator_class=DecisionTreeClassifier, hyperparameters={"max_depth": 2, "random_state": 42}, model_family="tree")
    ]

    trained_models, fitted_linear, fitted_tree, enc_lin, enc_tree = run_baseline_training(
        X_train=X,
        y_train=y,
        linear_preprocessor=linear_preproc,
        tree_preprocessor=tree_preproc,
        target_encoder_linear=target_enc_linear,
        target_encoder_tree=target_enc_tree,
        model_configs=configs
    )

    assert len(trained_models) == 1
    assert trained_models[0].config.name == "dt_test"
    # Preprocessors and encoders should be fitted
    assert hasattr(fitted_linear, "transform")
    assert hasattr(fitted_tree, "transform")
