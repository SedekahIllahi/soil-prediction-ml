import pytest
import numpy as np
import pandas as pd
from ml.models.registry import ModelConfig
from ml.pipeline.training import train_model
from ml.pipeline.preprocessing import build_tree_preprocessor
from ml.pipeline.target_encoding import TargetEncoderWrapper
from ml.pipeline.evaluation import evaluate_model, EvaluationResult
from ml.schema import MODEL_FEATURES, TARGET_CLASSES
from sklearn.tree import DecisionTreeClassifier

def test_evaluate_model_metrics_and_matrix(sample_raw_df, adapter):
    X, y, _ = adapter.transform(sample_raw_df)
    
    preproc = build_tree_preprocessor(MODEL_FEATURES)
    X_trans = preproc.fit_transform(X)
    
    target_enc = TargetEncoderWrapper(is_linear=False)
    y_trans = target_enc.fit_transform(y)
    
    config = ModelConfig(
        name="decision_tree",
        estimator_class=DecisionTreeClassifier,
        hyperparameters={"max_depth": 3, "random_state": 42},
        model_family="tree"
    )
    trained = train_model(config, X_trans, y_trans)
    
    eval_res = evaluate_model(
        trained_model=trained,
        X_val_raw=X,
        y_val_raw=y,
        preprocessor=preproc,
        target_encoder=target_enc
    )
    
    assert isinstance(eval_res, EvaluationResult)
    assert 0.0 <= eval_res.accuracy <= 1.0
    assert 0.0 <= eval_res.weighted_f1 <= 1.0
    assert 0.0 <= eval_res.macro_f1 <= 1.0
    assert len(eval_res.class_labels) == 4
    assert set(eval_res.class_labels) == set(TARGET_CLASSES)
    assert len(eval_res.confusion_matrix) == 4
    assert len(eval_res.confusion_matrix[0]) == 4
    for c in TARGET_CLASSES:
        assert c in eval_res.per_class
        assert "precision" in eval_res.per_class[c]
        assert "recall" in eval_res.per_class[c]
        assert "f1" in eval_res.per_class[c]
