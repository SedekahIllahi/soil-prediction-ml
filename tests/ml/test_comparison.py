import pytest
from ml.pipeline.evaluation import EvaluationResult
from ml.pipeline.comparison import compare_models, ComparisonReport

def test_compare_models_ranking():
    res1 = EvaluationResult(
        model_name="logistic_regression",
        accuracy=0.80,
        macro_precision=0.80,
        macro_recall=0.80,
        macro_f1=0.80,
        weighted_f1=0.80,
        per_class={"High": {"precision": 0.8, "recall": 0.75, "f1": 0.77}},
        confusion_matrix=[[10, 0], [0, 10]],
        class_labels=["Low", "High"]
    )

    res2 = EvaluationResult(
        model_name="xgboost",
        accuracy=0.95,
        macro_precision=0.95,
        macro_recall=0.95,
        macro_f1=0.95,
        weighted_f1=0.95,
        per_class={"High": {"precision": 0.95, "recall": 0.94, "f1": 0.94}},
        confusion_matrix=[[10, 0], [0, 10]],
        class_labels=["Low", "High"]
    )

    times = {"logistic_regression": 0.5, "xgboost": 1.2}
    report = compare_models([res1, res2], times)

    assert isinstance(report, ComparisonReport)
    assert report.best_model_name == "xgboost"
    assert len(report.entries) == 2
    assert report.entries[0].model_name == "xgboost"
    assert report.entries[0].rank == 1
    assert report.entries[0].high_class_recall == 0.94
    assert report.entries[0].high_risk_recall == 0.94
    assert report.entries[1].model_name == "logistic_regression"
    assert report.entries[1].rank == 2
    assert report.entries[1].high_class_recall == 0.75

def test_compare_models_tiebreaker():
    # Equal weighted_f1, but res2 has higher High risk recall
    res1 = EvaluationResult(
        model_name="model_a",
        accuracy=0.90,
        macro_precision=0.90,
        macro_recall=0.90,
        macro_f1=0.90,
        weighted_f1=0.90,
        per_class={"High": {"precision": 0.9, "recall": 0.80, "f1": 0.85}},
        confusion_matrix=[[10, 0], [0, 10]],
        class_labels=["Low", "High"]
    )

    res2 = EvaluationResult(
        model_name="model_b",
        accuracy=0.90,
        macro_precision=0.90,
        macro_recall=0.90,
        macro_f1=0.90,
        weighted_f1=0.90,
        per_class={"High": {"precision": 0.9, "recall": 0.90, "f1": 0.90}},
        confusion_matrix=[[10, 0], [0, 10]],
        class_labels=["Low", "High"]
    )

    times = {"model_a": 0.5, "model_b": 0.5}
    report = compare_models([res1, res2], times)

    assert report.best_model_name == "model_b"
    assert report.entries[0].model_name == "model_b"
