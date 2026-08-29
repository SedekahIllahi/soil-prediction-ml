from dataclasses import dataclass, field
from typing import Literal, Any
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

@dataclass
class ModelConfig:
    name: str
    estimator_class: type
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    model_family: Literal["linear", "tree"] = "tree"
    
    def build_estimator(self):
        """Instantiates the estimator with the configured hyperparameters."""
        return self.estimator_class(**self.hyperparameters)

def get_baseline_models(random_seed: int = 42) -> list[ModelConfig]:
    """
    Returns the list of baseline models to train in Phase 1.
    Reads ENABLE_SVM from environment variables to conditionally include SVM.
    """
    models = [
        ModelConfig(
            name="logistic_regression",
            estimator_class=LogisticRegression,
            hyperparameters={"max_iter": 1000, "C": 1.0, "random_state": random_seed},
            model_family="linear"
        ),
        ModelConfig(
            name="decision_tree",
            estimator_class=DecisionTreeClassifier,
            hyperparameters={"max_depth": None, "random_state": random_seed},
            model_family="tree"
        ),
        ModelConfig(
            name="random_forest",
            estimator_class=RandomForestClassifier,
            hyperparameters={"n_estimators": 100, "random_state": random_seed, "n_jobs": -1},
            model_family="tree"
        ),
        ModelConfig(
            name="xgboost",
            estimator_class=XGBClassifier,
            # Use hist method for faster training, objective for multi-class classification
            hyperparameters={"n_estimators": 100, "max_depth": 6, "random_state": random_seed, "n_jobs": -1, "objective": "multi:softprob"},
            model_family="tree"
        )
    ]
    
    # Conditionally add SVM based on environment flag (defaults to True per user approval)
    enable_svm_str = os.environ.get("ENABLE_SVM", "true").lower()
    if enable_svm_str in ("true", "1", "yes"):
        models.append(
            ModelConfig(
                name="svm",
                estimator_class=SVC,
                hyperparameters={"kernel": "rbf", "C": 1.0, "probability": True, "random_state": random_seed},
                model_family="linear"
            )
        )
        
    return models
