from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field

class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    training_run_id: str
    dataset_version_id: str
    version: int
    algorithm: str
    status: str
    metrics: Optional[dict[str, Any]] = None
    hyperparameters: Optional[dict[str, Any]] = None
    artifact_path: Optional[str] = None
    preprocessor_path: Optional[str] = None
    training_time_seconds: Optional[float] = None
    created_at: datetime

class ModelVersionListResponse(BaseModel):
    items: list[ModelVersionResponse]
    total: int
    page: int
    page_size: int

class ModelComparisonItem(BaseModel):
    rank: int
    model_id: str
    version: int
    algorithm: str
    status: str
    weighted_f1: float
    macro_f1: float
    accuracy: float
    high_class_recall: float
    high_risk_recall: float  # Compatibility alias
    training_time_seconds: float
    per_class_metrics: dict[str, dict[str, float]]
    created_at: datetime

class ModelComparisonResponse(BaseModel):
    best_model_id: Optional[str] = None
    best_model_algorithm: Optional[str] = None
    primary_metric: str = "weighted_f1"
    secondary_metric: str = "high_class_recall"
    compared_models: list[ModelComparisonItem]
    compared_at: datetime
