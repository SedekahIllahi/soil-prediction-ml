from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field

class TrainingRunCreate(BaseModel):
    dataset_version_id: str
    algorithms: Optional[list[str]] = Field(
        default=None, 
        description="Optional subset of algorithm names to train. If None, trains all registered baseline models."
    )

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

class TrainingRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_version_id: str
    status: str
    config: Optional[dict[str, Any]] = None
    comparison_results: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    model_versions: list[ModelVersionResponse] = []

class TrainingRunListResponse(BaseModel):
    items: list[TrainingRunResponse]
    total: int
    page: int
    page_size: int

class ModelPromoteRequest(BaseModel):
    model_version_id: str
