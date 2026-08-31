from typing import Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class DatasetVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    version: int
    file_path: str
    row_count: int
    column_info: dict[str, Any]
    created_at: datetime

class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str] = None
    adapter_type: str
    created_at: datetime
    latest_version: Optional[DatasetVersionResponse] = None
    versions: List[DatasetVersionResponse] = []

class DatasetListResponse(BaseModel):
    items: List[DatasetResponse]
    total: int
    page: int
    page_size: int

class DatasetUploadResponse(BaseModel):
    file_id: str
    filename: str
    stored_path: str
    size_bytes: int
    row_count: int
    is_valid: bool
    validation_errors: List[str]
    column_names: List[str]

class DatasetPreviewResponse(BaseModel):
    file_id: str
    filename: str
    total_rows: int
    columns: List[str]
    sample_data: List[dict[str, Any]]
    column_summary: dict[str, Any]

class DatasetIntegrateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Logical dataset name")
    description: Optional[str] = Field(None, max_length=1000, description="Optional dataset description")
    file_id: str = Field(..., description="Uploaded file reference ID from POST /api/datasets/upload")
    adapter_type: str = Field("urban_road_collapse", description="Dataset adapter identifier")
