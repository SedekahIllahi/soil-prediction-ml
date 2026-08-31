from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ml.schema import MODEL_FEATURES, FEATURE_RANGES, CAUTION_FEATURES, TARGET_CLASSES

class FeatureField(BaseModel):
    name: str
    type: str = "float"
    min_val: float
    max_val: float
    is_caution: bool = False
    category: str
    description: Optional[str] = None

class FeatureSchemaResponse(BaseModel):
    features: list[FeatureField]
    target_classes: list[str]
    total_features: int

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Dictionary mapping all 34 canonical feature names to numeric values."
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Geographic latitude metadata for map display."
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Geographic longitude metadata for map display."
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict[str, float]) -> dict[str, float]:
        if not isinstance(v, dict):
            raise ValueError("Features must be a JSON dictionary of key-value pairs.")
        
        missing = [f for f in MODEL_FEATURES if f not in v]
        if missing:
            raise ValueError(f"Missing required model features ({len(missing)}): {missing}")
        
        invalid_ranges = []
        for feat, (min_val, max_val) in FEATURE_RANGES.items():
            if feat in v:
                val = v[feat]
                if val is None or not isinstance(val, (int, float)):
                    invalid_ranges.append(f"{feat}: value '{val}' is not a valid number")
                elif val < min_val or val > max_val:
                    invalid_ranges.append(f"{feat}: value {val} is outside expected range [{min_val}, {max_val}]")
        
        if invalid_ranges:
            raise ValueError(f"Feature range validation errors ({len(invalid_ranges)}): {invalid_ranges}")
            
        return v

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model_version_id: Optional[str] = None
    model_version_number: Optional[int] = None
    algorithm: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime

class PredictionDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    features: dict[str, Any]
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model_version_id: Optional[str] = None
    model_version_number: Optional[int] = None
    algorithm: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: datetime

class PredictionListResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
    page: int
    page_size: int

class DashboardSummaryResponse(BaseModel):
    total_predictions: int
    risk_distribution: dict[str, int]
    active_model: Optional[dict[str, Any]] = None

class DashboardRecentResponse(BaseModel):
    items: list[PredictionResponse]
