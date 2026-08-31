from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.prediction_service import PredictionService
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionDetailResponse,
    PredictionListResponse,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Ground/Road Collapse Risk Prediction",
    description="Submit 34 environmental, geotechnical, traffic, and infrastructure features to generate a multi-class risk classification and confidence probabilities."
)
def create_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    service = PredictionService(db)
    return service.predict(request)

@router.get(
    "",
    response_model=PredictionListResponse,
    summary="List Prediction History",
    description="Retrieve paginated history of past risk predictions with optional risk category filtering."
)
def list_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    risk_category: Optional[str] = Query(None, description="Filter by risk category (Low, Moderate, High, Critical)"),
    db: Session = Depends(get_db),
) -> PredictionListResponse:
    service = PredictionService(db)
    return service.get_prediction_history(page=page, page_size=page_size, risk_category=risk_category)

@router.get(
    "/{prediction_id}",
    response_model=PredictionDetailResponse,
    summary="Get Prediction Details",
    description="Retrieve detailed record for a specific prediction including raw input features and probabilities."
)
def get_prediction_detail(
    prediction_id: str,
    db: Session = Depends(get_db),
) -> PredictionDetailResponse:
    service = PredictionService(db)
    return service.get_prediction_by_id(prediction_id)
