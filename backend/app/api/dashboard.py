from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.prediction_service import PredictionService
from app.schemas.prediction import DashboardSummaryResponse, DashboardRecentResponse

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
    summary="Get Dashboard Summary Statistics",
    description="Retrieve high-level metrics including total prediction volume, risk distribution counts across the 4 classes, and active model status."
)
def get_dashboard_summary(
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:
    service = PredictionService(db)
    return service.get_dashboard_summary()

@router.get(
    "/recent",
    response_model=DashboardRecentResponse,
    summary="Get Recent Predictions",
    description="Retrieve a list of the most recent predictions for dashboard display."
)
def get_dashboard_recent(
    limit: int = Query(10, ge=1, le=50, description="Number of recent records to return"),
    db: Session = Depends(get_db),
) -> DashboardRecentResponse:
    service = PredictionService(db)
    return service.get_dashboard_recent(limit=limit)
