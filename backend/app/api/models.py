from typing import Optional
from fastapi import APIRouter, Depends, Query, Path, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.model import (
    ModelVersionResponse,
    ModelVersionListResponse,
    ModelComparisonResponse,
)
from app.schemas.training import ModelPromoteRequest
from app.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["Models"])

@router.get("", response_model=ModelVersionListResponse)
def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status (e.g. active, candidate, evaluated, retired)"),
    algorithm: Optional[str] = Query(None, description="Filter by algorithm (e.g. logistic_regression, xgboost)"),
    db: Session = Depends(get_db)
):
    """Lists all trained model versions with pagination and filtering."""
    service = ModelService(db)
    return service.list_models(
        page=page,
        page_size=page_size,
        status_filter=status,
        algorithm_filter=algorithm
    )

@router.get("/active", response_model=Optional[ModelVersionResponse])
def get_active_model(
    db: Session = Depends(get_db)
):
    """Retrieves the currently active model version used for inference."""
    service = ModelService(db)
    return service.get_active_model()

@router.get("/compare", response_model=ModelComparisonResponse)
def compare_models(
    model_ids: Optional[list[str]] = Query(None, description="Optional list of specific model IDs to compare"),
    db: Session = Depends(get_db)
):
    """
    Compares evaluation metrics across model versions side-by-side,
    ranking by Weighted F1 and High-class Recall.
    """
    service = ModelService(db)
    return service.compare_models(model_ids=model_ids)

@router.get("/{id}", response_model=ModelVersionResponse)
def get_model_detail(
    id: str = Path(..., description="The unique ID of the model version"),
    db: Session = Depends(get_db)
):
    """Retrieves full details and metrics for a specific model version."""
    service = ModelService(db)
    return service.get_model(id)

@router.post("/{id}/promote", response_model=ModelVersionResponse)
def promote_model_by_path(
    id: str = Path(..., description="The model version ID to promote to active"),
    db: Session = Depends(get_db)
):
    """Promotes a model version to active status, demoting any previously active model to retired."""
    service = ModelService(db)
    return service.promote_model(id)

@router.post("/{id}/rollback", response_model=ModelVersionResponse)
def rollback_model_by_path(
    id: str = Path(..., description="The retired model version ID to rollback/re-promote to active"),
    db: Session = Depends(get_db)
):
    """Re-promotes a retired model version to active status."""
    service = ModelService(db)
    return service.rollback_model(id)

@router.post("/promote", response_model=ModelVersionResponse)
def promote_model_legacy_body(
    payload: ModelPromoteRequest,
    db: Session = Depends(get_db)
):
    """Legacy endpoint supporting promotion via JSON request body."""
    service = ModelService(db)
    return service.promote_model(payload.model_version_id)
