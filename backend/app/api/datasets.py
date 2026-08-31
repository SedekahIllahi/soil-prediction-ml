from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.dataset_service import DatasetService
from app.schemas.dataset import (
    DatasetUploadResponse,
    DatasetPreviewResponse,
    DatasetResponse,
    DatasetListResponse,
    DatasetIntegrateRequest,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and validate a raw CSV dataset file",
)
def upload_dataset(
    file: UploadFile = File(...),
    adapter_type: str = Query("urban_road_collapse", description="Adapter identifier for schema validation"),
    db: Session = Depends(get_db),
) -> DatasetUploadResponse:
    service = DatasetService(db)
    return service.upload_and_validate_dataset(file=file, adapter_type=adapter_type)

@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List logical datasets",
)
def list_datasets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
) -> DatasetListResponse:
    service = DatasetService(db)
    return service.list_datasets(page=page, page_size=page_size)

@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get dataset details and version history",
)
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
) -> DatasetResponse:
    service = DatasetService(db)
    return service.get_dataset(dataset_id)

@router.get(
    "/preview/{file_id}",
    response_model=DatasetPreviewResponse,
    summary="Preview raw CSV sample data and column summary by uploaded file ID",
)
def preview_dataset(
    file_id: str,
    limit: int = Query(10, ge=1, le=100, description="Sample row limit"),
    db: Session = Depends(get_db),
) -> DatasetPreviewResponse:
    service = DatasetService(db)
    return service.get_dataset_preview(file_id=file_id, limit=limit)

@router.post(
    "/integrate",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Integrate an uploaded validated dataset into a dataset version",
)
def integrate_dataset(
    payload: DatasetIntegrateRequest,
    db: Session = Depends(get_db),
) -> DatasetResponse:
    service = DatasetService(db)
    return service.integrate_dataset(
        name=payload.name,
        file_id=payload.file_id,
        adapter_type=payload.adapter_type,
        description=payload.description,
    )
