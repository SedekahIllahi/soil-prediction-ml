from typing import Optional
from fastapi import APIRouter, Depends, Query, BackgroundTasks, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.training import (
    TrainingRunCreate,
    TrainingRunResponse,
    TrainingRunListResponse,
    ModelVersionResponse,
    ModelPromoteRequest,
)
from app.services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["Training & Models"])

def run_training_task(run_id: str, db_factory):
    """Background task handler for executing training asynchronously."""
    db = db_factory()
    try:
        service = TrainingService(db)
        service.execute_training_run(run_id)
    finally:
        db.close()

@router.post("", response_model=TrainingRunResponse, status_code=status.HTTP_201_CREATED)
def start_training_run(
    payload: TrainingRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Creates a new training run record and triggers background training execution.
    Returns immediately with status='pending'.
    """
    service = TrainingService(db)
    run_resp = service.create_training_run(
        dataset_version_id=payload.dataset_version_id,
        algorithms=payload.algorithms
    )
    
    # Launch background execution task
    # We pass session factory via get_db iterator / SessionLocal
    from app.core.database import SessionLocal
    background_tasks.add_task(run_training_task, run_resp.id, SessionLocal)
    
    return run_resp

@router.get("", response_model=TrainingRunListResponse)
def list_training_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Lists training runs with pagination."""
    service = TrainingService(db)
    return service.list_runs(page=page, page_size=page_size)

@router.get("/{run_id}", response_model=TrainingRunResponse)
def get_training_run(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Retrieves a specific training run details and comparison results."""
    service = TrainingService(db)
    return service.get_run(run_id)
