from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.core.database import get_db

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Health check endpoint returning system status and DB connectivity.
    """
    db_status = "unhealthy"
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "ok",
        "environment": settings.ENVIRONMENT,
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_status,
    }
