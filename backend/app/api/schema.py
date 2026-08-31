from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.prediction_service import PredictionService
from app.schemas.prediction import FeatureSchemaResponse

router = APIRouter(prefix="/schema", tags=["Schema"])

@router.get(
    "/features",
    response_model=FeatureSchemaResponse,
    summary="Get Prediction Feature Schema",
    description="Retrieve the 34 canonical model input features, ranges, categories, and caution flags for dynamic UI form rendering."
)
def get_feature_schema(
    db: Session = Depends(get_db),
) -> FeatureSchemaResponse:
    service = PredictionService(db)
    return service.get_feature_schema()
