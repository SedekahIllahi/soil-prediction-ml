from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.models.prediction import Prediction
from ml.schema import TARGET_CLASSES

class PredictionRepository:
    """
    Encapsulates all database operations for Prediction records:
    - Creation and persistence of new predictions
    - Retrieval of individual prediction details
    - Paginated listing with optional risk category filtering
    - Summary aggregation for dashboard analytics
    - Recent predictions retrieval
    """

    def __init__(self, db: Session):
        self.db = db

    def create_prediction(
        self,
        model_version_id: Optional[str],
        features: dict,
        predicted_class: str,
        confidence: float,
        probabilities: dict,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Prediction:
        prediction = Prediction(
            model_version_id=model_version_id,
            features=features,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            latitude=latitude,
            longitude=longitude,
        )
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction

    def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        return self.db.query(Prediction).filter(Prediction.id == prediction_id).first()

    def list_predictions(
        self,
        skip: int = 0,
        limit: int = 20,
        risk_category: Optional[str] = None,
    ) -> tuple[list[Prediction], int]:
        query = self.db.query(Prediction)
        if risk_category:
            query = query.filter(Prediction.predicted_class == risk_category)
        
        total = query.count()
        items = query.order_by(desc(Prediction.created_at)).offset(skip).limit(limit).all()
        return items, total

    def get_dashboard_summary(self) -> dict:
        total = self.db.query(Prediction).count()
        
        # Aggregate counts by predicted_class
        distribution = {cls_name: 0 for cls_name in TARGET_CLASSES}
        rows = (
            self.db.query(Prediction.predicted_class, func.count(Prediction.id))
            .group_by(Prediction.predicted_class)
            .all()
        )
        for class_name, count in rows:
            if class_name in distribution:
                distribution[class_name] = count
            else:
                distribution[class_name] = count

        return {
            "total_predictions": total,
            "risk_distribution": distribution,
        }

    def get_recent_predictions(self, limit: int = 10) -> list[Prediction]:
        return (
            self.db.query(Prediction)
            .order_by(desc(Prediction.created_at))
            .limit(limit)
            .all()
        )
