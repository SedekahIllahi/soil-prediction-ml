from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.training import ModelVersion

class ModelRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_model(self, model_id: str) -> Optional[ModelVersion]:
        """Retrieves a single model version by ID."""
        return self.db.query(ModelVersion).filter(ModelVersion.id == model_id).first()

    def list_models(
        self,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
        algorithm: Optional[str] = None
    ) -> tuple[list[ModelVersion], int]:
        """
        Lists model versions with optional status/algorithm filtering and pagination.
        Returns (items, total_count).
        """
        query = self.db.query(ModelVersion)
        if status:
            query = query.filter(ModelVersion.status == status)
        if algorithm:
            query = query.filter(ModelVersion.algorithm == algorithm)

        total = query.count()
        items = query.order_by(ModelVersion.version.desc()).offset(skip).limit(limit).all()
        return items, total

    def get_active_model(self) -> Optional[ModelVersion]:
        """Retrieves the single currently active model version for inference."""
        return self.db.query(ModelVersion).filter(ModelVersion.status == "active").first()

    def set_active_model(self, model_id: str) -> Optional[ModelVersion]:
        """
        Promotes the specified model to 'active' status.
        Atomically transitions any currently 'active' model to 'retired'.
        Enforces that at most one model is 'active' at any time.
        """
        # Demote all current active models to retired
        current_actives = self.db.query(ModelVersion).filter(ModelVersion.status == "active").all()
        for mv in current_actives:
            if mv.id != model_id:
                mv.status = "retired"

        target = self.get_model(model_id)
        if target:
            target.status = "active"
            self.db.commit()
            self.db.refresh(target)
            return target
        return None

    def get_models_by_ids(self, model_ids: list[str]) -> list[ModelVersion]:
        """Retrieves multiple model versions matching the provided list of IDs."""
        return self.db.query(ModelVersion).filter(ModelVersion.id.in_(model_ids)).all()

    def get_all_evaluated_models(self) -> list[ModelVersion]:
        """Retrieves all model versions that have completed evaluation."""
        return (
            self.db.query(ModelVersion)
            .filter(ModelVersion.status.in_(["evaluated", "active", "retired", "promoted"]))
            .order_by(ModelVersion.version.desc())
            .all()
        )
