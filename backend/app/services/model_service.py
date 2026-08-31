import os
from datetime import datetime, timezone
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repositories.model_repository import ModelRepository
from app.schemas.model import (
    ModelVersionResponse,
    ModelVersionListResponse,
    ModelComparisonItem,
    ModelComparisonResponse,
)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ModelService:
    """
    Encapsulates business rules and lifecycle operations for ModelVersion entities:
    - Model listing and detail retrieval
    - Active model querying
    - Model promotion with prerequisite validation (metrics & artifact existence)
    - Model rollback (re-promoting retired models)
    - Side-by-side model comparison and ranking
    """

    def __init__(self, db: Session):
        self.db = db
        self.repo = ModelRepository(db)

    def list_models(
        self,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[str] = None,
        algorithm_filter: Optional[str] = None,
    ) -> ModelVersionListResponse:
        skip = (page - 1) * page_size
        items, total = self.repo.list_models(
            skip=skip,
            limit=page_size,
            status=status_filter,
            algorithm=algorithm_filter,
        )
        return ModelVersionListResponse(
            items=[ModelVersionResponse.model_validate(mv) for mv in items],
            total=total,
            page=page,
            page_size=page_size,
        )

    def get_model(self, model_id: str) -> ModelVersionResponse:
        mv = self.repo.get_model(model_id)
        if not mv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ModelVersion with ID '{model_id}' not found.",
            )
        return ModelVersionResponse.model_validate(mv)

    def get_active_model(self) -> Optional[ModelVersionResponse]:
        active_mv = self.repo.get_active_model()
        if not active_mv:
            return None
        return ModelVersionResponse.model_validate(active_mv)

    def promote_model(self, model_id: str) -> ModelVersionResponse:
        """
        Promotes a model version to active status for inference.
        Enforces constraints:
        1. Model must exist.
        2. Model must have valid evaluation metrics recorded.
        3. Serialized artifact and preprocessor files must exist on disk.
        4. Demotes previously active model to 'retired'.
        """
        mv = self.repo.get_model(model_id)
        if not mv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ModelVersion with ID '{model_id}' not found.",
            )

        # Enforce: model must have evaluation metrics
        if not mv.metrics or not isinstance(mv.metrics, dict) or len(mv.metrics) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"ModelVersion '{model_id}' cannot be promoted because it has no evaluation metrics.",
            )

        # Enforce: artifact files must exist on disk
        if mv.artifact_path and not os.path.exists(mv.artifact_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model artifact file '{mv.artifact_path}' not found on disk.",
            )
        if mv.preprocessor_path and not os.path.exists(mv.preprocessor_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Preprocessor artifact file '{mv.preprocessor_path}' not found on disk.",
            )

        promoted_mv = self.repo.set_active_model(model_id)
        from app.services.prediction_service import invalidate_prediction_cache
        invalidate_prediction_cache()
        return ModelVersionResponse.model_validate(promoted_mv)

    def rollback_model(self, model_id: str) -> ModelVersionResponse:
        """
        Semantic alias for re-promoting a retired or previous model version.
        """
        return self.promote_model(model_id)

    def compare_models(self, model_ids: Optional[list[str]] = None) -> ModelComparisonResponse:
        """
        Compares multiple model versions side-by-side and ranks them by Weighted F1
        (primary) and High-class Recall (secondary).
        """
        if model_ids:
            models = self.repo.get_models_by_ids(model_ids)
            found_ids = {m.id for m in models}
            missing_ids = set(model_ids) - found_ids
            if missing_ids:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"The following model IDs were not found: {list(missing_ids)}",
                )
        else:
            models = self.repo.get_all_evaluated_models()

        if not models:
            return ModelComparisonResponse(
                best_model_id=None,
                best_model_algorithm=None,
                primary_metric="weighted_f1",
                secondary_metric="high_class_recall",
                compared_models=[],
                compared_at=utc_now(),
            )

        comparison_items: list[ModelComparisonItem] = []
        for mv in models:
            metrics = mv.metrics or {}
            weighted_f1 = float(metrics.get("weighted_f1", 0.0))
            macro_f1 = float(metrics.get("macro_f1", 0.0))
            accuracy = float(metrics.get("accuracy", 0.0))
            
            per_class = metrics.get("per_class", {})
            high_class_recall = float(per_class.get("High", {}).get("recall", 0.0))

            comparison_items.append(
                ModelComparisonItem(
                    rank=0,  # assigned after sorting
                    model_id=mv.id,
                    version=mv.version,
                    algorithm=mv.algorithm,
                    status=mv.status,
                    weighted_f1=weighted_f1,
                    macro_f1=macro_f1,
                    accuracy=accuracy,
                    high_class_recall=high_class_recall,
                    high_risk_recall=high_class_recall,
                    training_time_seconds=float(mv.training_time_seconds or 0.0),
                    per_class_metrics=per_class,
                    created_at=mv.created_at,
                )
            )

        # Rank by Weighted F1 descending, then High-class Recall descending
        comparison_items.sort(key=lambda x: (x.weighted_f1, x.high_class_recall), reverse=True)

        for i, item in enumerate(comparison_items):
            item.rank = i + 1

        best_item = comparison_items[0] if comparison_items else None

        return ModelComparisonResponse(
            best_model_id=best_item.model_id if best_item else None,
            best_model_algorithm=best_item.algorithm if best_item else None,
            primary_metric="weighted_f1",
            secondary_metric="high_class_recall",
            compared_models=comparison_items,
            compared_at=utc_now(),
        )
