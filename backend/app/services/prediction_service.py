import os
from typing import Optional, Any
import numpy as np
import pandas as pd
import joblib
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.prediction import Prediction
from app.repositories.model_repository import ModelRepository
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionDetailResponse,
    PredictionListResponse,
    FeatureField,
    FeatureSchemaResponse,
    DashboardSummaryResponse,
    DashboardRecentResponse,
)
from ml.schema import (
    MODEL_FEATURES,
    FEATURE_RANGES,
    CAUTION_FEATURES,
    TARGET_CLASSES,
)

# Global in-memory cache for the active estimator and preprocessor
_CACHED_MODEL_ID: Optional[str] = None
_CACHED_MODEL: Any = None
_CACHED_PREPROCESSOR: Any = None

def invalidate_prediction_cache() -> None:
    """Clears the loaded in-memory model and preprocessor cache."""
    global _CACHED_MODEL_ID, _CACHED_MODEL, _CACHED_PREPROCESSOR
    _CACHED_MODEL_ID = None
    _CACHED_MODEL = None
    _CACHED_PREPROCESSOR = None


# Feature category taxonomy for frontend schema display
FEATURE_CATEGORIES = {
    "road_age_years": "Road Infrastructure",
    "road_length_m": "Road Infrastructure",
    "pavement_thickness_cm": "Road Infrastructure",
    "surface_crack_density_pct": "Road Infrastructure",
    "pavement_condition_index": "Road Infrastructure",
    "rut_depth_mm": "Road Infrastructure",
    "avg_daily_traffic": "Traffic Load",
    "heavy_vehicle_pct": "Traffic Load",
    "avg_vehicle_speed_kmh": "Traffic Load",
    "surface_deformation_mm": "Traffic Load",
    "soil_moisture_pct": "Geotechnical & Soil",
    "soil_density_g_cm3": "Geotechnical & Soil",
    "soil_bearing_capacity_kpa": "Geotechnical & Soil",
    "groundwater_depth_m": "Geotechnical & Soil",
    "soil_porosity_pct": "Geotechnical & Soil",
    "void_ratio": "Geotechnical & Soil",
    "soil_settlement_mm": "Geotechnical & Soil",
    "elevation_m": "Climatic & Hydrological",
    "annual_rainfall_mm": "Climatic & Hydrological",
    "max_daily_rainfall_mm": "Climatic & Hydrological",
    "flood_frequency_per_year": "Climatic & Hydrological",
    "temperature_variation_c": "Climatic & Hydrological",
    "waterlogging_duration_hr": "Climatic & Hydrological",
    "drainage_efficiency": "Drainage & Hydrology",
    "distance_to_water_body_m": "Drainage & Hydrology",
    "underground_pipe_density": "Underground Infrastructure",
    "pipe_age_years": "Underground Infrastructure",
    "distance_to_pipeline_m": "Underground Infrastructure",
    "utility_excavation_count": "Underground Infrastructure",
    "sewer_condition_index": "Underground Infrastructure",
    "land_subsidence_rate_mm_year": "Underground Infrastructure",
    "nearby_construction_intensity": "Urban & Environmental",
    "building_density_per_km2": "Urban & Environmental",
    "distance_to_previous_collapse_m": "Urban & Environmental",
}


class PredictionService:
    """
    Orchestrates the complete prediction lifecycle:
    - In-memory model and preprocessor caching
    - Feature validation and schema generation
    - Inference execution and probability mapping
    - Prediction history management and analytics
    """

    def __init__(self, db: Session):
        self.db = db
        self.model_repo = ModelRepository(db)
        self.prediction_repo = PredictionRepository(db)

    def _get_loaded_artifacts(self) -> tuple[Any, Any, Any]:
        """
        Retrieves the active model version and returns (model_version, loaded_model, loaded_preprocessor)
        utilizing the in-memory cache to prevent redundant disk reads.
        """
        global _CACHED_MODEL_ID, _CACHED_MODEL, _CACHED_PREPROCESSOR

        active_mv = self.model_repo.get_active_model()
        if not active_mv:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active model available for predictions. Please promote a model version first.",
            )

        if _CACHED_MODEL_ID != active_mv.id or _CACHED_MODEL is None or _CACHED_PREPROCESSOR is None:
            if not active_mv.artifact_path or not os.path.exists(active_mv.artifact_path):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Model artifact file for active model {active_mv.id} was not found on disk.",
                )
            if not active_mv.preprocessor_path or not os.path.exists(active_mv.preprocessor_path):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Preprocessor artifact file for active model {active_mv.id} was not found on disk.",
                )

            try:
                _CACHED_MODEL = joblib.load(active_mv.artifact_path)
                _CACHED_PREPROCESSOR = joblib.load(active_mv.preprocessor_path)
                _CACHED_MODEL_ID = active_mv.id
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load model artifacts for active model {active_mv.id}: {str(e)}",
                )

        return active_mv, _CACHED_MODEL, _CACHED_PREPROCESSOR

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Executes a real-time risk prediction for a 34-feature payload:
        1. Loads active model & preprocessor (cached)
        2. Formats features into a single-row DataFrame in canonical column order
        3. Transforms inputs through fitted preprocessor
        4. Predicts class probabilities and selects highest-confidence class
        5. Persists prediction to database
        6. Returns structured response
        """
        active_mv, model, preprocessor = self._get_loaded_artifacts()

        # Construct DataFrame strictly adhering to canonical feature order
        ordered_data = {col: [request.features[col]] for col in MODEL_FEATURES}
        df_input = pd.DataFrame(ordered_data)

        try:
            X_trans = preprocessor.transform(df_input)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Preprocessing failed on submitted features: {str(e)}",
            )

        # Predict probabilities
        probabilities: dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(X_trans)[0]
            # Map probabilities to classes
            model_classes = getattr(model, "classes_", np.arange(len(raw_probs)))
            for idx, prob in enumerate(raw_probs):
                cls_key = model_classes[idx]
                # If class is integer 0..3, map to TARGET_CLASSES
                if isinstance(cls_key, (int, np.integer)):
                    cls_name = TARGET_CLASSES[int(cls_key)] if int(cls_key) < len(TARGET_CLASSES) else str(cls_key)
                else:
                    cls_name = str(cls_key)
                probabilities[cls_name] = round(float(prob), 4)
        else:
            # Fallback for models without predict_proba
            pred_idx = model.predict(X_trans)[0]
            predicted_class_name = TARGET_CLASSES[int(pred_idx)] if isinstance(pred_idx, (int, np.integer)) else str(pred_idx)
            for c in TARGET_CLASSES:
                probabilities[c] = 1.0 if c == predicted_class_name else 0.0

        # Ensure all 4 classes are present in probabilities dict
        for c in TARGET_CLASSES:
            if c not in probabilities:
                probabilities[c] = 0.0

        # Determine predicted class (highest probability)
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_class]

        # Save to database
        saved_pred = self.prediction_repo.create_prediction(
            model_version_id=active_mv.id,
            features=request.features,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            latitude=request.latitude,
            longitude=request.longitude,
        )

        return PredictionResponse(
            id=saved_pred.id,
            predicted_class=saved_pred.predicted_class,
            confidence=saved_pred.confidence,
            probabilities=saved_pred.probabilities,
            model_version_id=active_mv.id,
            model_version_number=active_mv.version,
            algorithm=active_mv.algorithm,
            latitude=saved_pred.latitude,
            longitude=saved_pred.longitude,
            created_at=saved_pred.created_at,
        )

    def get_prediction_history(
        self,
        page: int = 1,
        page_size: int = 20,
        risk_category: Optional[str] = None,
    ) -> PredictionListResponse:
        skip = (page - 1) * page_size
        items, total = self.prediction_repo.list_predictions(
            skip=skip,
            limit=page_size,
            risk_category=risk_category,
        )
        
        response_items = []
        for p in items:
            mv = p.model_version
            response_items.append(
                PredictionResponse(
                    id=p.id,
                    predicted_class=p.predicted_class,
                    confidence=p.confidence,
                    probabilities=p.probabilities,
                    model_version_id=p.model_version_id,
                    model_version_number=mv.version if mv else None,
                    algorithm=mv.algorithm if mv else None,
                    latitude=p.latitude,
                    longitude=p.longitude,
                    created_at=p.created_at,
                )
            )

        return PredictionListResponse(
            items=response_items,
            total=total,
            page=page,
            page_size=page_size,
        )

    def get_prediction_by_id(self, prediction_id: str) -> PredictionDetailResponse:
        p = self.prediction_repo.get_prediction(prediction_id)
        if not p:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prediction with ID '{prediction_id}' not found.",
            )
        mv = p.model_version
        return PredictionDetailResponse(
            id=p.id,
            features=p.features,
            predicted_class=p.predicted_class,
            confidence=p.confidence,
            probabilities=p.probabilities,
            model_version_id=p.model_version_id,
            model_version_number=mv.version if mv else None,
            algorithm=mv.algorithm if mv else None,
            latitude=p.latitude,
            longitude=p.longitude,
            created_at=p.created_at,
        )

    def get_feature_schema(self) -> FeatureSchemaResponse:
        """Returns metadata for all 34 canonical features for frontend form generation."""
        fields: list[FeatureField] = []
        for feat in MODEL_FEATURES:
            min_val, max_val = FEATURE_RANGES.get(feat, (0.0, 100.0))
            is_caution = feat in CAUTION_FEATURES
            category = FEATURE_CATEGORIES.get(feat, "General")
            fields.append(
                FeatureField(
                    name=feat,
                    type="float",
                    min_val=min_val,
                    max_val=max_val,
                    is_caution=is_caution,
                    category=category,
                    description=f"{feat.replace('_', ' ').title()}",
                )
            )

        return FeatureSchemaResponse(
            features=fields,
            target_classes=list(TARGET_CLASSES),
            total_features=len(fields),
        )

    def get_dashboard_summary(self) -> DashboardSummaryResponse:
        summary_data = self.prediction_repo.get_dashboard_summary()
        active_mv = self.model_repo.get_active_model()
        active_model_dict = None
        if active_mv:
            active_model_dict = {
                "id": active_mv.id,
                "version": active_mv.version,
                "algorithm": active_mv.algorithm,
                "metrics": active_mv.metrics,
                "created_at": active_mv.created_at.isoformat() if active_mv.created_at else None,
            }

        return DashboardSummaryResponse(
            total_predictions=summary_data["total_predictions"],
            risk_distribution=summary_data["risk_distribution"],
            active_model=active_model_dict,
        )

    def get_dashboard_recent(self, limit: int = 10) -> DashboardRecentResponse:
        items = self.prediction_repo.get_recent_predictions(limit=limit)
        response_items = []
        for p in items:
            mv = p.model_version
            response_items.append(
                PredictionResponse(
                    id=p.id,
                    predicted_class=p.predicted_class,
                    confidence=p.confidence,
                    probabilities=p.probabilities,
                    model_version_id=p.model_version_id,
                    model_version_number=mv.version if mv else None,
                    algorithm=mv.algorithm if mv else None,
                    latitude=p.latitude,
                    longitude=p.longitude,
                    created_at=p.created_at,
                )
            )
        return DashboardRecentResponse(items=response_items)
