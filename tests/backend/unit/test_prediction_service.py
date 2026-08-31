import pytest
import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression

from app.core.database import Base
from app.models.dataset import Dataset, DatasetVersion
from app.models.training import TrainingRun, ModelVersion
from app.models.prediction import Prediction
from app.services.prediction_service import PredictionService, invalidate_prediction_cache
from app.schemas.prediction import PredictionRequest
from ml.schema import MODEL_FEATURES, FEATURE_RANGES, TARGET_CLASSES
from ml.pipeline.preprocessing import build_linear_preprocessor

SAMPLE_VALID_FEATURES = {
    "road_age_years": 15.0,
    "road_length_m": 500.0,
    "pavement_thickness_cm": 25.0,
    "surface_crack_density_pct": 12.0,
    "pavement_condition_index": 75.0,
    "rut_depth_mm": 10.0,
    "avg_daily_traffic": 25000.0,
    "heavy_vehicle_pct": 15.0,
    "avg_vehicle_speed_kmh": 45.0,
    "surface_deformation_mm": 15.0,
    "soil_moisture_pct": 25.0,
    "soil_density_g_cm3": 1.75,
    "soil_bearing_capacity_kpa": 250.0,
    "groundwater_depth_m": 8.0,
    "soil_porosity_pct": 35.0,
    "void_ratio": 0.65,
    "soil_settlement_mm": 20.0,
    "elevation_m": 35.0,
    "annual_rainfall_mm": 1500.0,
    "max_daily_rainfall_mm": 85.0,
    "flood_frequency_per_year": 2.0,
    "temperature_variation_c": 20.0,
    "waterlogging_duration_hr": 12.0,
    "drainage_efficiency": 0.75,
    "distance_to_water_body_m": 500.0,
    "underground_pipe_density": 0.45,
    "pipe_age_years": 20.0,
    "distance_to_pipeline_m": 15.0,
    "utility_excavation_count": 3.0,
    "sewer_condition_index": 70.0,
    "land_subsidence_rate_mm_year": 8.0,
    "nearby_construction_intensity": 0.35,
    "building_density_per_km2": 3500.0,
    "distance_to_previous_collapse_m": 1200.0,
}

@pytest.fixture
def prediction_test_setup(tmp_path):
    invalidate_prediction_cache()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()

    # Create dummy dataset and training run
    dataset = Dataset(name="Test DS", adapter_type="urban_road_collapse")
    session.add(dataset)
    session.commit()

    ds_ver = DatasetVersion(
        dataset_id=dataset.id,
        version=1,
        file_path=str(tmp_path / "data.csv"),
        row_count=100,
        column_info={"feature_count": 34}
    )
    session.add(ds_ver)
    session.commit()

    run = TrainingRun(dataset_version_id=ds_ver.id, status="completed")
    session.add(run)
    session.commit()

    # Fit a real small dummy model and preprocessor
    X_dummy = pd.DataFrame([SAMPLE_VALID_FEATURES for _ in range(8)])
    # Add slight variation
    X_dummy["road_age_years"] = [5, 10, 15, 20, 25, 30, 35, 40]
    y_dummy = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # Low, Low, Mod, Mod, High, High, Crit, Crit

    prep = build_linear_preprocessor(feature_names=MODEL_FEATURES)
    X_trans = prep.fit_transform(X_dummy)

    model = LogisticRegression(random_state=42)
    model.fit(X_trans, y_dummy)

    model_path = str(tmp_path / "active_model.joblib")
    prep_path = str(tmp_path / "active_prep.joblib")
    joblib.dump(model, model_path)
    joblib.dump(prep, prep_path)

    mv = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=1,
        algorithm="logistic_regression",
        status="active",
        metrics={"weighted_f1": 0.75, "accuracy": 0.75},
        artifact_path=model_path,
        preprocessor_path=prep_path,
    )
    session.add(mv)
    session.commit()

    yield session, mv, tmp_path
    invalidate_prediction_cache()
    session.close()


def test_predict_success(prediction_test_setup):
    session, mv, tmp_path = prediction_test_setup
    service = PredictionService(session)

    req = PredictionRequest(
        features=SAMPLE_VALID_FEATURES,
        latitude=37.7749,
        longitude=-122.4194,
    )

    resp = service.predict(req)

    assert resp.id is not None
    assert resp.predicted_class in TARGET_CLASSES
    assert 0.0 <= resp.confidence <= 1.0
    assert len(resp.probabilities) == 4
    for c in TARGET_CLASSES:
        assert c in resp.probabilities
        assert 0.0 <= resp.probabilities[c] <= 1.0
    assert resp.model_version_id == mv.id
    assert resp.model_version_number == 1
    assert resp.algorithm == "logistic_regression"
    assert resp.latitude == 37.7749
    assert resp.longitude == -122.4194


def test_predict_no_active_model(prediction_test_setup):
    session, mv, tmp_path = prediction_test_setup
    mv.status = "retired"
    session.commit()
    invalidate_prediction_cache()

    service = PredictionService(session)
    req = PredictionRequest(features=SAMPLE_VALID_FEATURES)

    with pytest.raises(HTTPException) as exc_info:
        service.predict(req)
    assert exc_info.value.status_code == 400
    assert "No active model available" in exc_info.value.detail


def test_predict_missing_feature_raises_validation_error():
    incomplete_features = dict(SAMPLE_VALID_FEATURES)
    del incomplete_features["road_age_years"]

    with pytest.raises(ValueError) as exc_info:
        PredictionRequest(features=incomplete_features)
    assert "Missing required model features" in str(exc_info.value)


def test_predict_out_of_range_raises_validation_error():
    invalid_features = dict(SAMPLE_VALID_FEATURES)
    invalid_features["road_age_years"] = 999.0  # Max is 60

    with pytest.raises(ValueError) as exc_info:
        PredictionRequest(features=invalid_features)
    assert "Feature range validation errors" in str(exc_info.value)


def test_prediction_history_and_detail(prediction_test_setup):
    session, mv, tmp_path = prediction_test_setup
    service = PredictionService(session)

    # Initially empty
    empty_history = service.get_prediction_history()
    assert empty_history.total == 0

    # Make 2 predictions
    req1 = PredictionRequest(features=SAMPLE_VALID_FEATURES)
    p1 = service.predict(req1)

    req2 = PredictionRequest(features=SAMPLE_VALID_FEATURES)
    p2 = service.predict(req2)

    history = service.get_prediction_history(page=1, page_size=10)
    assert history.total == 2
    assert len(history.items) == 2

    # Get detail by ID
    detail = service.get_prediction_by_id(p1.id)
    assert detail.id == p1.id
    assert detail.features["road_age_years"] == 15.0
    assert detail.predicted_class == p1.predicted_class

    # 404 for nonexistent ID
    with pytest.raises(HTTPException) as exc_info:
        service.get_prediction_by_id("nonexistent-id")
    assert exc_info.value.status_code == 404


def test_feature_schema_endpoint(prediction_test_setup):
    session, mv, tmp_path = prediction_test_setup
    service = PredictionService(session)

    schema_resp = service.get_feature_schema()
    assert schema_resp.total_features == 34
    assert len(schema_resp.features) == 34
    assert schema_resp.target_classes == ["Low", "Moderate", "High", "Critical"]

    feature_names = [f.name for f in schema_resp.features]
    for feat in MODEL_FEATURES:
        assert feat in feature_names


def test_dashboard_summary_and_recent(prediction_test_setup):
    session, mv, tmp_path = prediction_test_setup
    service = PredictionService(session)

    # Initially 0 predictions
    summary = service.get_dashboard_summary()
    assert summary.total_predictions == 0
    assert summary.active_model["algorithm"] == "logistic_regression"

    # Make predictions
    service.predict(PredictionRequest(features=SAMPLE_VALID_FEATURES))
    service.predict(PredictionRequest(features=SAMPLE_VALID_FEATURES))

    summary2 = service.get_dashboard_summary()
    assert summary2.total_predictions == 2
    assert sum(summary2.risk_distribution.values()) == 2

    recent = service.get_dashboard_recent(limit=5)
    assert len(recent.items) == 2
