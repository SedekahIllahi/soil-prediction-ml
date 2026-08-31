import pytest
import os
import joblib
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sklearn.linear_model import LogisticRegression

from app.main import app
from app.core.database import Base, get_db
from app.models.dataset import Dataset, DatasetVersion
from app.models.training import TrainingRun, ModelVersion
from app.services.prediction_service import invalidate_prediction_cache
from ml.schema import TARGET_CLASSES, MODEL_FEATURES
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
def api_test_client(tmp_path):
    invalidate_prediction_cache()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

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
    X_dummy["road_age_years"] = [5, 10, 15, 20, 25, 30, 35, 40]
    y_dummy = np.array([0, 0, 1, 1, 2, 2, 3, 3])

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
    mv_id = mv.id
    session.close()

    client = TestClient(app)
    yield client, mv_id
    app.dependency_overrides.clear()
    invalidate_prediction_cache()


def test_post_prediction_endpoint(api_test_client):
    client, mv_id = api_test_client

    payload = {
        "features": SAMPLE_VALID_FEATURES,
        "latitude": -6.2088,
        "longitude": 106.8456
    }
    response = client.post("/api/predictions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["predicted_class"] in TARGET_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
    assert data["model_version_id"] == mv_id
    assert data["model_version_number"] == 1
    assert data["algorithm"] == "logistic_regression"
    assert data["latitude"] == -6.2088
    assert data["longitude"] == 106.8456


def test_post_prediction_missing_features(api_test_client):
    client, _ = api_test_client
    incomplete = dict(SAMPLE_VALID_FEATURES)
    del incomplete["road_age_years"]

    response = client.post("/api/predictions", json={"features": incomplete})
    assert response.status_code == 422


def test_post_prediction_out_of_range(api_test_client):
    client, _ = api_test_client
    invalid = dict(SAMPLE_VALID_FEATURES)
    invalid["road_age_years"] = 500  # Max is 60

    response = client.post("/api/predictions", json={"features": invalid})
    assert response.status_code == 422


def test_get_predictions_history(api_test_client):
    client, _ = api_test_client

    # Create 2 predictions
    client.post("/api/predictions", json={"features": SAMPLE_VALID_FEATURES})
    client.post("/api/predictions", json={"features": SAMPLE_VALID_FEATURES})

    response = client.get("/api/predictions?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


def test_get_prediction_detail(api_test_client):
    client, _ = api_test_client

    create_res = client.post("/api/predictions", json={"features": SAMPLE_VALID_FEATURES})
    pred_id = create_res.json()["id"]

    detail_res = client.get(f"/api/predictions/{pred_id}")
    assert detail_res.status_code == 200
    data = detail_res.json()
    assert data["id"] == pred_id
    assert "features" in data
    assert data["features"]["road_age_years"] == 15.0


def test_schema_features_endpoint(api_test_client):
    client, _ = api_test_client

    response = client.get("/api/schema/features")
    assert response.status_code == 200
    data = response.json()
    assert data["total_features"] == 34
    assert len(data["features"]) == 34
    assert data["target_classes"] == ["Low", "Moderate", "High", "Critical"]


def test_dashboard_endpoints(api_test_client):
    client, _ = api_test_client

    # Check initial summary
    sum_res = client.get("/api/dashboard/summary")
    assert sum_res.status_code == 200
    sum_data = sum_res.json()
    assert sum_data["total_predictions"] == 0
    assert sum_data["active_model"]["algorithm"] == "logistic_regression"

    # Post a prediction
    client.post("/api/predictions", json={"features": SAMPLE_VALID_FEATURES})

    # Check updated summary
    sum_res2 = client.get("/api/dashboard/summary")
    assert sum_res2.status_code == 200
    assert sum_res2.json()["total_predictions"] == 1

    # Check recent predictions
    rec_res = client.get("/api/dashboard/recent?limit=5")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert len(rec_data["items"]) == 1
