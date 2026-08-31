import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.models.dataset import Dataset, DatasetVersion
from app.models.training import TrainingRun, ModelVersion

@pytest.fixture
def test_setup(tmp_path):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Create dataset, dataset_version, training_run
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

    # Create dummy artifact files
    art1 = str(tmp_path / "model1.joblib")
    prep1 = str(tmp_path / "prep1.joblib")
    art2 = str(tmp_path / "model2.joblib")
    prep2 = str(tmp_path / "prep2.joblib")
    for p in [art1, prep1, art2, prep2]:
        with open(p, "w") as f:
            f.write("artifact data")

    # Create 2 model versions
    mv1 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=1,
        algorithm="logistic_regression",
        status="evaluated",
        metrics={"weighted_f1": 0.74, "accuracy": 0.73, "macro_f1": 0.74, "per_class": {"High": {"recall": 0.65}}},
        artifact_path=art1,
        preprocessor_path=prep1,
        training_time_seconds=0.11
    )
    mv2 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=2,
        algorithm="xgboost",
        status="evaluated",
        metrics={"weighted_f1": 0.69, "accuracy": 0.68, "macro_f1": 0.69, "per_class": {"High": {"recall": 0.59}}},
        artifact_path=art2,
        preprocessor_path=prep2,
        training_time_seconds=1.9
    )
    session.add_all([mv1, mv2])
    session.commit()

    client = TestClient(app)
    yield client, session, mv1, mv2

    app.dependency_overrides.clear()
    session.close()

def test_models_api_endpoints(test_setup):
    client, session, mv1, mv2 = test_setup

    # 1. List all models
    list_res = client.get("/api/models")
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2

    # Filter by algorithm
    filter_res = client.get("/api/models?algorithm=logistic_regression")
    assert filter_res.status_code == 200
    assert filter_res.json()["total"] == 1
    assert filter_res.json()["items"][0]["algorithm"] == "logistic_regression"

    # 2. Get active model (initially none)
    active_res1 = client.get("/api/models/active")
    assert active_res1.status_code == 200
    assert active_res1.json() is None

    # 3. Get model by ID
    get_res = client.get(f"/api/models/{mv1.id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == mv1.id

    # 4. Get non-existent model
    not_found = client.get("/api/models/non-existent-id")
    assert not_found.status_code == 404

    # 5. Promote mv1 via POST /api/models/{id}/promote
    promote_res = client.post(f"/api/models/{mv1.id}/promote")
    assert promote_res.status_code == 200
    assert promote_res.json()["status"] == "active"

    # Check active model
    active_res2 = client.get("/api/models/active")
    assert active_res2.status_code == 200
    assert active_res2.json()["id"] == mv1.id
    assert active_res2.json()["status"] == "active"

    # 6. Promote mv2
    promote_res2 = client.post(f"/api/models/{mv2.id}/promote")
    assert promote_res2.status_code == 200
    assert promote_res2.json()["status"] == "active"

    # Check mv1 is now retired
    get_mv1 = client.get(f"/api/models/{mv1.id}")
    assert get_mv1.json()["status"] == "retired"

    # 7. Rollback mv1 via POST /api/models/{id}/rollback
    rollback_res = client.post(f"/api/models/{mv1.id}/rollback")
    assert rollback_res.status_code == 200
    assert rollback_res.json()["status"] == "active"

    # Check mv2 is now retired
    get_mv2 = client.get(f"/api/models/{mv2.id}")
    assert get_mv2.json()["status"] == "retired"

    # 8. Compare models endpoint
    compare_res = client.get("/api/models/compare")
    assert compare_res.status_code == 200
    comp_data = compare_res.json()
    assert comp_data["best_model_id"] == mv1.id
    assert comp_data["best_model_algorithm"] == "logistic_regression"
    assert len(comp_data["compared_models"]) == 2
    assert comp_data["compared_models"][0]["rank"] == 1
    assert comp_data["compared_models"][0]["model_id"] == mv1.id
    assert comp_data["compared_models"][1]["rank"] == 2
    assert comp_data["compared_models"][1]["model_id"] == mv2.id
