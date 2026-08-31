import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.services.training_service import TrainingService

@pytest.fixture
def test_db_session():
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
    yield session
    app.dependency_overrides.clear()
    session.close()

@pytest.fixture
def sample_csv_content():
    fixture_path = Path(__file__).parent.parent.parent / "fixtures" / "sample_dataset.csv"
    with open(fixture_path, "rb") as f:
        return f.read()

def test_training_and_model_api_lifecycle(test_db_session, sample_csv_content, monkeypatch):
    # Prevent FastAPI BackgroundTasks from spawning against main DB in tests
    monkeypatch.setattr("app.api.training.run_training_task", lambda run_id, db_factory: None)

    client = TestClient(app)


    # 1. Upload & Integrate Dataset
    files = {"file": ("test_road_collapse.csv", sample_csv_content, "text/csv")}
    upload_res = client.post("/api/datasets/upload", files=files)
    assert upload_res.status_code == 201
    file_id = upload_res.json()["file_id"]

    integrate_res = client.post(
        "/api/datasets/integrate",
        json={"name": "Training Test Dataset", "file_id": file_id},
    )
    assert integrate_res.status_code == 201
    dataset_version_id = integrate_res.json()["latest_version"]["id"]

    # 2. Start Training Run (request decision_tree only for fast test execution)
    start_res = client.post(
        "/api/training",
        json={
            "dataset_version_id": dataset_version_id,
            "algorithms": ["decision_tree"]
        }
    )
    assert start_res.status_code == 201
    run_id = start_res.json()["id"]
    assert start_res.json()["status"] == "pending"

    # 3. Synchronously execute training run using service with test session
    service = TrainingService(test_db_session)
    completed_run = service.execute_training_run(run_id)
    assert completed_run.status == "completed"

    # 4. Get Training Run Details via API
    get_run_res = client.get(f"/api/training/{run_id}")
    assert get_run_res.status_code == 200
    run_data = get_run_res.json()
    assert run_data["status"] == "completed"
    assert "comparison_results" in run_data
    assert len(run_data["model_versions"]) == 1
    model_version_id = run_data["model_versions"][0]["id"]

    # 5. List Training Runs
    list_runs_res = client.get("/api/training")
    assert list_runs_res.status_code == 200
    assert list_runs_res.json()["total"] == 1

    # 6. Promote Model Version to Active
    promote_res = client.post(
        "/api/models/promote",
        json={"model_version_id": model_version_id}
    )
    assert promote_res.status_code == 200
    assert promote_res.json()["status"] == "active"

    # 7. Get Active Model
    active_res = client.get("/api/models/active")
    assert active_res.status_code == 200
    assert active_res.json()["id"] == model_version_id
    assert active_res.json()["status"] == "active"

    # Cleanup uploaded file
    stored_path = upload_res.json()["stored_path"]
    if os.path.exists(stored_path):
        os.remove(stored_path)
