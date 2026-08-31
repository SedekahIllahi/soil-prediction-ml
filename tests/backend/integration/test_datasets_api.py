import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db

@pytest.fixture
def test_db_session():
    # Use StaticPool so all threads share the exact same in-memory SQLite connection
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

def test_api_dataset_lifecycle(test_db_session, sample_csv_content):
    client = TestClient(app)

    # 1. Upload CSV dataset
    files = {"file": ("test_road_collapse.csv", sample_csv_content, "text/csv")}
    upload_res = client.post("/api/datasets/upload", files=files)
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    assert upload_data["is_valid"] is True
    file_id = upload_data["file_id"]

    # 2. Preview dataset
    preview_res = client.get(f"/api/datasets/preview/{file_id}")
    assert preview_res.status_code == 200
    preview_data = preview_res.json()
    assert preview_data["total_rows"] > 0

    # 3. Integrate dataset
    integrate_res = client.post(
        "/api/datasets/integrate",
        json={
            "name": "Road Collapse Trial Dataset",
            "file_id": file_id,
            "description": "Integration test dataset",
        },
    )
    assert integrate_res.status_code == 201
    dataset_data = integrate_res.json()
    dataset_id = dataset_data["id"]
    assert dataset_data["name"] == "Road Collapse Trial Dataset"
    assert len(dataset_data["versions"]) == 1

    # 4. List datasets
    list_res = client.get("/api/datasets")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] == 1

    # 5. Get dataset details
    get_res = client.get(f"/api/datasets/{dataset_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["id"] == dataset_id

    # Cleanup created storage file
    if os.path.exists(upload_data["stored_path"]):
        os.remove(upload_data["stored_path"])
