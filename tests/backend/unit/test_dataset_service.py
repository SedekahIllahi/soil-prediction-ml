import os
import io
from pathlib import Path
import pytest
from fastapi import UploadFile, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.services.dataset_service import DatasetService

@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def valid_csv_bytes():
    fixture_path = Path(__file__).parent.parent.parent / "fixtures" / "sample_dataset.csv"
    with open(fixture_path, "rb") as f:
        return f.read()

def test_upload_invalid_file_extension(in_memory_db):
    service = DatasetService(in_memory_db)
    file = UploadFile(filename="test.txt", file=io.BytesIO(b"dummy data"))
    with pytest.raises(HTTPException) as exc_info:
        service.upload_and_validate_dataset(file)
    assert exc_info.value.status_code == 400
    assert "Only CSV files" in exc_info.value.detail

def test_upload_valid_csv(in_memory_db, valid_csv_bytes):
    service = DatasetService(in_memory_db)
    file = UploadFile(filename="valid_dataset.csv", file=io.BytesIO(valid_csv_bytes))
    res = service.upload_and_validate_dataset(file)

    assert res.is_valid is True
    assert res.row_count > 0
    assert res.filename == "valid_dataset.csv"
    assert len(res.validation_errors) == 0

    # Cleanup generated file
    if os.path.exists(res.stored_path):
        os.remove(res.stored_path)

def test_preview_and_integration_flow(in_memory_db, valid_csv_bytes):
    service = DatasetService(in_memory_db)
    file = UploadFile(filename="road_data.csv", file=io.BytesIO(valid_csv_bytes))
    upload_res = service.upload_and_validate_dataset(file)

    # Preview
    preview_res = service.get_dataset_preview(upload_res.file_id, limit=5)
    assert preview_res.total_rows > 0
    assert len(preview_res.sample_data) == 5

    # Integration
    ds_res = service.integrate_dataset(
        name="Urban Collapse Dataset V1",
        file_id=upload_res.file_id,
        description="Initial dataset",
    )

    assert ds_res.name == "Urban Collapse Dataset V1"
    assert len(ds_res.versions) == 1
    assert ds_res.latest_version.version == 1

    # Cleanup generated file
    if os.path.exists(upload_res.stored_path):
        os.remove(upload_res.stored_path)
