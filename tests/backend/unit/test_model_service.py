import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException

from app.core.database import Base
from app.models.dataset import Dataset, DatasetVersion
from app.models.training import TrainingRun, ModelVersion
from app.services.model_service import ModelService

@pytest.fixture
def db_session(tmp_path):
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

    yield session, run, ds_ver, tmp_path
    session.close()

def test_list_and_get_models(db_session):
    session, run, ds_ver, tmp_path = db_session
    service = ModelService(session)

    # Empty list
    empty_list = service.list_models()
    assert empty_list.total == 0
    assert len(empty_list.items) == 0

    # Add 2 model versions
    mv1 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=1,
        algorithm="logistic_regression",
        status="evaluated",
        metrics={"weighted_f1": 0.75, "accuracy": 0.74, "macro_f1": 0.74, "per_class": {"High": {"recall": 0.65}}},
        training_time_seconds=0.1
    )
    mv2 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=2,
        algorithm="random_forest",
        status="evaluated",
        metrics={"weighted_f1": 0.68, "accuracy": 0.67, "macro_f1": 0.67, "per_class": {"High": {"recall": 0.58}}},
        training_time_seconds=0.8
    )
    session.add_all([mv1, mv2])
    session.commit()

    # List
    res = service.list_models(page=1, page_size=10)
    assert res.total == 2
    assert len(res.items) == 2

    # Filter by algorithm
    lr_res = service.list_models(algorithm_filter="logistic_regression")
    assert lr_res.total == 1
    assert lr_res.items[0].algorithm == "logistic_regression"

    # Get by ID
    model_detail = service.get_model(mv1.id)
    assert model_detail.id == mv1.id
    assert model_detail.algorithm == "logistic_regression"

    # Get non-existent
    with pytest.raises(HTTPException) as exc:
        service.get_model("non-existent-id")
    assert exc.value.status_code == 404

def test_active_model_and_promotion_lifecycle(db_session):
    session, run, ds_ver, tmp_path = db_session
    service = ModelService(session)

    # Initially no active model
    assert service.get_active_model() is None

    # Create dummy artifact files
    artifact_path1 = str(tmp_path / "model1.joblib")
    preproc_path1 = str(tmp_path / "preproc1.joblib")
    with open(artifact_path1, "w") as f:
        f.write("dummy model")
    with open(preproc_path1, "w") as f:
        f.write("dummy preproc")

    mv1 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=1,
        algorithm="logistic_regression",
        status="evaluated",
        metrics={"weighted_f1": 0.75, "accuracy": 0.74, "macro_f1": 0.74, "per_class": {"High": {"recall": 0.65}}},
        artifact_path=artifact_path1,
        preprocessor_path=preproc_path1,
        training_time_seconds=0.1
    )
    session.add(mv1)
    session.commit()

    # Promote mv1
    promoted1 = service.promote_model(mv1.id)
    assert promoted1.status == "active"
    assert service.get_active_model().id == mv1.id

    # Create second model
    artifact_path2 = str(tmp_path / "model2.joblib")
    preproc_path2 = str(tmp_path / "preproc2.joblib")
    with open(artifact_path2, "w") as f:
        f.write("dummy model 2")
    with open(preproc_path2, "w") as f:
        f.write("dummy preproc 2")

    mv2 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=2,
        algorithm="xgboost",
        status="evaluated",
        metrics={"weighted_f1": 0.78, "accuracy": 0.77, "macro_f1": 0.77, "per_class": {"High": {"recall": 0.70}}},
        artifact_path=artifact_path2,
        preprocessor_path=preproc_path2,
        training_time_seconds=1.2
    )
    session.add(mv2)
    session.commit()

    # Promote mv2 -> mv1 should become retired, mv2 active
    promoted2 = service.promote_model(mv2.id)
    assert promoted2.status == "active"
    assert service.get_active_model().id == mv2.id

    session.refresh(mv1)
    assert mv1.status == "retired"

    # Rollback to mv1 -> mv1 active, mv2 retired
    rolled_back = service.rollback_model(mv1.id)
    assert rolled_back.status == "active"
    assert service.get_active_model().id == mv1.id

    session.refresh(mv2)
    assert mv2.status == "retired"

def test_promotion_validations(db_session):
    session, run, ds_ver, tmp_path = db_session
    service = ModelService(session)

    # 1. Missing evaluation metrics
    mv_no_metrics = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=1,
        algorithm="logistic_regression",
        status="evaluated",
        metrics={},
        artifact_path=None,
        preprocessor_path=None
    )
    session.add(mv_no_metrics)
    session.commit()

    with pytest.raises(HTTPException) as exc:
        service.promote_model(mv_no_metrics.id)
    assert exc.value.status_code == 400
    assert "no evaluation metrics" in exc.value.detail

    # 2. Missing artifact file on disk
    mv_missing_file = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=2,
        algorithm="logistic_regression",
        status="evaluated",
        metrics={"weighted_f1": 0.75},
        artifact_path=str(tmp_path / "non_existent_model.joblib"),
        preprocessor_path=None
    )
    session.add(mv_missing_file)
    session.commit()

    with pytest.raises(HTTPException) as exc2:
        service.promote_model(mv_missing_file.id)
    assert exc2.value.status_code == 400
    assert "not found on disk" in exc2.value.detail

def test_compare_models(db_session):
    session, run, ds_ver, tmp_path = db_session
    service = ModelService(session)

    mv1 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=1,
        algorithm="decision_tree",
        status="evaluated",
        metrics={"weighted_f1": 0.54, "accuracy": 0.54, "macro_f1": 0.54, "per_class": {"High": {"recall": 0.40}}},
        training_time_seconds=0.3
    )
    mv2 = ModelVersion(
        training_run_id=run.id,
        dataset_version_id=ds_ver.id,
        version=2,
        algorithm="logistic_regression",
        status="evaluated",
        metrics={"weighted_f1": 0.74, "accuracy": 0.73, "macro_f1": 0.74, "per_class": {"High": {"recall": 0.65}}},
        training_time_seconds=0.1
    )
    session.add_all([mv1, mv2])
    session.commit()

    # Compare all
    comp = service.compare_models()
    assert comp.best_model_id == mv2.id
    assert comp.best_model_algorithm == "logistic_regression"
    assert len(comp.compared_models) == 2
    assert comp.compared_models[0].rank == 1
    assert comp.compared_models[0].model_id == mv2.id
    assert comp.compared_models[1].rank == 2
    assert comp.compared_models[1].model_id == mv1.id

    # Compare subset
    comp_subset = service.compare_models(model_ids=[mv1.id])
    assert len(comp_subset.compared_models) == 1
    assert comp_subset.compared_models[0].model_id == mv1.id

    # Compare with non-existent ID
    with pytest.raises(HTTPException) as exc:
        service.compare_models(model_ids=["fake-id"])
    assert exc.value.status_code == 404
