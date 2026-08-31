from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.training import TrainingRun, ModelVersion

class TrainingRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_run(self, dataset_version_id: str, config: Optional[dict] = None) -> TrainingRun:
        run = TrainingRun(
            dataset_version_id=dataset_version_id,
            status="pending",
            config=config or {}
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        return self.db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

    def list_runs(self, skip: int = 0, limit: int = 20) -> tuple[list[TrainingRun], int]:
        total = self.db.query(func.count(TrainingRun.id)).scalar()
        runs = self.db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).offset(skip).limit(limit).all()
        return runs, total

    def update_run_status(
        self, 
        run_id: str, 
        status: str, 
        comparison_results: Optional[dict] = None, 
        error_message: Optional[str] = None,
        started_at=None,
        completed_at=None
    ) -> Optional[TrainingRun]:
        run = self.get_run(run_id)
        if not run:
            return None
        
        run.status = status
        if comparison_results is not None:
            run.comparison_results = comparison_results
        if error_message is not None:
            run.error_message = error_message
        if started_at is not None:
            run.started_at = started_at
        if completed_at is not None:
            run.completed_at = completed_at

        self.db.commit()
        self.db.refresh(run)
        return run

    def get_next_version_number(self) -> int:
        max_ver = self.db.query(func.max(ModelVersion.version)).scalar()
        return (max_ver or 0) + 1

    def create_model_version(
        self,
        training_run_id: str,
        dataset_version_id: str,
        algorithm: str,
        metrics: dict,
        hyperparameters: dict,
        artifact_path: str,
        preprocessor_path: str,
        training_time_seconds: float,
        status: str = "evaluated"
    ) -> ModelVersion:
        version_num = self.get_next_version_number()
        model_ver = ModelVersion(
            training_run_id=training_run_id,
            dataset_version_id=dataset_version_id,
            version=version_num,
            algorithm=algorithm,
            status=status,
            metrics=metrics,
            hyperparameters=hyperparameters,
            artifact_path=artifact_path,
            preprocessor_path=preprocessor_path,
            training_time_seconds=training_time_seconds
        )
        self.db.add(model_ver)
        self.db.commit()
        self.db.refresh(model_ver)
        return model_ver

    def get_model_version(self, model_version_id: str) -> Optional[ModelVersion]:
        return self.db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()

    def get_active_model(self) -> Optional[ModelVersion]:
        return self.db.query(ModelVersion).filter(ModelVersion.status == "active").first()

    def set_active_model(self, model_version_id: str) -> Optional[ModelVersion]:
        # Demote current active model to evaluated/retired if exists
        current_active = self.db.query(ModelVersion).filter(ModelVersion.status == "active").all()
        for mv in current_active:
            mv.status = "promoted"  # or retired
        
        target = self.get_model_version(model_version_id)
        if target:
            target.status = "active"
            self.db.commit()
            self.db.refresh(target)
            return target
        return None
