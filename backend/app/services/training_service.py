import os
import time
import joblib
from datetime import datetime, timezone
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.repositories.training_repository import TrainingRepository
from app.schemas.training import (
    TrainingRunResponse,
    TrainingRunListResponse,
    ModelVersionResponse,
)
from ml.adapters.adapter_registry import get_adapter
from ml.models.registry import get_baseline_models, ModelConfig
from ml.pipeline.splitting import DataSplitter
from ml.pipeline.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
    save_preprocessor,
)
from ml.pipeline.target_encoding import TargetEncoderWrapper
from ml.pipeline.training import run_baseline_training
from ml.pipeline.evaluation import evaluate_model
from ml.pipeline.comparison import compare_models
from ml.schema import MODEL_FEATURES

def utc_now():
    return datetime.now(timezone.utc)

class TrainingService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = TrainingRepository(db)
        self.models_dir = os.path.abspath(os.path.join(os.getcwd(), "storage", "models"))
        self.preprocessors_dir = os.path.abspath(os.path.join(os.getcwd(), "storage", "preprocessors"))
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.preprocessors_dir, exist_ok=True)

    def create_training_run(self, dataset_version_id: str, algorithms: list[str] = None) -> TrainingRunResponse:
        ds_ver = self.db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
        if not ds_ver:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DatasetVersion with ID '{dataset_version_id}' not found.",
            )

        config = {"algorithms": algorithms} if algorithms else {}
        run = self.repo.create_run(dataset_version_id=dataset_version_id, config=config)
        return TrainingRunResponse.model_validate(run)

    def execute_training_run(self, run_id: str) -> TrainingRunResponse:
        run = self.repo.get_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"TrainingRun with ID '{run_id}' not found.",
            )

        # Mark as running
        start_time = utc_now()
        self.repo.update_run_status(run_id=run_id, status="running", started_at=start_time)

        try:
            # 1. Load dataset version file
            ds_ver = run.dataset_version
            if not ds_ver or not os.path.exists(ds_ver.file_path):
                raise FileNotFoundError(f"Dataset version file not found at '{ds_ver.file_path if ds_ver else None}'")

            df = pd.read_csv(ds_ver.file_path)

            # 2. Extract X and y using adapter
            adapter_type = ds_ver.dataset.adapter_type if ds_ver.dataset else "urban_road_collapse"
            adapter = get_adapter(adapter_type)
            X, y, _ = adapter.transform(df)

            # 3. Stratified 70/15/15 Split
            splitter = DataSplitter()
            split = splitter.split(X, y)

            # 4. Build unfitted preprocessors & target encoders
            linear_preproc = build_linear_preprocessor(MODEL_FEATURES)
            tree_preproc = build_tree_preprocessor(MODEL_FEATURES)

            target_enc_linear = TargetEncoderWrapper(is_linear=True)
            target_enc_tree = TargetEncoderWrapper(is_linear=False)

            # 5. Filter baseline models if specific algorithms requested
            all_configs = get_baseline_models()
            requested_algos = run.config.get("algorithms") if run.config else None
            if requested_algos:
                model_configs = [c for c in all_configs if c.name in requested_algos]
                if not model_configs:
                    raise ValueError(f"No valid models found matching: {requested_algos}")
            else:
                model_configs = all_configs

            # 6. Run Baseline Training
            (
                trained_models,
                fitted_linear_preproc,
                fitted_tree_preproc,
                fitted_target_enc_linear,
                fitted_target_enc_tree,
            ) = run_baseline_training(
                X_train=split.X_train,
                y_train=split.y_train,
                linear_preprocessor=linear_preproc,
                tree_preprocessor=tree_preproc,
                target_encoder_linear=target_enc_linear,
                target_encoder_tree=target_enc_tree,
                model_configs=model_configs,
            )

            # 7. Evaluate each model on validation set
            eval_results = []
            training_times = {}

            for tm in trained_models:
                if tm.config.model_family == "linear":
                    preproc = fitted_linear_preproc
                    enc = fitted_target_enc_linear
                else:
                    preproc = fitted_tree_preproc
                    enc = fitted_target_enc_tree

                eval_res = evaluate_model(
                    trained_model=tm,
                    X_val_raw=split.X_val,
                    y_val_raw=split.y_val,
                    preprocessor=preproc,
                    target_encoder=enc,
                )
                eval_results.append(eval_res)
                training_times[tm.config.name] = tm.training_time_seconds

            # 8. Compare all candidate models
            comparison_report = compare_models(eval_results, training_times)

            # 9. Save artifacts and persist ModelVersion records to DB
            for tm in trained_models:
                # Find matching eval_result
                matching_eval = next((e for e in eval_results if e.model_name == tm.config.name), None)
                metrics_dict = matching_eval.to_dict() if matching_eval else {}

                # Create preprocessor & model file artifacts
                # Generating model version ID first via repo or temp string
                dummy_ver = self.repo.get_next_version_number()
                model_artifact_path = os.path.join(self.models_dir, f"model_v{dummy_ver}_{tm.config.name}.joblib")
                preproc_artifact_path = os.path.join(self.preprocessors_dir, f"preproc_v{dummy_ver}_{tm.config.name}.joblib")

                # Save trained model estimator
                joblib.dump(tm.model, model_artifact_path)

                # Save corresponding fitted preprocessor
                preproc_to_save = fitted_linear_preproc if tm.config.model_family == "linear" else fitted_tree_preproc
                save_preprocessor(preproc_to_save, preproc_artifact_path)

                # Record in DB
                self.repo.create_model_version(
                    training_run_id=run.id,
                    dataset_version_id=ds_ver.id,
                    algorithm=tm.config.name,
                    metrics=metrics_dict,
                    hyperparameters=tm.config.hyperparameters,
                    artifact_path=model_artifact_path,
                    preprocessor_path=preproc_artifact_path,
                    training_time_seconds=tm.training_time_seconds,
                    status="evaluated",
                )

            # 10. Update TrainingRun status to completed
            end_time = utc_now()
            updated_run = self.repo.update_run_status(
                run_id=run.id,
                status="completed",
                comparison_results=comparison_report.to_dict(),
                completed_at=end_time,
            )
            return TrainingRunResponse.model_validate(updated_run)

        except Exception as err:
            end_time = utc_now()
            self.repo.update_run_status(
                run_id=run.id,
                status="failed",
                error_message=str(err),
                completed_at=end_time,
            )
            raise err

    def get_run(self, run_id: str) -> TrainingRunResponse:
        run = self.repo.get_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"TrainingRun with ID '{run_id}' not found.",
            )
        return TrainingRunResponse.model_validate(run)

    def list_runs(self, page: int = 1, page_size: int = 20) -> TrainingRunListResponse:
        skip = (page - 1) * page_size
        runs, total = self.repo.list_runs(skip=skip, limit=page_size)
        items = [TrainingRunResponse.model_validate(r) for r in runs]
        return TrainingRunListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    def promote_model(self, model_version_id: str) -> ModelVersionResponse:
        mv = self.repo.get_model_version(model_version_id)
        if not mv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ModelVersion with ID '{model_version_id}' not found.",
            )
        active_mv = self.repo.set_active_model(model_version_id)
        return ModelVersionResponse.model_validate(active_mv)

    def get_active_model(self) -> Optional[ModelVersionResponse]:
        active_mv = self.repo.get_active_model()
        if not active_mv:
            return None
        return ModelVersionResponse.model_validate(active_mv)
