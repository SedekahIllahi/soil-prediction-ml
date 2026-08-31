import os
import re
import uuid
from typing import Any, List, Tuple
import pandas as pd
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.dataset import (
    DatasetUploadResponse,
    DatasetPreviewResponse,
    DatasetResponse,
    DatasetVersionResponse,
    DatasetListResponse,
)
from ml.adapters.adapter_registry import get_adapter

class DatasetService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = DatasetRepository(db)
        self.storage_dir = os.path.abspath(os.path.join(os.getcwd(), "storage", "datasets"))
        os.makedirs(self.storage_dir, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        base = os.path.basename(filename)
        sanitized = re.sub(r"[^a-zA-Z0-9_\.-]", "_", base)
        return sanitized

    def _get_file_path_for_id(self, file_id: str) -> str:
        for fname in os.listdir(self.storage_dir):
            if fname.startswith(f"{file_id}_"):
                return os.path.join(self.storage_dir, fname)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Uploaded file for ID '{file_id}' not found.",
        )

    def upload_and_validate_dataset(self, file: UploadFile, adapter_type: str = "urban_road_collapse") -> DatasetUploadResponse:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Only CSV files (.csv) are accepted.",
            )

        # Read file contents and verify size limit
        content = file.file.read()
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum permitted limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

        file_id = str(uuid.uuid4())
        sanitized_name = self._sanitize_filename(file.filename)
        saved_filename = f"{file_id}_{sanitized_name}"
        saved_path = os.path.join(self.storage_dir, saved_filename)

        with open(saved_path, "wb") as f:
            f.write(content)

        # Parse CSV with pandas and run schema validation
        try:
            df = pd.read_csv(saved_path)
        except Exception as err:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse CSV file: {str(err)}",
            )

        adapter = get_adapter(adapter_type)
        report = adapter.validate(df)
        is_valid = report.is_valid
        validation_errors = [str(err) for err in report.errors]

        return DatasetUploadResponse(
            file_id=file_id,
            filename=file.filename,
            stored_path=saved_path,
            size_bytes=len(content),
            row_count=len(df),
            is_valid=is_valid,
            validation_errors=validation_errors,
            column_names=list(df.columns),
        )

    def get_dataset_preview(self, file_id: str, limit: int = 10) -> DatasetPreviewResponse:
        file_path = self._get_file_path_for_id(file_id)
        df = pd.read_csv(file_path)

        preview_rows = df.head(limit).fillna("").to_dict(orient="records")
        col_summary = {
            col: {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
            }
            for col in df.columns
        }

        filename = os.path.basename(file_path).split("_", 1)[-1]
        return DatasetPreviewResponse(
            file_id=file_id,
            filename=filename,
            total_rows=len(df),
            columns=list(df.columns),
            sample_data=preview_rows,
            column_summary=col_summary,
        )

    def integrate_dataset(self, name: str, file_id: str, adapter_type: str = "urban_road_collapse", description: str = None) -> DatasetResponse:
        file_path = self._get_file_path_for_id(file_id)
        df = pd.read_csv(file_path)

        adapter = get_adapter(adapter_type)
        report = adapter.validate(df)
        if not report.is_valid:
            err_msg = "; ".join(str(e) for e in report.errors)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot integrate invalid dataset: {err_msg}",
            )


        column_info = {
            "columns": list(df.columns),
            "feature_count": len(df.columns),
            "null_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        }

        # Check existing dataset or create new
        dataset = self.repo.get_dataset_by_name(name)
        if not dataset:
            dataset = self.repo.create_dataset(name=name, adapter_type=adapter_type, description=description)

        version = self.repo.create_dataset_version(
            dataset_id=dataset.id,
            file_path=file_path,
            row_count=len(df),
            column_info=column_info,
        )

        return self._to_dataset_response(dataset)

    def get_dataset(self, dataset_id: str) -> DatasetResponse:
        dataset = self.repo.get_dataset_by_id(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID '{dataset_id}' not found.",
            )
        return self._to_dataset_response(dataset)

    def list_datasets(self, page: int = 1, page_size: int = 20) -> DatasetListResponse:
        items, total = self.repo.list_datasets(page=page, page_size=page_size)
        resp_items = [self._to_dataset_response(d) for d in items]
        return DatasetListResponse(
            items=resp_items,
            total=total,
            page=page,
            page_size=page_size,
        )

    def _to_dataset_response(self, dataset) -> DatasetResponse:
        versions_resp = [
            DatasetVersionResponse.model_validate(v) for v in dataset.versions
        ]
        latest_ver = versions_resp[-1] if versions_resp else None
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            adapter_type=dataset.adapter_type,
            created_at=dataset.created_at,
            latest_version=latest_ver,
            versions=versions_resp,
        )
