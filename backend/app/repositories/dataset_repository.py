from typing import List, Optional, Tuple
from sqlalchemy.orm import Session, joinedload
from app.models.dataset import Dataset, DatasetVersion

class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_dataset(self, name: str, adapter_type: str, description: Optional[str] = None) -> Dataset:
        dataset = Dataset(
            name=name,
            adapter_type=adapter_type,
            description=description,
        )
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def get_dataset_by_id(self, dataset_id: str) -> Optional[Dataset]:
        return (
            self.db.query(Dataset)
            .options(joinedload(Dataset.versions))
            .filter(Dataset.id == dataset_id)
            .first()
        )

    def get_dataset_by_name(self, name: str) -> Optional[Dataset]:
        return (
            self.db.query(Dataset)
            .options(joinedload(Dataset.versions))
            .filter(Dataset.name == name)
            .first()
        )

    def list_datasets(self, page: int = 1, page_size: int = 20) -> Tuple[List[Dataset], int]:
        query = self.db.query(Dataset).options(joinedload(Dataset.versions))
        total = query.count()
        items = query.order_by(Dataset.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
        return items, total

    def create_dataset_version(
        self,
        dataset_id: str,
        file_path: str,
        row_count: int,
        column_info: dict,
    ) -> DatasetVersion:
        # Determine current highest version for this dataset
        latest_version_number = (
            self.db.query(DatasetVersion.version)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version.desc())
            .first()
        )
        next_version = (latest_version_number[0] + 1) if latest_version_number else 1

        version = DatasetVersion(
            dataset_id=dataset_id,
            version=next_version,
            file_path=file_path,
            row_count=row_count,
            column_info=column_info,
        )
        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)
        return version

    def get_latest_version(self, dataset_id: str) -> Optional[DatasetVersion]:
        return (
            self.db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version.desc())
            .first()
        )
