import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    adapter_type = Column(String(100), nullable=False, default="urban_road_collapse")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan", order_by="DatasetVersion.version")

class DatasetVersion(Base):
    __tablename__ = "dataset_version"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey("dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=False)
    column_info = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    dataset = relationship("Dataset", back_populates="versions")
