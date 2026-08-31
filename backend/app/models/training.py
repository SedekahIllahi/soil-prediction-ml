import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, JSON, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from app.core.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class TrainingRun(Base):
    __tablename__ = "training_run"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_version_id = Column(String(36), ForeignKey("dataset_version.id"), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    config = Column(JSON, nullable=True)
    comparison_results = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    model_versions = relationship("ModelVersion", back_populates="training_run", cascade="all, delete-orphan")
    dataset_version = relationship("DatasetVersion")

class ModelVersion(Base):
    __tablename__ = "model_version"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    training_run_id = Column(String(36), ForeignKey("training_run.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_version_id = Column(String(36), ForeignKey("dataset_version.id"), nullable=False)
    version = Column(Integer, nullable=False)
    algorithm = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="candidate")  # candidate, evaluated, promoted, active, retired
    metrics = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    artifact_path = Column(String(512), nullable=True)
    preprocessor_path = Column(String(512), nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    training_run = relationship("TrainingRun", back_populates="model_versions")
