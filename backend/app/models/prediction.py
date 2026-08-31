import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, JSON, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.core.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class Prediction(Base):
    """
    Stores historical record of each risk prediction request:
    - Raw input features (34 numerical values)
    - Predicted risk category ('Low', 'Moderate', 'High', 'Critical')
    - Prediction confidence (highest class probability)
    - Full probability distribution across all 4 classes
    - ModelVersion foreign key reference
    - Geographic coordinates if provided (for map visualization)
    - Timestamp
    """
    __tablename__ = "prediction"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(String(36), ForeignKey("model_version.id", ondelete="SET NULL"), nullable=True, index=True)
    features = Column(JSON, nullable=False)
    predicted_class = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)

    model_version = relationship("ModelVersion")
