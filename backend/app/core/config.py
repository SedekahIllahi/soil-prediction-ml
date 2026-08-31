from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    PROJECT_NAME: str = "ML-Based Ground/Soil Risk Prediction and Monitoring System"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = "development"
    
    # ML settings
    RANDOM_SEED: int = 42
    ENABLE_SVM: bool = True
    MAX_UPLOAD_SIZE_MB: int = 50
    DATASET_PATH: str = "storage/datasets/urban_road_collapse_risk_dataset.csv"
    
    # CORS origins
    CORS_ORIGINS: str | List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Database settings
    DATABASE_URL: str = "sqlite:///./soil_ml.db"  # Fallback SQLite for local testing without Postgres Docker

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

settings = Settings()
