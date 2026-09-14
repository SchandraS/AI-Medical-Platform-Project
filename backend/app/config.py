"""Application configuration via environment variables."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://manas:manas@db:5432/manas"

    # Auth
    jwt_secret: str = "dev-secret-change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 8

    # Artifacts
    artifacts_dir: str = "artifacts"
    data_dir: str = "data"
    hf_model_repo: str = "Neperl/diabetes-health-indicators-study"
    eval_data_url: str = (
        "https://huggingface.co/datasets/jason1966/"
        "alexteboul_diabetes-health-indicators-dataset/resolve/main/"
        "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    )

    # Batch limits
    max_batch_rows: int = 10_000
    max_upload_bytes: int = 15 * 1024 * 1024  # 15 MB

    # Inference
    default_threshold: float = 0.5
    preprocessing_version: str = "v1-standard-scaler-21feat"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
