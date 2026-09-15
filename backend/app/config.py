"""Application configuration via environment variables."""
import json
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://manas:manas@db:5432/manas"

    @field_validator("database_url")
    @classmethod
    def _ensure_psycopg_driver(cls, v: str) -> str:
        """Managed Postgres providers (Render, Heroku, etc.) typically hand
        out a bare postgresql:// or postgres:// URL. SQLAlchemy would then
        default to the psycopg2 driver, which isn't installed here (we pin
        psycopg[binary]==3.2.3, i.e. psycopg 3) -- rewriting the scheme to
        postgresql+psycopg:// makes any such URL work without the deploying
        platform needing to know our driver choice."""
        if v.startswith("postgres://"):
            v = "postgresql://" + v[len("postgres://"):]
        if v.startswith("postgresql://"):
            v = "postgresql+psycopg://" + v[len("postgresql://"):]
        return v

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

    # CORS. Declared as `str | list[str]` rather than plain `list[str]`:
    # pydantic-settings JSON-decodes any env var for a list-typed field
    # *before* field validators run, so a bare hostname (not a JSON array)
    # would blow up at that decode step, never reaching a validator at all.
    # Accepting `str` too lets a plain string skip that JSON-decode path and
    # reach _normalize_cors_origins below, which does the parsing itself.
    cors_origins: str | list[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _normalize_cors_origins(cls, v):
        """Accepts either the usual JSON array of full URLs
        (`'["https://x.example.com"]'`) or a single bare hostname/URL string
        -- the shape Render's Blueprint `fromService`/`property: host`
        cross-service reference produces (e.g. `manas-frontend.onrender.com`,
        no scheme, no brackets/quotes). A JSON array is parsed and returned
        as-is; a bare value is wrapped into a single-item https:// URL list."""
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            if not stripped.startswith(("http://", "https://")):
                stripped = f"https://{stripped}"
            return [stripped]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
