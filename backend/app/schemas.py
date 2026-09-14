"""Pydantic v2 request/response contracts.

Validation policy (per assignment requirement "no silent coercion of garbage
data"): every model uses extra="forbid" so unexpected fields are a 422, not
silently dropped; every numeric field is bounded to its real observed range;
booleans-as-0/1 are validated as strict int in {0,1}, not coerced from
arbitrary truthy values or strings.
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ml.schema import FEATURE_ORDER


class PatientRecord(BaseModel):
    """A single patient/sample record: exactly the 21 clinical indicators,
    no more, no less. extra='forbid' rejects unexpected fields outright."""

    model_config = ConfigDict(extra="forbid", strict=False)

    HighBP: int = Field(..., ge=0, le=1)
    HighChol: int = Field(..., ge=0, le=1)
    CholCheck: int = Field(..., ge=0, le=1)
    BMI: float = Field(..., ge=10, le=100)
    Smoker: int = Field(..., ge=0, le=1)
    Stroke: int = Field(..., ge=0, le=1)
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1)
    PhysActivity: int = Field(..., ge=0, le=1)
    Fruits: int = Field(..., ge=0, le=1)
    Veggies: int = Field(..., ge=0, le=1)
    HvyAlcoholConsump: int = Field(..., ge=0, le=1)
    AnyHealthcare: int = Field(..., ge=0, le=1)
    NoDocbcCost: int = Field(..., ge=0, le=1)
    GenHlth: int = Field(..., ge=1, le=5)
    MentHlth: float = Field(..., ge=0, le=30)
    PhysHlth: float = Field(..., ge=0, le=30)
    DiffWalk: int = Field(..., ge=0, le=1)
    Sex: int = Field(..., ge=0, le=1)
    Age: int = Field(..., ge=1, le=13)
    Education: int = Field(..., ge=1, le=6)
    Income: int = Field(..., ge=1, le=8)

    @field_validator(
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
        "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "DiffWalk", "Sex",
        mode="before",
    )
    @classmethod
    def _binary_no_coercion(cls, v: Any) -> Any:
        """Reject bools, strings like 'true'/'yes', and floats like 0.5 —
        only literal 0/1 (int or the int-valued float 0.0/1.0) are accepted."""
        if isinstance(v, bool):
            raise ValueError("must be integer 0 or 1, not a boolean")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("must be integer 0 or 1")
        if isinstance(v, str):
            raise ValueError("must be integer 0 or 1, not a string")
        return v

    def to_feature_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in FEATURE_ORDER}


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record: PatientRecord
    model_version_id: str | None = Field(
        default=None, description="Specific model version to use; defaults to the active production model."
    )


class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    confidence: float
    threshold: float
    model_version_id: str
    model_name: str
    model_version: str
    preprocessing_version: str
    input_hash: str
    input_echo: dict[str, float]
    timestamp: datetime
    latency_ms: float
    log_id: str


class BatchJobAccepted(BaseModel):
    job_id: str
    status: str
    total_rows: int
    message: str = "batch job accepted; poll GET /jobs/{job_id} for status"


class RowError(BaseModel):
    row_index: int
    errors: list[str]


class BatchJobStatus(BaseModel):
    job_id: str
    filename: str
    status: str
    total_rows: int
    processed_rows: int
    succeeded_rows: int
    failed_rows: int
    model_version_id: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class BatchRowResult(BaseModel):
    row_index: int
    status: Literal["ok", "error"]
    prediction: int | None = None
    label: str | None = None
    probability: float | None = None
    confidence: float | None = None
    error: str | None = None


class BatchJobResults(BaseModel):
    job_id: str
    status: str
    rows: list[BatchRowResult]


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class ModelVersionOut(BaseModel):
    id: str
    name: str
    version: str
    framework: str
    status: str
    metrics: dict | None
    artifact_sha256: str
    preprocessing_version: str
    created_at: datetime
    promoted_at: datetime | None


class EvaluateResponse(BaseModel):
    model_version_id: str
    metrics: dict
    evaluated_at: datetime


class PromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(..., min_length=1, max_length=1000)


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(..., min_length=1, max_length=1000)
    target_model_version_id: str | None = Field(
        default=None,
        description="Explicit version to roll back to; defaults to the most recently archived production version.",
    )


class PromotionEventOut(BaseModel):
    id: str
    model_version_id: str
    from_status: str | None
    to_status: str
    actor_user_id: str | None
    reason: str | None
    metrics_snapshot: dict | None
    created_at: datetime


class InferenceLogOut(BaseModel):
    id: str
    user_id: str | None
    batch_job_id: str | None
    row_index: int | None
    model_version_id: str | None
    input_hash: str
    prediction: int | None
    probability: float | None
    status: str
    error_code: str | None
    latency_ms: float | None
    created_at: datetime


class ErrorResponse(BaseModel):
    """Standard error envelope for all non-2xx responses."""
    detail: str
    error_code: str | None = None
    request_id: str | None = None
