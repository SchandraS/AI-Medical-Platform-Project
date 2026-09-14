import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.ml.errors import InferenceError, ModelLoadError, NoProductionModelError
from app.ml.predictor import canonical_input_hash, predict_one
from app.ml.resolve import get_version_or_default, load_for_version
from app.models_db import InferenceLog, User
from app.schemas import PatientRecord, PredictRequest, PredictResponse
from app.security import require_clinician
from app.workers.batch import parse_csv_bytes

router = APIRouter(tags=["predict"])


def _run_predict(
    record: dict,
    *,
    model_version_id: str | None,
    db: Session,
    user: User,
    settings: Settings,
) -> PredictResponse:
    """Shared by both /predict (JSON) and /predict/csv (single-row CSV):
    resolves the model version, runs inference, and persists the log —
    exactly the same path and the same failure handling either way, so a
    CSV-submitted record is traced and reproducible identically to a
    JSON-submitted one."""
    start = time.perf_counter()
    input_hash = canonical_input_hash(record)
    version = None

    try:
        version = get_version_or_default(db, model_version_id)
        model, scaler = load_for_version(version)
        result = predict_one(
            record,
            model=model,
            scaler=scaler,
            framework=version.framework.value,
            threshold=settings.default_threshold,
        )
    except (NoProductionModelError, ModelLoadError, InferenceError) as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        db.add(
            InferenceLog(
                user_id=user.id,
                model_version_id=version.id if version else None,
                input_hash=input_hash,
                input_echo=record,
                latency_ms=latency_ms,
                status="error",
                error_code=type(exc).__name__,
                error_message=str(exc),
            )
        )
        db.commit()
        raise

    latency_ms = (time.perf_counter() - start) * 1000

    log = InferenceLog(
        user_id=user.id,
        model_version_id=version.id,
        input_hash=input_hash,
        input_echo=record,
        prediction=result.prediction,
        probability=result.probability,
        threshold=result.threshold,
        latency_ms=latency_ms,
        status="ok",
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return PredictResponse(
        prediction=result.prediction,
        label=result.label,
        probability=result.probability,
        confidence=result.confidence,
        threshold=result.threshold,
        model_version_id=version.id,
        model_name=version.name,
        model_version=version.version,
        preprocessing_version=version.preprocessing_version,
        input_hash=input_hash,
        input_echo=record,
        timestamp=datetime.now(timezone.utc),
        latency_ms=latency_ms,
        log_id=log.id,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    record = payload.record.to_feature_dict()
    return _run_predict(
        record, model_version_id=payload.model_version_id, db=db, user=user, settings=settings
    )


@router.post("/predict/csv", response_model=PredictResponse)
async def predict_csv(
    file: UploadFile,
    model_version_id: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    """Single-record prediction from a CSV upload: a header row matching the
    21 feature names plus exactly one data row. Uses the same validation
    (PatientRecord, no silent coercion) and the same inference/logging path
    as the JSON /predict endpoint — only the input transport differs."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="only .csv files are accepted")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=422, detail="uploaded file is empty")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file exceeds max upload size of {settings.max_upload_bytes} bytes",
        )

    try:
        df = parse_csv_bytes(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if len(df) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"/predict/csv expects exactly one data row, got {len(df)}; "
            "use /jobs/upload for multiple rows",
        )

    row = df.iloc[0].to_dict()
    clean_row = {k: v for k, v in row.items() if v == v}  # drop NaN (missing cell)
    try:
        record_model = PatientRecord(**clean_row)
    except ValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise HTTPException(status_code=422, detail="; ".join(errors)) from exc

    record = record_model.to_feature_dict()
    return _run_predict(
        record, model_version_id=model_version_id, db=db, user=user, settings=settings
    )
