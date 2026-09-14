import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.ml.errors import InferenceError, ModelLoadError, NoProductionModelError
from app.ml.predictor import canonical_input_hash, predict_one
from app.ml.resolve import get_version_or_default, load_for_version
from app.models_db import InferenceLog, User
from app.schemas import PredictRequest, PredictResponse
from app.security import require_clinician

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    start = time.perf_counter()
    record = payload.record.to_feature_dict()
    input_hash = canonical_input_hash(record)
    version = None

    try:
        version = get_version_or_default(db, payload.model_version_id)
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
