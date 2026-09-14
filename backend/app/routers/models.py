"""Model registry: list versions, evaluate a candidate, promote, rollback.

Promotion invariant: at most one ModelVersion per `name` has status
'production', enforced both at the DB level (partial unique index in
models_db.py) and here via an explicit transaction that demotes the current
production version before promoting the new one, with a PromotionEvent
audit row for every transition (who, when, why, metrics at the time).
"""
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.ml import registry as model_registry
from app.ml.errors import ModelLoadError
from app.ml.evaluate import EvalDataUnavailable, evaluate_model
from app.ml.resolve import load_for_version
from app.models_db import ModelStatus, ModelVersion, PromotionEvent, User
from app.schemas import (
    EvaluateResponse,
    ModelVersionOut,
    PromoteRequest,
    PromotionEventOut,
    RollbackRequest,
)
from app.security import get_current_user, require_ml_engineer

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ModelVersionOut]:
    versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [_to_out(v) for v in versions]


@router.get("/active", response_model=ModelVersionOut)
def active_model(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ModelVersionOut:
    version = db.query(ModelVersion).filter(ModelVersion.status == ModelStatus.production).first()
    if version is None:
        raise HTTPException(status_code=503, detail="no production model is currently deployed")
    return _to_out(version)


@router.post("/{version_id}/evaluate", response_model=EvaluateResponse)
def evaluate(
    version_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_ml_engineer),
    settings: Settings = Depends(get_settings),
) -> EvaluateResponse:
    version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if version is None:
        raise HTTPException(status_code=404, detail=f"model version '{version_id}' not found")

    data_path = Path(settings.data_dir) / "brfss2015_5050split.csv"
    try:
        model, scaler = load_for_version(version)
        metrics = evaluate_model(
            model=model, scaler=scaler, framework=version.framework.value, data_path=data_path,
            threshold=settings.default_threshold,
        )
    except EvalDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"evaluation dataset unavailable: {exc}") from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=f"could not load model for evaluation: {exc}") from exc

    version.metrics = metrics
    db.commit()

    return EvaluateResponse(model_version_id=version.id, metrics=metrics, evaluated_at=datetime.now(timezone.utc))


@router.post("/{version_id}/promote", response_model=ModelVersionOut)
def promote(
    version_id: str,
    payload: PromoteRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_ml_engineer),
) -> ModelVersionOut:
    candidate = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"model version '{version_id}' not found")
    if candidate.status == ModelStatus.production:
        raise HTTPException(status_code=409, detail="this version is already in production")
    if not candidate.metrics:
        raise HTTPException(
            status_code=422,
            detail="cannot promote a version with no recorded evaluation metrics; call /evaluate first",
        )

    # Verify the candidate actually loads before promoting it — never
    # promote a version that would immediately 503 on the next prediction.
    try:
        load_for_version(candidate)
    except ModelLoadError as exc:
        raise HTTPException(status_code=422, detail=f"candidate model fails to load, refusing to promote: {exc}") from exc

    current_production = (
        db.query(ModelVersion)
        .filter(ModelVersion.name == candidate.name, ModelVersion.status == ModelStatus.production)
        .first()
    )

    try:
        if current_production is not None:
            db.add(
                PromotionEvent(
                    model_version_id=current_production.id,
                    from_status=current_production.status.value,
                    to_status=ModelStatus.archived.value,
                    actor_user_id=user.id,
                    reason=f"superseded by promotion of {candidate.version}: {payload.reason}",
                    metrics_snapshot=current_production.metrics,
                )
            )
            current_production.status = ModelStatus.archived
            model_registry.evict(current_production.id)
            # Flush the demotion before promoting the candidate: SQLAlchemy
            # does not guarantee UPDATE order within one flush matches
            # attribute-assignment order, and the partial unique index
            # (at most one row with status='production' per name) is
            # enforced immediately, not deferred -- so promoting first can
            # transiently violate it even though the end state is valid.
            db.flush()

        from_status = candidate.status.value
        candidate.status = ModelStatus.production
        candidate.promoted_at = datetime.now(timezone.utc)
        db.add(
            PromotionEvent(
                model_version_id=candidate.id,
                from_status=from_status,
                to_status=ModelStatus.production.value,
                actor_user_id=user.id,
                reason=payload.reason,
                metrics_snapshot=candidate.metrics,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(candidate)
    return _to_out(candidate)


@router.post("/rollback", response_model=ModelVersionOut)
def rollback(
    payload: RollbackRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_ml_engineer),
) -> ModelVersionOut:
    current_production = db.query(ModelVersion).filter(ModelVersion.status == ModelStatus.production).first()
    if current_production is None:
        raise HTTPException(status_code=409, detail="no production model to roll back from")

    if payload.target_model_version_id:
        target = db.query(ModelVersion).filter(ModelVersion.id == payload.target_model_version_id).first()
        if target is None:
            raise HTTPException(status_code=404, detail=f"target version '{payload.target_model_version_id}' not found")
        if target.id == current_production.id:
            raise HTTPException(status_code=409, detail="target version is already in production")
    else:
        target = (
            db.query(ModelVersion)
            .filter(
                ModelVersion.name == current_production.name,
                ModelVersion.status == ModelStatus.archived,
                ModelVersion.id != current_production.id,
            )
            .order_by(ModelVersion.promoted_at.desc().nullslast(), ModelVersion.created_at.desc())
            .first()
        )
        if target is None:
            raise HTTPException(status_code=409, detail="no previously archived production version to roll back to")

    try:
        load_for_version(target)
    except ModelLoadError as exc:
        raise HTTPException(status_code=422, detail=f"rollback target fails to load, refusing to roll back: {exc}") from exc

    try:
        db.add(
            PromotionEvent(
                model_version_id=current_production.id,
                from_status=current_production.status.value,
                to_status=ModelStatus.archived.value,
                actor_user_id=user.id,
                reason=f"rolled back: {payload.reason}",
                metrics_snapshot=current_production.metrics,
            )
        )
        current_production.status = ModelStatus.archived
        model_registry.evict(current_production.id)
        db.flush()  # see the matching comment in promote() above

        from_status = target.status.value
        target.status = ModelStatus.production
        target.promoted_at = datetime.now(timezone.utc)
        db.add(
            PromotionEvent(
                model_version_id=target.id,
                from_status=from_status,
                to_status=ModelStatus.production.value,
                actor_user_id=user.id,
                reason=f"rollback target: {payload.reason}",
                metrics_snapshot=target.metrics,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(target)
    return _to_out(target)


@router.get("/promotions/history", response_model=list[PromotionEventOut])
def promotion_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[PromotionEventOut]:
    events = db.query(PromotionEvent).order_by(PromotionEvent.created_at.desc()).limit(200).all()
    return [
        PromotionEventOut(
            id=e.id, model_version_id=e.model_version_id, from_status=e.from_status, to_status=e.to_status,
            actor_user_id=e.actor_user_id, reason=e.reason, metrics_snapshot=e.metrics_snapshot,
            created_at=e.created_at,
        )
        for e in events
    ]


def _to_out(v: ModelVersion) -> ModelVersionOut:
    return ModelVersionOut(
        id=v.id, name=v.name, version=v.version, framework=v.framework.value, status=v.status.value,
        metrics=v.metrics, artifact_sha256=v.artifact_sha256, preprocessing_version=v.preprocessing_version,
        created_at=v.created_at, promoted_at=v.promoted_at,
    )
