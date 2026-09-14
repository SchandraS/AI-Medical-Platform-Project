from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.models_db import ModelStatus, ModelVersion

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness: process is up. Does not touch the DB or models."""
    return {"status": "ok"}


@router.get("/ready")
def ready(db: Session = Depends(get_db)) -> dict:
    """Readiness: DB reachable AND a production model row exists. Does not
    force-load the model artifact itself (that's checked lazily per-request
    and reported as 503 there) to keep this endpoint cheap for orchestrators."""
    checks = {"database": False, "production_model_registered": False}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:  # noqa: BLE001
        pass

    if checks["database"]:
        version = db.query(ModelVersion).filter(ModelVersion.status == ModelStatus.production).first()
        checks["production_model_registered"] = version is not None

    overall = all(checks.values())
    return {"status": "ready" if overall else "not_ready", "checks": checks}
