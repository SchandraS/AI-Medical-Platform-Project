from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db import get_db
from app.models_db import InferenceLog, Role, User
from app.schemas import InferenceLogOut
from app.security import get_current_user

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/inferences", response_model=list[InferenceLogOut])
def list_inference_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[InferenceLogOut]:
    query = db.query(InferenceLog).order_by(desc(InferenceLog.created_at))
    if user.role != Role.ml_engineer:
        query = query.filter(InferenceLog.user_id == user.id)
    logs = query.offset(offset).limit(limit).all()
    return [
        InferenceLogOut(
            id=l.id, user_id=l.user_id, batch_job_id=l.batch_job_id, row_index=l.row_index,
            model_version_id=l.model_version_id, input_hash=l.input_hash, prediction=l.prediction,
            probability=l.probability, status=l.status, error_code=l.error_code,
            latency_ms=l.latency_ms, created_at=l.created_at,
        )
        for l in logs
    ]
