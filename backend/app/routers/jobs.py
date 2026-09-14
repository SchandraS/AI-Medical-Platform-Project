from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.models_db import BatchJob, InferenceLog, JobStatus, Role, User
from app.schemas import BatchJobAccepted, BatchJobResults, BatchJobStatus, BatchRowResult
from app.security import get_current_user, require_clinician
from app.workers.batch import parse_csv_bytes, submit_job

router = APIRouter(prefix="/jobs", tags=["batch"])


@router.post("/upload", response_model=BatchJobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def upload_batch(
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(require_clinician),
    settings: Settings = Depends(get_settings),
) -> BatchJobAccepted:
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

    if len(df) > settings.max_batch_rows:
        raise HTTPException(
            status_code=422,
            detail=f"batch has {len(df)} rows, exceeding the max of {settings.max_batch_rows}",
        )

    job = BatchJob(user_id=user.id, filename=file.filename, status=JobStatus.queued, total_rows=len(df))
    db.add(job)
    db.commit()
    db.refresh(job)

    submit_job(job.id, raw)

    return BatchJobAccepted(job_id=job.id, status=job.status.value, total_rows=len(df))


@router.get("", response_model=list[BatchJobStatus])
def list_jobs(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[BatchJobStatus]:
    query = db.query(BatchJob).order_by(desc(BatchJob.created_at))
    if user.role != Role.ml_engineer:
        query = query.filter(BatchJob.user_id == user.id)
    jobs = query.limit(200).all()
    return [BatchJobStatus(**_job_status_dict(j)) for j in jobs]


@router.get("/{job_id}", response_model=BatchJobStatus)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> BatchJobStatus:
    job = _get_owned_job(db, job_id, user)
    return BatchJobStatus(**_job_status_dict(job))


@router.get("/{job_id}/results", response_model=BatchJobResults)
def get_job_results(
    job_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> BatchJobResults:
    job = _get_owned_job(db, job_id, user)
    logs = (
        db.query(InferenceLog)
        .filter(InferenceLog.batch_job_id == job.id)
        .order_by(InferenceLog.row_index)
        .all()
    )
    rows = [
        BatchRowResult(
            row_index=log.row_index,
            status=log.status,
            prediction=log.prediction,
            label=None,
            probability=log.probability,
            confidence=(max(log.probability, 1 - log.probability) if log.probability is not None else None),
            error=log.error_message,
        )
        for log in logs
    ]
    return BatchJobResults(job_id=job.id, status=job.status.value, rows=rows)


def _get_owned_job(db: Session, job_id: str, user: User) -> BatchJob:
    job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    if user.role != Role.ml_engineer and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="you do not have access to this job")
    return job


def _job_status_dict(job: BatchJob) -> dict:
    return {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "total_rows": job.total_rows,
        "processed_rows": job.processed_rows,
        "succeeded_rows": job.succeeded_rows,
        "failed_rows": job.failed_rows,
        "model_version_id": job.model_version_id,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }
