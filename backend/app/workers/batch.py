"""In-process async batch worker.

A DB-backed FIFO queue (BatchJob rows with status='queued') means the queue
survives a process restart — a queued job just gets picked up whenever a
worker loop next polls, rather than being lost. This is intentionally simple
for a take-home deployment (single API process); a multi-instance deployment
would swap this for Celery/RQ + Redis or a cloud queue without changing the
job/row schema, noted in REPORT.md.

Row failures are captured individually and do NOT abort the job — a batch
with 3 bad rows out of 500 still scores the other 497 and reports a
'partial' terminal status with per-row errors.
"""
import asyncio
import io
import logging
import time
from datetime import datetime, timezone

import pandas as pd
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.config import get_settings
from app import db as db_module
from app.ml.errors import InferenceError, ModelLoadError, NoProductionModelError
from app.ml.predictor import canonical_input_hash, predict_one
from app.ml.resolve import get_production_version, load_for_version
from app.models_db import BatchJob, InferenceLog, JobStatus
from app.schemas import PatientRecord

logger = logging.getLogger("manas.batch_worker")

# The queue is created lazily in start_worker(), not at module import time:
# an asyncio.Queue binds to whichever event loop is running when it's first
# used, and a module-level instance created at import would bind to the
# *first* loop ever seen -- breaking every subsequent loop (e.g. each
# pytest TestClient spins up its own loop; a container restart likewise
# starts a fresh loop). Lazily creating it inside start_worker(), which
# always runs inside the loop that will actually consume it, avoids that.
_queue: "asyncio.Queue[str] | None" = None
_worker_task: asyncio.Task | None = None
# Uploaded CSV bytes staged between "accept the upload" (router) and "the
# worker loop picks it up". A job_id is only ever enqueued once its bytes are
# staged here, so this never races the consumer.
_pending_payloads: dict[str, bytes] = {}


def enqueue(job_id: str) -> None:
    if _queue is None:
        raise RuntimeError("batch worker queue not initialized; call start_worker() first")
    _queue.put_nowait(job_id)


def parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Raises ValueError with a clear message on malformed CSV — caught by
    the router and turned into a 422, never a 500."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {exc}") from exc
    if not text.strip():
        raise ValueError("uploaded file is empty")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse CSV: {exc}") from exc
    if df.empty:
        raise ValueError("CSV has a header but no data rows")
    return df


def _validate_row(row: dict) -> tuple[dict | None, list[str]]:
    try:
        record = PatientRecord(**row)
        return record.to_feature_dict(), []
    except ValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return None, errors


def _process_job(db: Session, job: BatchJob, df: pd.DataFrame) -> None:
    settings = get_settings()
    job.status = JobStatus.running
    job.started_at = datetime.now(timezone.utc)
    db.commit()

    try:
        version = get_production_version(db)
        model, scaler = load_for_version(version)
    except (NoProductionModelError, ModelLoadError) as exc:
        job.status = JobStatus.failed
        job.error_summary = f"no usable production model: {exc}"
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        return

    job.model_version_id = version.id
    job.total_rows = len(df)
    db.commit()

    succeeded = 0
    failed = 0
    raw_rows = df.to_dict(orient="records")

    for idx, raw_row in enumerate(raw_rows):
        start = time.perf_counter()
        try:
            clean_row = {k: v for k, v in raw_row.items() if not (isinstance(v, float) and pd.isna(v))}
            record, errors = _validate_row(clean_row)

            if record is None:
                failed += 1
                db.add(
                    InferenceLog(
                        user_id=job.user_id,
                        batch_job_id=job.id,
                        row_index=idx,
                        model_version_id=version.id,
                        input_hash=canonical_input_hash(clean_row),
                        input_echo=clean_row,
                        status="error",
                        error_code="ValidationError",
                        error_message="; ".join(errors),
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
                )
            else:
                try:
                    result = predict_one(
                        record, model=model, scaler=scaler,
                        framework=version.framework.value, threshold=settings.default_threshold,
                    )
                    succeeded += 1
                    db.add(
                        InferenceLog(
                            user_id=job.user_id,
                            batch_job_id=job.id,
                            row_index=idx,
                            model_version_id=version.id,
                            input_hash=canonical_input_hash(record),
                            input_echo=record,
                            prediction=result.prediction,
                            probability=result.probability,
                            threshold=result.threshold,
                            status="ok",
                            latency_ms=(time.perf_counter() - start) * 1000,
                        )
                    )
                except InferenceError as exc:
                    failed += 1
                    db.add(
                        InferenceLog(
                            user_id=job.user_id,
                            batch_job_id=job.id,
                            row_index=idx,
                            model_version_id=version.id,
                            input_hash=canonical_input_hash(record),
                            input_echo=record,
                            status="error",
                            error_code="InferenceError",
                            error_message=str(exc),
                            latency_ms=(time.perf_counter() - start) * 1000,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            # Last-resort guard: an unexpected error on one row (e.g. a
            # non-serializable value, an encoding quirk) must never abort
            # the whole job -- log it as a failed row and keep going.
            failed += 1
            logger.exception("batch worker: unexpected error on row %d of job %s", idx, job.id)
            db.add(
                InferenceLog(
                    user_id=job.user_id,
                    batch_job_id=job.id,
                    row_index=idx,
                    model_version_id=version.id,
                    input_hash="unhashable",
                    input_echo={},
                    status="error",
                    error_code="UnexpectedRowError",
                    error_message=str(exc),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            )

        job.processed_rows = idx + 1
        job.succeeded_rows = succeeded
        job.failed_rows = failed
        if (idx + 1) % 50 == 0:
            db.commit()

    job.status = JobStatus.succeeded if failed == 0 else (JobStatus.partial if succeeded > 0 else JobStatus.failed)
    job.finished_at = datetime.now(timezone.utc)
    db.commit()


async def _worker_loop() -> None:
    while True:
        job_id = await _queue.get()
        try:
            db = db_module.new_session()
            try:
                job = db.query(BatchJob).filter(BatchJob.id == job_id).first()
                if job is None:
                    logger.warning("batch worker: job %s not found, skipping", job_id)
                    continue
                raw = _pending_payloads.pop(job_id, None)
                if raw is None:
                    job.status = JobStatus.failed
                    job.error_summary = "internal error: uploaded payload lost before processing"
                    db.commit()
                    continue
                df = parse_csv_bytes(raw)
                _process_job(db, job, df)
            finally:
                db.close()
        except Exception:  # noqa: BLE001
            logger.exception("batch worker: unhandled error processing job %s", job_id)
        finally:
            _queue.task_done()


def submit_job(job_id: str, raw_csv: bytes) -> None:
    _pending_payloads[job_id] = raw_csv
    enqueue(job_id)


def start_worker() -> None:
    global _worker_task, _queue
    _queue = asyncio.Queue()  # bind to the loop that's running right now
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())


async def stop_worker() -> None:
    global _worker_task, _queue
    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
    _queue = None
