"""Batch inference: valid CSV, malformed CSV, mixed-quality rows, oversized."""
import asyncio
import io
import time

import pytest

from app.workers.batch import _process_job
from tests.conftest import VALID_RECORD, auth_headers


def _csv_bytes(rows: list[dict]) -> bytes:
    import csv

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def test_batch_upload_valid_csv_returns_202(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    csv_bytes = _csv_bytes([VALID_RECORD, VALID_RECORD])
    resp = client.post(
        "/jobs/upload", headers=headers, files={"file": ("sample.csv", csv_bytes, "text/csv")}
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    assert body["total_rows"] == 2
    assert body["job_id"]


def test_batch_job_processes_synchronously_via_worker_internals(
    client, seeded_users, production_model_version, db_session
):
    """Exercises _process_job directly (bypassing the async queue timing) to
    deterministically assert on terminal state without sleeping in a test."""
    import pandas as pd
    from app.models_db import BatchJob, JobStatus

    job = BatchJob(user_id=seeded_users["clinician"].id, filename="x.csv", status=JobStatus.queued, total_rows=2)
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    df = pd.DataFrame([VALID_RECORD, VALID_RECORD])
    _process_job(db_session, job, df)

    db_session.refresh(job)
    assert job.status == JobStatus.succeeded
    assert job.succeeded_rows == 2
    assert job.failed_rows == 0
    assert job.processed_rows == 2


def test_batch_job_partial_success_on_mixed_rows(client, seeded_users, production_model_version, db_session):
    import pandas as pd
    from app.models_db import BatchJob, JobStatus

    bad_row = {**VALID_RECORD, "BMI": 999}  # out of range -> validation failure
    job = BatchJob(user_id=seeded_users["clinician"].id, filename="x.csv", status=JobStatus.queued, total_rows=2)
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    df = pd.DataFrame([VALID_RECORD, bad_row])
    _process_job(db_session, job, df)

    db_session.refresh(job)
    assert job.status == JobStatus.partial
    assert job.succeeded_rows == 1
    assert job.failed_rows == 1


def test_batch_job_row_missing_all_fields_does_not_crash_worker(
    client, seeded_users, production_model_version, db_session
):
    """A row with completely unrelated/garbage column names (e.g. from a
    CSV with the wrong header entirely) must be recorded as a failed row,
    not crash the worker loop and leave the job stuck in 'running' forever."""
    import pandas as pd
    from app.models_db import BatchJob, JobStatus

    garbage_row = {"foo": 1, "bar": 2}
    job = BatchJob(user_id=seeded_users["clinician"].id, filename="x.csv", status=JobStatus.queued, total_rows=1)
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    df = pd.DataFrame([garbage_row])
    _process_job(db_session, job, df)

    db_session.refresh(job)
    assert job.status == JobStatus.failed
    assert job.failed_rows == 1


def test_batch_job_all_rows_invalid_yields_failed_status(client, seeded_users, production_model_version, db_session):
    import pandas as pd
    from app.models_db import BatchJob, JobStatus

    bad_row = {**VALID_RECORD, "BMI": 999}
    job = BatchJob(user_id=seeded_users["clinician"].id, filename="x.csv", status=JobStatus.queued, total_rows=1)
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    df = pd.DataFrame([bad_row])
    _process_job(db_session, job, df)

    db_session.refresh(job)
    assert job.status == JobStatus.failed
    assert job.succeeded_rows == 0
    assert job.failed_rows == 1


def test_batch_upload_rejects_non_csv_extension(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(
        "/jobs/upload", headers=headers, files={"file": ("sample.txt", b"not a csv", "text/plain")}
    )
    assert resp.status_code == 422


def test_batch_upload_rejects_empty_file(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post("/jobs/upload", headers=headers, files={"file": ("sample.csv", b"", "text/csv")})
    assert resp.status_code == 422


def test_batch_upload_rejects_unparseable_csv(client, seeded_users, production_model_version):
    """Bytes that pandas' CSV parser itself cannot tokenize (inconsistent
    column counts per row) must be rejected at upload time with a 422."""
    headers = auth_headers(client, "clinician", "clinician123")
    unparseable = b'"unterminated quote,and,mismatched\nfield,counts,across\nrows,with,extra,columns,here'
    resp = client.post(
        "/jobs/upload", headers=headers, files={"file": ("sample.csv", unparseable, "text/csv")}
    )
    assert resp.status_code == 422


def test_batch_upload_accepts_csv_with_invalid_data_rows(client, seeded_users, production_model_version):
    """A CSV that parses fine but contains semantically invalid rows (wrong
    values/types) is accepted at upload (202) -- validation happens per-row
    during processing, not at upload time, so one bad row can't block the
    whole batch. See test_batch_job_partial_success_on_mixed_rows for the
    per-row outcome."""
    headers = auth_headers(client, "clinician", "clinician123")
    bad_row = {**VALID_RECORD, "BMI": "not-a-number"}
    csv_bytes = _csv_bytes([VALID_RECORD, bad_row])
    resp = client.post(
        "/jobs/upload", headers=headers, files={"file": ("sample.csv", csv_bytes, "text/csv")}
    )
    assert resp.status_code == 202


def test_batch_upload_rejects_header_only_csv(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    header_only = (",".join(VALID_RECORD.keys()) + "\n").encode()
    resp = client.post(
        "/jobs/upload", headers=headers, files={"file": ("sample.csv", header_only, "text/csv")}
    )
    assert resp.status_code == 422


def test_job_status_and_results_ownership_enforced(client, seeded_users, production_model_version, db_session):
    from app.models_db import BatchJob, JobStatus

    job = BatchJob(user_id=seeded_users["clinician"].id, filename="x.csv", status=JobStatus.succeeded)
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    viewer_headers = auth_headers(client, "viewer", "viewer123")
    resp = client.get(f"/jobs/{job.id}", headers=viewer_headers)
    assert resp.status_code == 403

    mlops_headers = auth_headers(client, "mlops", "mlops123")
    resp = client.get(f"/jobs/{job.id}", headers=mlops_headers)
    assert resp.status_code == 200  # ml_engineer can see all jobs


def test_job_not_found_returns_404(client, seeded_users):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.get("/jobs/does-not-exist", headers=headers)
    assert resp.status_code == 404
