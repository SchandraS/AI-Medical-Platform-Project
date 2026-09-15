# REPORT

## 1. Approach

### 1.1 Understanding the model artifacts first

Before writing any service code, I downloaded and inspected the actual artifacts from
[`Neperl/diabetes-health-indicators-study`](https://huggingface.co/Neperl/diabetes-health-indicators-study)
rather than guessing their contract from the model card prose:

- `diabetes_random_forest_tuned.joblib` — a `RandomForestClassifier` (scikit-learn **1.9.0**).
- `diabetes_mlp_tuned.keras` — a Keras **3.15.0** `Sequential` model: `Dense(128, relu) →
  Dropout(0.4) → Dense(1, sigmoid)`, input shape `(None, 21)`, only 2,945 parameters.
- `scaler.joblib` — a fitted `StandardScaler` **required by both models**.

Critically, the scaler pickle carries `feature_names_in_`, which I extracted directly (via
pickle-stream inspection, without needing scikit-learn installed) as the authoritative
21-feature order:

```
HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity,
Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth,
PhysHlth, DiffWalk, Sex, Age, Education, Income
```

The RandomForest itself has **no** `feature_names_in_` — it was fit on the scaler's raw
NumPy output — so passing a differently-ordered array would silently produce a
plausible-looking but wrong prediction with no error at all. This is the single
highest-risk detail in the whole system, so:

- `app/ml/schema.py` defines `FEATURE_ORDER` as the one source of truth.
- `app/ml/registry.py`'s scaler loader **asserts** `scaler.feature_names_in_ ==
  FEATURE_ORDER` at load time and refuses to serve (`ModelLoadError` → 503) on mismatch.
- `app/ml/predictor.py` is the **only** place that builds the model input array, always
  reading fields in `FEATURE_ORDER`.
- `tests/test_schema_contract.py` and `tests/test_real_artifacts.py` guard against this
  ever drifting, checking the *real* artifact's `feature_names_in_` byte-for-byte.

I also verified a public, no-auth mirror of the Kaggle BRFSS2015 5050-split CSV
(`huggingface.co/datasets/jason1966/alexteboul_diabetes-health-indicators-dataset`) has the
exact same column order and 70,692 rows with a perfect 50/50 class balance — this is fetched
automatically at container boot and used as the fixed evaluation/holdout set for the
promotion gate and as the drift-detection baseline.

### 1.2 Version pinning

Both joblib files embed `_sklearn_version: 1.9.0`. TensorFlow 2.21 (needed for Keras 3.15)
ships wheels only up to Python 3.13, so the backend Docker image is pinned to
**python:3.12-slim** and `requirements.txt` pins `scikit-learn==1.9.0`, `keras==3.15.1`,
`tensorflow-cpu==2.21.0`. Loading the RF under a *different* sklearn version reproduces
scikit-learn's own `InconsistentVersionWarning` (I hit this in local venv testing, since the
host machine runs Python 3.14 which has no matching sklearn/TF wheels at all) — pinning
avoids silent numerical drift from the model authors' training environment.

### 1.3 Layered architecture

```
routers/  (HTTP layer: auth, validation via Pydantic, RBAC, request/response shaping)
   ↓
ml/       (framework-agnostic: registry/loading, predictor, evaluate, drift)
   ↓
models_db.py + Alembic  (persistence: users, model_versions, promotion_events,
                          batch_jobs, inference_logs, request_logs)
```

Every ML failure mode maps to a typed exception (`ModelLoadError`, `InferenceError`,
`NoProductionModelError`) caught by a global FastAPI exception handler in `app/main.py` and
turned into a clean, logged HTTP response — there is no path from a model/scaler failure to
an unhandled 500.

## 2. Example requests and responses

Generated against the **real** RandomForest + scaler artifacts (not synthetic stubs).

### 2.1 Login

```http
POST /auth/login
{"username": "clinician", "password": "clinician123"}
```
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "role": "clinician",
  "username": "clinician"
}
```

### 2.2 Single prediction

```http
POST /predict
Authorization: Bearer <token>
{
  "record": {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32.0, "Smoker": 1, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 4,
    "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1, "Sex": 0, "Age": 10, "Education": 4,
    "Income": 3
  }
}
```

```json
{
  "prediction": 1,
  "label": "diabetes_or_prediabetes",
  "probability": 0.8574557155730359,
  "confidence": 0.8574557155730359,
  "threshold": 0.5,
  "model_version_id": "14f2519a-5215-4772-9128-b38192fe69fe",
  "model_name": "diabetes-risk",
  "model_version": "rf-v1",
  "preprocessing_version": "v1-standard-scaler-21feat",
  "input_hash": "b6decfe136ea16c16b1ebdef80280a64cd9102e1480e1dbb2e35c8d258ec6210",
  "input_echo": { "...": "the 21 fields, echoed back as floats" },
  "timestamp": "2026-09-14T14:38:19.630869Z",
  "latency_ms": 318.98,
  "log_id": "72f13cda-8b66-4600-8b03-b25fce0ca519"
}
```

### 2.2b Single prediction via CSV upload

The same single-record path is also reachable via a one-row CSV upload
(`POST /predict/csv`, `multipart/form-data`), for the "single sample as CSV"
requirement — it shares `PatientRecord` validation and the inference/logging
path with the JSON endpoint above, so the two are provably equivalent (see
`test_predict_csv_matches_json_predict_for_same_record`):

```http
POST /predict/csv
Authorization: Bearer <token>
Content-Type: multipart/form-data; boundary=...

--...
Content-Disposition: form-data; name="file"; filename="sample.csv"
Content-Type: text/csv

HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income
1,1,1,32.0,1,0,0,0,0,1,0,1,0,4,5,10,1,0,10,4,3
--...--
```

Response body is identical in shape to §2.2's `PredictResponse`. A file with
zero or more than one data row is rejected with a 422 pointing to
`/jobs/upload` for multi-row batches instead.

### 2.3 Validation error (out-of-range value, no silent coercion)

```http
POST /predict   { "record": { ...same as above, "BMI": 500 } }
```
```json
{
  "detail": "input validation failed",
  "error_code": "validation_error",
  "errors": [
    { "loc": ["body", "record", "BMI"], "msg": "Input should be less than or equal to 100", "type": "less_than_equal" }
  ],
  "request_id": "7925d590-de68-4a0f-ba61-de1d7da85248"
}
```

### 2.4 Batch inference (3-row CSV, 1 deliberately invalid row)

```http
POST /jobs/upload   (multipart/form-data, file=sample.csv)
```
```json
{ "job_id": "63bc6fa0-...", "status": "queued", "total_rows": 3, "message": "batch job accepted; poll GET /jobs/{job_id} for status" }
```
— HTTP status **202 Accepted**.

Polling `GET /jobs/{job_id}` until terminal:
```json
{
  "job_id": "63bc6fa0-...", "filename": "sample.csv", "status": "partial",
  "total_rows": 3, "processed_rows": 3, "succeeded_rows": 2, "failed_rows": 1,
  "model_version_id": "14f2519a-...",
  "created_at": "2026-09-14T14:38:19Z", "started_at": "...", "finished_at": "..."
}
```

`GET /jobs/{job_id}/results`:
```json
{
  "job_id": "63bc6fa0-...", "status": "partial",
  "rows": [
    { "row_index": 0, "status": "ok",    "prediction": 1, "probability": 0.857, "confidence": 0.857, "error": null },
    { "row_index": 1, "status": "ok",    "prediction": 0, "probability": 0.043, "confidence": 0.957, "error": null },
    { "row_index": 2, "status": "error", "prediction": null, "probability": null, "confidence": null,
      "error": "BMI: Input should be less than or equal to 100" }
  ]
}
```

One bad row (`BMI=999`) never blocks the other two — the batch terminates `partial`, with
per-row detail.

## 3. Versioning and logging design

**`model_versions`** rows are the unit of versioning: `name` + `version` + `framework` +
`artifact_path` + `artifact_sha256` (+ the scaler's own path/hash) + `preprocessing_version`
+ `status` (`candidate → staging/production → archived`) + `metrics` (JSONB) +
`promoted_at`. A **partial unique index** (`WHERE status = 'production'`) enforces *at the
database level* that at most one version per model name is ever live — not just an
application-level check, so even a bug or a race can't create two "production" versions.

**`promotion_events`** is an append-only audit log: every status transition records who
(`actor_user_id`), when, why (`reason`, required, non-empty), and the metrics snapshot at
that moment. This is what a rollback replays.

**`inference_logs`** captures enough to reproduce or dispute any single inference later:
`model_version_id`, `input_hash` (SHA-256 of the canonical input, for dedup/audit),
`input_echo` (the actual values scored), `prediction`, `probability`, `threshold`,
`latency_ms`, and — for a failed inference — `status='error'` with `error_code` /
`error_message`, so failures are traceable, not silently dropped. Batch rows use the same
table (`batch_job_id` + `row_index` set), so single and batch inference share one audit
trail and one set of dashboard/API queries.

**`request_logs`** captures every HTTP request (endpoint, method, status, latency) for the
monitoring dashboard's error-rate/latency panels, independent of whether the request
reached the ML layer at all (e.g. a 401 or 422 still gets logged).

I chose **PostgreSQL** (the assignment's stated preference) via SQLAlchemy + Alembic
migrations, over SQLite: the "exactly one production model" invariant needs a real partial
unique index (SQLite supports this too via `WHERE`, but Postgres's concurrent-write
semantics are the safer choice for a service that logs on every request), and JSONB
`metrics`/`metrics_snapshot` columns are natural in Postgres. Alembic gives a real, reviewable
migration history rather than `create_all()`, which matters for a service that's meant to
evolve past a take-home.

## 4. Promotion / rollback strategy

See [DESIGN.md](DESIGN.md) for the full answer to the retraining-process question. In
short, this service implements the gate end-to-end, not just describes it:

1. New artifacts are registered as `status='candidate'` — never auto-served.
2. `POST /models/{id}/evaluate` scores the candidate against a **fixed, seeded holdout
   slice** (last 20% of a deterministic shuffle of the eval CSV, `random_state=42`) so
   metrics are directly comparable run-to-run and version-to-version. Results (accuracy,
   precision, recall, F1, ROC-AUC) are stored on the version row.
3. `POST /models/{id}/promote` (role: `ml_engineer` only) requires a non-empty `reason`,
   refuses to promote a version with no recorded metrics, and verifies the candidate
   actually *loads* before flipping any status (never promote something that would 503 on
   the next request). The transaction demotes the current production version to
   `archived` — explicitly flushed before promoting the candidate, since Postgres/SQLite
   enforce the "one production" unique index immediately, not deferred — and both
   transitions are recorded in `promotion_events`.
4. `POST /models/rollback` (role: `ml_engineer`) restores either an explicit target version
   or the most recently archived production version, through the identical audited,
   load-verified transaction.

The seeded state starts the RandomForest in production (the model card's own conclusion —
tree models are competitive-to-superior on this medium-sized tabular data — plus it avoids
a TensorFlow cold-start on first request) and the MLP as a `candidate`, so the whole
evaluate → promote → rollback path is exercisable immediately after `docker compose up`
without any manual setup.

## 5. Testing approach

53+ pytest tests across `backend/tests/`, run against an **isolated in-memory SQLite** DB
(no Postgres needed to run the suite) with the FastAPI DB dependency overridden per test.
Most tests use a small deterministic fake classifier (not the real 21MB RF) so the default
suite runs in well under a minute and never imports TensorFlow; a separate
`test_real_artifacts.py` module runs the *actual* RF + scaler end-to-end (auto-skipped if
the artifacts haven't been fetched locally, but always run for real inside the Docker image
via CI or `docker compose up` + manual pytest).

Coverage, by requirement:

- **Valid single + batch inference** — `test_predict.py`, `test_batch.py`.
- **Invalid/malformed input** — unparseable CSV (422), empty file (422), header-only CSV
  (422), wrong types (`"not-a-number"` for BMI → 422), extra fields (`extra="forbid"` →
  422), missing fields (422), out-of-range values (422), and — importantly — no silent
  coercion: `Smoker: true` (a JSON boolean) is explicitly rejected rather than cast to `1`.
- **Model load / inference failure** — a corrupted (non-pickle) artifact file → 503 via
  `ModelLoadError`; a model whose `predict_proba` raises → 500 via `InferenceError` with a
  clean body, not the generic 500 handler; a batch row causing an unexpected exception is
  caught per-row and marked failed rather than crashing the whole job (this was a real bug
  I found and fixed during development — see below).
- **Version loading/selection** — explicit `model_version_id` on `/predict`, unknown
  version → 503, no-production-model → 503.
- **Persistence** — every successful and failed inference is asserted present in
  `inference_logs` with the right `model_version_id` and `input_hash`.
- **Promotion/rollback** — RBAC (403 for non-`ml_engineer`), missing-metrics rejection,
  missing-reason rejection, the "exactly one production version" DB invariant after
  promote, full rollback round-trip, audit event creation.
- **API error handling sweep** — every scenario above asserts a *specific* status code and
  `error_code`, not just "not a 500" — e.g. `test_predict_model_inference_failure_...`
  explicitly checks the body is the clean `inference_error` message, not the catch-all
  `internal_error` string, proving the typed-exception path was actually taken.

Run `pytest -q` from `backend/`; see README.md for setup. Latest local run: **53 passed** in
~50s (plus 3 passed / 1 skipped in `test_real_artifacts.py` against the real RF artifact;
the Keras case is skipped on the host machine, which can't install TensorFlow under Python
3.14, and runs inside Docker instead).

### A real bug this testing approach caught

While writing `test_batch.py`, a garbage CSV row (columns unrelated to the 21 expected
fields) crashed the batch worker with an unhandled `KeyError` inside
`canonical_input_hash`, which assumed every field of `FEATURE_ORDER` was present even for a
row that had already failed validation — leaving the job stuck in `running` forever. Fixed
by making the hash function accept arbitrary partial dicts, plus a last-resort per-row
`try/except` in the worker loop so no single malformed row can ever take down batch
processing for the rest of the file.

## 6. Assumptions, tradeoffs, and limitations

- **In-process asyncio batch worker**, not Celery/Redis. A `BatchJob` row (`status`,
  counts) is the durable unit of work; the queue itself is in-memory and re-created per
  process start. For a single-instance take-home deployment this is simpler to run and
  test, and a batch that was mid-flight on restart just needs re-uploading — acceptable
  here, explicitly noted as the first thing to swap for a real multi-instance deployment.
- **Model cache is per-process**, not shared across workers. Fine for the single-`uvicorn`-
  process deployment in `docker-compose.yml`; a multi-worker deployment would need a shared
  cache or per-worker warm-up on promotion (the registry's `evict()` call already
  centralizes the invalidation point for this).
- **Evaluation holdout is a slice of the same labeled dataset**, not an independent
  physician-adjudicated set — there is no such second dataset available for this take-home.
  This is called out explicitly as a real limitation: it validates *consistency* across
  model versions on the same distribution, not real-world generalization. DESIGN.md
  addresses this directly for the production retraining process.
- **Bonus scope fully implemented**: RBAC (JWT + 3-tier roles enforced on every mutating
  endpoint), basic monitoring (`/metrics/summary` for latency percentiles/error rate/
  prediction distribution, `/metrics/drift` for PSI-based feature drift vs the training
  baseline), and OpenAPI docs at `/docs` (FastAPI default, kept rather than hand-rolled).
- **passlib was dropped in favor of calling `bcrypt` directly** for password hashing:
  passlib's bcrypt backend does a version probe (`bcrypt.__about__`) that no longer exists
  in modern `bcrypt` releases, a known upstream incompatibility that surfaced immediately
  in testing — calling `bcrypt.hashpw`/`checkpw` directly avoids it entirely.
- **ONNX Runtime was identified but scoped out.** Converting both models to ONNX would
  drop the backend image's TensorFlow dependency (~600MB+) and likely cut inference
  latency, since `onnxruntime` is a much lighter, CPU-optimized runtime than full
  TensorFlow for a 2,945-parameter MLP. Deferred for this submission to keep the served
  artifacts numerically identical to the published `.keras`/`.joblib` files with zero
  conversion risk; worth revisiting if image size or cold-start latency become real
  constraints in production.
