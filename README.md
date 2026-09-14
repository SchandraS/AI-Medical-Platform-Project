# MANAS — Diabetes Risk Inference Service

A production-style inference service for the diabetes-risk models published at
[`Neperl/diabetes-health-indicators-study`](https://huggingface.co/Neperl/diabetes-health-indicators-study)
(a RandomForest and a small Keras MLP), with model versioning, an evaluation/promotion
gate, audited logging, RBAC, and basic monitoring.

**Stack:** FastAPI + PostgreSQL + SQLAlchemy/Alembic (backend), React + Vite + TypeScript
(frontend), Docker Compose. See [REPORT.md](REPORT.md) for the full design writeup and
[DESIGN.md](DESIGN.md) for the model-update/retraining design question.

## Quickstart

```bash
docker compose up --build
```

This builds and starts three services:

| Service  | URL                     | Notes |
|----------|-------------------------|-------|
| frontend | http://localhost:8080   | React app (nginx), proxies `/api` to the backend |
| backend  | http://localhost:8000   | FastAPI; interactive docs at `/docs` |
| db       | localhost:5432           | PostgreSQL 16 |

On first boot the backend automatically:
1. Downloads the 3 pretrained artifacts (RF, MLP, scaler) from HuggingFace into a volume.
2. Downloads a public no-auth mirror of the BRFSS2015 evaluation CSV (used for the
   promotion gate and drift baseline) into a volume.
3. Runs Alembic migrations.
4. Seeds 3 demo users and registers both model versions (RF starts in production, MLP
   starts as an evaluable candidate).

If your machine has no internet access at boot, the service still starts — model-dependent
endpoints return a clean `503` until the artifacts are fetched (see
`backend/scripts/fetch_artifacts.py` / `fetch_data.py`, safe to re-run any time).

### Demo accounts (seeded automatically)

| Username    | Password       | Role         |
|-------------|----------------|--------------|
| `clinician` | `clinician123` | clinician — run single/batch predictions |
| `mlops`     | `mlops123`     | ml_engineer — everything + evaluate/promote/rollback |
| `viewer`    | `viewer123`    | viewer — read-only dashboard |

## Project layout

```
repo-root/
├── backend/            FastAPI service, Alembic migrations, pytest suite, Dockerfile
├── frontend/            React/Vite app, Dockerfile, nginx.conf
├── docker-compose.yml   One command to bring up db + backend + frontend
├── README.md            (this file)
├── REPORT.md             Design write-up, example requests, test results
└── DESIGN.md             Answer to the retraining/promotion design question
```

## Running the backend without Docker

Requires **Python 3.12** (TensorFlow 2.21's wheels stop at cp313; the pinned
`scikit-learn==1.9.0`/`keras==3.15.*` versions match the published artifacts exactly —
using a different Python/library version risks silent numerical drift from the model
authors' environment).

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# fetch artifacts + eval data (idempotent, safe to skip if already present)
python scripts/fetch_artifacts.py
python scripts/fetch_data.py

# point at a local Postgres (or export DATABASE_URL to something else)
export DATABASE_URL=postgresql+psycopg://manas:manas@localhost:5432/manas
alembic upgrade head
python scripts/seed.py

uvicorn app.main:app --reload
```

## Running the frontend without Docker

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies /api to http://localhost:8000
```

## Running the tests

```bash
cd backend
pip install -r requirements.txt   # includes pytest, pytest-asyncio, pytest-cov
pytest -q
```

The suite runs against an isolated **in-memory SQLite** database (no Postgres needed to
run `pytest`) and uses lightweight fake model objects for most cases, so it completes in
well under a minute and never imports TensorFlow unless the real Keras artifact is present
(`tests/test_real_artifacts.py`, auto-skipped if `backend/artifacts/` is empty — populated
automatically inside the Docker image, or manually via `python scripts/fetch_artifacts.py`).

```bash
pytest -q                          # run everything
pytest -q --cov=app --cov-report=term-missing   # with coverage
pytest tests/test_predict.py -v    # a single file, verbose
```

The test suite is also present inside the running backend container, so it can be run
against the real environment (Python 3.12, pinned TensorFlow/scikit-learn/Keras, the real
fetched artifacts) without a local Python install:

```bash
docker compose exec backend pytest -q
```

Test coverage includes: valid single/batch inference, malformed/missing/extra-field input,
out-of-range values, non-coerced types (e.g. `true` is rejected for a binary field, not
silently cast to `1`), model load failure (corrupted artifact → 503, not 500), inference
failure (shape mismatch → 500 with a clean body, not an unhandled crash), persistence of
inference logs, RBAC on every mutating endpoint, the "exactly one production model" DB
invariant under promotion/rollback, and a guard test asserting the API's validation ranges
never drift from the ML feature contract extracted from the real scaler.

## API documentation

Interactive OpenAPI/Swagger docs are served at `/docs` (and `/redoc`) once the backend is
running — e.g. http://localhost:8000/docs. See REPORT.md for example request/response
bodies for the main endpoints.

## Known deviations from a "preferred stack, no substitutions" reading

None required — PostgreSQL, FastAPI, and React are all used as preferred. The batch queue
is an in-process `asyncio` queue rather than Celery/Redis (see REPORT.md's tradeoffs
section for the reasoning); this is an addition, not a substitution of anything requested.
