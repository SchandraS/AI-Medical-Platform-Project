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
├── frontend/            React/Vite app, Dockerfile, nginx.conf.template
├── docker-compose.yml   One command to bring up db + backend + frontend
├── render.yaml           Render Blueprint for a free public deployment
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

## Deploying to Render (free tier)

`render.yaml` at the repo root is a [Render Blueprint](https://render.com/docs/blueprint-spec)
that deploys the same three services as `docker-compose.yml` — Postgres, backend, frontend —
as independent public services instead of one Docker Compose network.

**Steps:**
1. Push this repo to GitHub (already done if you're reading this from there).
2. On [render.com](https://render.com), **New → Blueprint**, connect the repo. Render reads
   `render.yaml` and provisions `manas-db` (Postgres), `manas-backend`, and `manas-frontend`.
3. Wait for all three to finish deploying (the backend build is the slowest — it installs
   TensorFlow, ~2–4 min on Render's free build machines).
4. Open the `manas-frontend` service's URL. Log in with the same seeded demo accounts as
   local Docker Compose (see above) — the backend seeds them on every boot, same as locally.

**What's different from local Docker Compose, and why:**
- **`frontend/nginx.conf.template` + `frontend/docker-entrypoint.sh`**: locally, nginx
  proxies `/api/` to `http://backend:8000/` — a hostname only resolvable inside Docker
  Compose's private network. On Render the frontend and backend are separately-addressed
  services, so the proxy target is templated: `docker-entrypoint.sh` runs `envsubst` on
  `nginx.conf.template` at container start, filling in `BACKEND_HOST`/`BACKEND_PORT` from
  environment variables `render.yaml` wires to the backend service's real Render hostname.
  With no env vars set (e.g. a plain `docker build` outside Compose), it falls back to
  `backend:8000`, so this is a no-op change for local Docker Compose behavior.
- **`app/config.py`'s `database_url` and `cors_origins`** gained small normalizing
  validators: Render hands out a bare `postgresql://` connection string (SQLAlchemy would
  otherwise default to the unin­stalled `psycopg2` driver instead of the pinned `psycopg` 3
  driver), and a bare hostname for `CORS_ORIGINS` (Render's Blueprint cross-service
  references return a hostname, not a full URL or JSON array). Both are covered by
  `backend/tests/test_config.py`.
- **Postgres free tier expires after 90 days** unless upgraded to a paid plan — fine for a
  demo/portfolio deployment; recreate the database (or upgrade it) when Render emails you
  before expiry. An always-free alternative (e.g. Supabase Postgres) can be substituted by
  pointing the backend's `DATABASE_URL` env var at it instead of `fromDatabase: manas-db`.
- **Free web services sleep after 15 minutes idle** and cold-start slowly on the next
  request (the backend's boot sequence re-downloads model artifacts/eval data, runs
  migrations, and reseeds on every start — see Quickstart above), so the first request
  after a period of inactivity can take 30–60+ seconds. This is a demo-tier tradeoff, not a
  production one.

## Known deviations from a "preferred stack, no substitutions" reading

None required — PostgreSQL, FastAPI, and React are all used as preferred. The batch queue
is an in-process `asyncio` queue rather than Celery/Redis (see REPORT.md's tradeoffs
section for the reasoning); this is an addition, not a substitution of anything requested.
