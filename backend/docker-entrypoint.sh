#!/bin/sh
# Boot sequence: fetch artifacts/data (idempotent, network-tolerant) -> run
# migrations -> seed demo users + register model versions -> start the app.
set -e

echo "[entrypoint] fetching model artifacts..."
python scripts/fetch_artifacts.py

echo "[entrypoint] fetching evaluation dataset..."
python scripts/fetch_data.py

echo "[entrypoint] running database migrations..."
alembic upgrade head

echo "[entrypoint] seeding demo users and model registry..."
python scripts/seed.py

echo "[entrypoint] starting application..."
exec "$@"
