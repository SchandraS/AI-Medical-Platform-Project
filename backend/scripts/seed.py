"""Idempotent startup seed: creates the 3 demo RBAC users and registers both
pretrained models in model_versions. The RandomForest starts in production
(it's the more standard/interpretable choice for a tabular clinical model
per the model card's own conclusion); the MLP starts as a candidate,
available for evaluation/promotion through the normal gate — it is NOT
auto-promoted, demonstrating the promotion workflow end-to-end.

Safe to run every container start: skips creation if rows already exist.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import SessionLocal
from app.models_db import Framework, ModelStatus, ModelVersion, Role, User
from app.security import hash_password

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manas.seed")

DEMO_USERS = [
    ("viewer", "viewer123", Role.viewer),
    ("clinician", "clinician123", Role.clinician),
    ("mlops", "mlops123", Role.ml_engineer),
]

MODEL_NAME = "diabetes-risk"


def seed_users(db) -> None:
    for username, password, role in DEMO_USERS:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            continue
        db.add(User(username=username, hashed_password=hash_password(password), role=role))
        logger.info("seeded user '%s' with role '%s'", username, role.value)
    db.commit()


def seed_models(db) -> None:
    settings = get_settings()
    artifacts_dir = Path(settings.artifacts_dir)
    manifest_path = artifacts_dir / "manifest.json"

    if not manifest_path.exists():
        logger.warning(
            "artifacts manifest not found at %s; skipping model registration. "
            "Run scripts/fetch_artifacts.py first.",
            manifest_path,
        )
        return

    manifest = json.loads(manifest_path.read_text())
    required = ["diabetes_random_forest_tuned.joblib", "diabetes_mlp_tuned.keras", "scaler.joblib"]
    missing = [f for f in required if f not in manifest]
    if missing:
        logger.warning("manifest missing entries for %s; skipping model registration", missing)
        return

    scaler_path = str(artifacts_dir / "scaler.joblib")
    scaler_sha = manifest["scaler.joblib"]["sha256"]

    specs = [
        ("rf-v1", Framework.sklearn, "diabetes_random_forest_tuned.joblib", ModelStatus.production),
        ("mlp-v1", Framework.keras, "diabetes_mlp_tuned.keras", ModelStatus.candidate),
    ]

    for version, framework, filename, status in specs:
        existing = (
            db.query(ModelVersion)
            .filter(ModelVersion.name == MODEL_NAME, ModelVersion.version == version)
            .first()
        )
        if existing:
            continue
        db.add(
            ModelVersion(
                name=MODEL_NAME,
                version=version,
                framework=framework,
                artifact_path=str(artifacts_dir / filename),
                artifact_sha256=manifest[filename]["sha256"],
                scaler_artifact_path=scaler_path,
                scaler_sha256=scaler_sha,
                preprocessing_version=settings.preprocessing_version,
                status=status,
                notes="Pretrained artifact from HuggingFace Neperl/diabetes-health-indicators-study",
            )
        )
        logger.info("registered model version '%s' (%s) with status '%s'", version, framework.value, status.value)
    db.commit()


def main() -> int:
    db = SessionLocal()
    try:
        seed_users(db)
        seed_models(db)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
