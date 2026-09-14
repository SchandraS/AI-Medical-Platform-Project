"""Resolves a requested (or default/active) ModelVersion row to a loaded
(model, scaler) pair, with clean errors for every failure mode."""
from sqlalchemy.orm import Session

from app.ml.errors import NoProductionModelError
from app.ml.registry import get_model_and_scaler
from app.models_db import ModelStatus, ModelVersion


def get_production_version(db: Session, name: str = "diabetes-risk") -> ModelVersion:
    version = (
        db.query(ModelVersion)
        .filter(ModelVersion.name == name, ModelVersion.status == ModelStatus.production)
        .first()
    )
    if version is None:
        raise NoProductionModelError(f"no production model version found for '{name}'")
    return version


def get_version_or_default(db: Session, version_id: str | None, name: str = "diabetes-risk") -> ModelVersion:
    if version_id:
        version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
        if version is None:
            raise NoProductionModelError(f"model version '{version_id}' not found")
        return version
    return get_production_version(db, name)


def load_for_version(version: ModelVersion):
    framework = version.framework.value if hasattr(version.framework, "value") else version.framework
    return get_model_and_scaler(
        version_id=version.id,
        artifact_path=version.artifact_path,
        scaler_path=version.scaler_artifact_path,
        framework=framework,
        artifact_sha256=version.artifact_sha256,
        scaler_sha256=version.scaler_sha256,
    )
