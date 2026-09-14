"""Loads and caches model/scaler artifacts from disk, keyed by ModelVersion.id.

Loading is lazy (first request pays the cost) and guarded: any failure to
read or deserialize an artifact raises ModelLoadError, which the API layer
turns into a clean 503 rather than crashing the process or the request.

A process-level cache means repeated predictions against the same version
don't re-deserialize the model each time. The cache is intentionally simple
(dict keyed by version id) — fine for a single-process take-home deployment;
a multi-worker deployment would front this with a shared cache or per-worker
warm-up, noted in REPORT.md.
"""
import hashlib
import logging
from pathlib import Path
from typing import Any

import joblib

from app.ml.errors import ModelLoadError
from app.ml.schema import FEATURE_ORDER

logger = logging.getLogger("manas.registry")

_model_cache: dict[str, Any] = {}
_scaler_cache: dict[str, Any] = {}


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_scaler(scaler_path: str, expected_sha256: str | None = None) -> Any:
    cache_key = scaler_path
    if cache_key in _scaler_cache:
        return _scaler_cache[cache_key]

    p = Path(scaler_path)
    if not p.exists():
        raise ModelLoadError(f"scaler artifact not found: {scaler_path}")

    if expected_sha256:
        actual = sha256_of_file(p)
        if actual != expected_sha256:
            raise ModelLoadError(
                f"scaler artifact hash mismatch for {scaler_path}: "
                f"expected {expected_sha256}, got {actual} (possible corruption)"
            )

    try:
        scaler = joblib.load(p)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"failed to deserialize scaler {scaler_path}: {exc}") from exc

    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is None:
        raise ModelLoadError(
            f"scaler {scaler_path} has no feature_names_in_; cannot verify feature contract"
        )
    if list(feature_names) != FEATURE_ORDER:
        raise ModelLoadError(
            "scaler.feature_names_in_ does not match the expected FEATURE_ORDER "
            f"contract. Refusing to serve. Expected={FEATURE_ORDER} Got={list(feature_names)}"
        )

    _scaler_cache[cache_key] = scaler
    return scaler


def _load_sklearn_model(artifact_path: str, expected_sha256: str | None) -> Any:
    p = Path(artifact_path)
    if not p.exists():
        raise ModelLoadError(f"model artifact not found: {artifact_path}")
    if expected_sha256:
        actual = sha256_of_file(p)
        if actual != expected_sha256:
            raise ModelLoadError(
                f"model artifact hash mismatch for {artifact_path}: "
                f"expected {expected_sha256}, got {actual} (possible corruption)"
            )
    try:
        return joblib.load(p)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"failed to deserialize sklearn model {artifact_path}: {exc}") from exc


def _load_keras_model(artifact_path: str, expected_sha256: str | None) -> Any:
    p = Path(artifact_path)
    if not p.exists():
        raise ModelLoadError(f"model artifact not found: {artifact_path}")
    if expected_sha256:
        actual = sha256_of_file(p)
        if actual != expected_sha256:
            raise ModelLoadError(
                f"model artifact hash mismatch for {artifact_path}: "
                f"expected {expected_sha256}, got {actual} (possible corruption)"
            )
    try:
        # Imported lazily: keras/tensorflow is a heavy import, and importing
        # it only when a keras model is actually requested keeps sklearn-only
        # code paths (and most of the test suite) fast.
        from keras.models import load_model

        return load_model(p)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"failed to deserialize keras model {artifact_path}: {exc}") from exc


def get_model_and_scaler(
    *,
    version_id: str,
    artifact_path: str,
    scaler_path: str,
    framework: str,
    artifact_sha256: str | None = None,
    scaler_sha256: str | None = None,
) -> tuple[Any, Any]:
    """Returns (model, scaler) for the given ModelVersion, loading + caching
    on first use. Raises ModelLoadError on any failure."""
    scaler = _load_scaler(scaler_path, scaler_sha256)

    if version_id in _model_cache:
        return _model_cache[version_id], scaler

    if framework == "sklearn":
        model = _load_sklearn_model(artifact_path, artifact_sha256)
    elif framework == "keras":
        model = _load_keras_model(artifact_path, artifact_sha256)
    else:
        raise ModelLoadError(f"unknown framework: {framework}")

    _model_cache[version_id] = model
    logger.info("loaded model version_id=%s framework=%s from %s", version_id, framework, artifact_path)
    return model, scaler


def evict(version_id: str) -> None:
    """Drop a cached model (e.g. after archiving) to free memory."""
    _model_cache.pop(version_id, None)


def clear_cache() -> None:
    """Test helper: reset all caches between test cases."""
    _model_cache.clear()
    _scaler_cache.clear()
