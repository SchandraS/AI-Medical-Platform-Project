"""Runs a single inference through scaler -> model, returning a structured
result. This is the ONLY place that builds the model input array — every
caller (single predict, batch worker, evaluate) goes through here so the
feature-ordering contract is enforced in exactly one location.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.ml.errors import InferenceError
from app.ml.schema import CLASS_LABELS, FEATURE_ORDER


@dataclass
class PredictionResult:
    prediction: int
    probability: float  # P(class=1), i.e. P(diabetes_or_prediabetes)
    confidence: float  # max(p, 1-p)
    label: str
    threshold: float


def canonical_input_hash(record: dict[str, Any]) -> str:
    """SHA256 of the input record in a stable, field-order-independent form.
    Used to detect duplicate/repeated requests and to let a clinician verify
    a stored log matches the input they submitted.

    Accepts any dict, not just a fully-valid 21-field record: a row that
    failed validation (missing/extra/malformed fields) is still hashed from
    whatever it actually contains, sorted by key, so a batch row that never
    became a valid record can still be logged and traced back to its input
    rather than crashing the batch worker.
    """
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_input_array(record: dict[str, Any]) -> np.ndarray:
    try:
        row = [float(record[name]) for name in FEATURE_ORDER]
    except KeyError as exc:
        raise InferenceError(f"missing feature in validated record: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise InferenceError(f"non-numeric feature value in validated record: {exc}") from exc
    return np.array([row], dtype=float)


def predict_one(
    record: dict[str, Any],
    *,
    model: Any,
    scaler: Any,
    framework: str,
    threshold: float = 0.5,
) -> PredictionResult:
    """record must already be validated (see app.schemas) and contain exactly
    the 21 expected fields. Raises InferenceError on any runtime failure
    (shape mismatch, NaN, framework error) — never lets a raw exception from
    sklearn/keras/numpy propagate to the API layer."""
    X = _build_input_array(record)

    try:
        Xs = scaler.transform(X)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"feature scaling failed: {exc}") from exc

    try:
        if framework == "sklearn":
            proba = model.predict_proba(Xs)
            prob = float(proba[0, 1])
        elif framework == "keras":
            pred = model.predict(Xs, verbose=0)
            prob = float(np.asarray(pred).reshape(-1)[0])
        else:
            raise InferenceError(f"unsupported framework: {framework}")
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"model inference failed: {exc}") from exc

    if not np.isfinite(prob):
        raise InferenceError(f"model produced a non-finite probability: {prob}")
    prob = min(max(prob, 0.0), 1.0)

    label_id = int(prob >= threshold)
    return PredictionResult(
        prediction=label_id,
        probability=prob,
        confidence=max(prob, 1 - prob),
        label=CLASS_LABELS[label_id],
        threshold=threshold,
    )


def predict_batch(
    records: list[dict[str, Any]],
    *,
    model: Any,
    scaler: Any,
    framework: str,
    threshold: float = 0.5,
) -> list[PredictionResult]:
    """Vectorized batch inference for throughput; falls back to per-row calls
    only if the vectorized path fails, so a single bad row doesn't require
    aborting the whole batch (the caller is expected to have validated each
    record already; this is purely a performance path)."""
    if not records:
        return []
    X = np.array([[float(r[name]) for name in FEATURE_ORDER] for r in records], dtype=float)
    try:
        Xs = scaler.transform(X)
        if framework == "sklearn":
            probs = model.predict_proba(Xs)[:, 1]
        elif framework == "keras":
            probs = np.asarray(model.predict(Xs, verbose=0)).reshape(-1)
        else:
            raise InferenceError(f"unsupported framework: {framework}")
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"batch model inference failed: {exc}") from exc

    results = []
    for prob in probs:
        prob = float(prob)
        if not np.isfinite(prob):
            raise InferenceError(f"model produced a non-finite probability: {prob}")
        prob = min(max(prob, 0.0), 1.0)
        label_id = int(prob >= threshold)
        results.append(
            PredictionResult(
                prediction=label_id,
                probability=prob,
                confidence=max(prob, 1 - prob),
                label=CLASS_LABELS[label_id],
                threshold=threshold,
            )
        )
    return results
