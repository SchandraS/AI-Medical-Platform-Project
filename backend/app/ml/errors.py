"""Typed ML-layer exceptions. These are caught by FastAPI exception handlers
in app.main and mapped to clean HTTP responses — never an unhandled 500."""


class ModelLoadError(Exception):
    """Raised when a model/scaler artifact is missing, corrupted, or fails to
    deserialize. Maps to HTTP 503 (service temporarily unable to serve this
    model), not 500, because it's an infrastructure/data condition, not a bug."""


class InferenceError(Exception):
    """Raised when a loaded model fails during prediction (e.g. shape
    mismatch, NaN propagation). Maps to HTTP 500 with a clean, logged body."""


class NoProductionModelError(Exception):
    """Raised when no ModelVersion has status='production'. Maps to 503."""
