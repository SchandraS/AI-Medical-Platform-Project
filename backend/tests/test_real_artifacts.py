"""Integration tests against the REAL pretrained artifacts (RandomForest +
scaler; the Keras MLP separately, since it needs TensorFlow). These are
skipped automatically if the artifacts haven't been fetched (e.g. offline
dev, or a machine where scripts/fetch_artifacts.py hasn't run) -- they are
NOT part of the fast default suite's required-to-pass set, but run for real
in CI/Docker where the artifacts are always fetched at boot.

This is also where the feature-order contract gets its strongest check: the
real scaler.feature_names_in_ is compared against app.ml.schema.FEATURE_ORDER
byte for byte.
"""
from pathlib import Path

import pytest

from app.ml.registry import get_model_and_scaler, sha256_of_file
from app.ml.predictor import predict_one
from app.ml.schema import FEATURE_ORDER

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RF_PATH = ARTIFACTS_DIR / "diabetes_random_forest_tuned.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
MLP_PATH = ARTIFACTS_DIR / "diabetes_mlp_tuned.keras"

requires_real_artifacts = pytest.mark.skipif(
    not (RF_PATH.exists() and SCALER_PATH.exists()),
    reason="real model artifacts not present locally; run scripts/fetch_artifacts.py "
    "or rely on the Docker image, which fetches them at boot",
)

requires_keras = pytest.mark.skipif(
    not MLP_PATH.exists(), reason="keras artifact not present locally"
)

SAMPLE_RECORD = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32.0, "Smoker": 1, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 4,
    "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1, "Sex": 0, "Age": 10, "Education": 4,
    "Income": 3,
}


@requires_real_artifacts
def test_real_scaler_feature_order_matches_schema_contract():
    """The single highest-value check in this test suite: confirms the
    ACTUAL published scaler's feature_names_in_ matches FEATURE_ORDER."""
    import joblib

    scaler = joblib.load(SCALER_PATH)
    assert list(scaler.feature_names_in_) == FEATURE_ORDER


@requires_real_artifacts
def test_real_random_forest_produces_valid_prediction():
    model, scaler = get_model_and_scaler(
        version_id="test-real-rf",
        artifact_path=str(RF_PATH),
        scaler_path=str(SCALER_PATH),
        framework="sklearn",
        artifact_sha256=sha256_of_file(RF_PATH),
        scaler_sha256=sha256_of_file(SCALER_PATH),
    )
    result = predict_one(SAMPLE_RECORD, model=model, scaler=scaler, framework="sklearn", threshold=0.5)
    assert result.prediction in (0, 1)
    assert 0.0 <= result.probability <= 1.0


@requires_real_artifacts
def test_real_random_forest_rejects_wrong_shape_gracefully():
    """Feeding a mismatched feature count must raise InferenceError (caught
    -> 500), never crash the process or return a bogus result."""
    from app.ml.errors import InferenceError

    model, scaler = get_model_and_scaler(
        version_id="test-real-rf-2",
        artifact_path=str(RF_PATH),
        scaler_path=str(SCALER_PATH),
        framework="sklearn",
    )
    bad_record = {k: v for k, v in list(SAMPLE_RECORD.items())[:5]}  # missing 16 features
    with pytest.raises(InferenceError):
        predict_one(bad_record, model=model, scaler=scaler, framework="sklearn")


@requires_keras
def test_real_mlp_produces_valid_prediction():
    """Exercises the actual Keras artifact end-to-end. Skipped if
    tensorflow/keras aren't installed in the current environment (e.g. this
    repo's host Python) -- runs for real inside the Docker image."""
    pytest.importorskip("keras")

    model, scaler = get_model_and_scaler(
        version_id="test-real-mlp",
        artifact_path=str(MLP_PATH),
        scaler_path=str(SCALER_PATH),
        framework="keras",
        artifact_sha256=sha256_of_file(MLP_PATH),
        scaler_sha256=sha256_of_file(SCALER_PATH),
    )
    result = predict_one(SAMPLE_RECORD, model=model, scaler=scaler, framework="keras", threshold=0.5)
    assert result.prediction in (0, 1)
    assert 0.0 <= result.probability <= 1.0
