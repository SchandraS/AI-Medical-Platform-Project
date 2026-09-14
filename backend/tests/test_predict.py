"""Single-prediction endpoint: happy path, validation, model errors."""
from tests.conftest import VALID_RECORD, auth_headers


def test_predict_valid_record_returns_full_metadata(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["model_version_id"] == production_model_version.id
    assert body["model_version"] == "fake-v1"
    assert body["preprocessing_version"] == "test-v1"
    assert len(body["input_hash"]) == 64  # sha256 hex
    assert body["input_echo"]["BMI"] == VALID_RECORD["BMI"]
    assert body["log_id"]


def test_predict_persists_inference_log(client, seeded_users, production_model_version, db_session):
    from app.models_db import InferenceLog

    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    assert resp.status_code == 200
    log_id = resp.json()["log_id"]

    log = db_session.query(InferenceLog).filter(InferenceLog.id == log_id).first()
    assert log is not None
    assert log.status == "ok"
    assert log.model_version_id == production_model_version.id
    assert log.probability is not None


def test_predict_rejects_unexpected_field(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {**VALID_RECORD, "SomeExtraField": 1}
    resp = client.post("/predict", json={"record": bad}, headers=headers)
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "validation_error"


def test_predict_rejects_missing_field(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {k: v for k, v in VALID_RECORD.items() if k != "BMI"}
    resp = client.post("/predict", json={"record": bad}, headers=headers)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_value(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {**VALID_RECORD, "BMI": 500}
    resp = client.post("/predict", json={"record": bad}, headers=headers)
    assert resp.status_code == 422


def test_predict_rejects_wrong_type(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {**VALID_RECORD, "BMI": "not-a-number"}
    resp = client.post("/predict", json={"record": bad}, headers=headers)
    assert resp.status_code == 422


def test_predict_rejects_non_binary_value_for_binary_field(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {**VALID_RECORD, "HighBP": 2}
    resp = client.post("/predict", json={"record": bad}, headers=headers)
    assert resp.status_code == 422


def test_predict_rejects_boolean_for_binary_field(client, seeded_users, production_model_version):
    """No silent coercion: `true`/`false` must not silently become 1/0."""
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {**VALID_RECORD, "Smoker": True}
    resp = client.post("/predict", json={"record": bad}, headers=headers)
    assert resp.status_code == 422


def test_predict_requires_auth(client, seeded_users, production_model_version):
    resp = client.post("/predict", json={"record": VALID_RECORD})
    assert resp.status_code == 401


def test_predict_viewer_role_forbidden(client, seeded_users, production_model_version):
    headers = auth_headers(client, "viewer", "viewer123")
    resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    assert resp.status_code == 403


def test_predict_no_production_model_returns_503(client, seeded_users):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    assert resp.status_code == 503


def test_predict_corrupted_model_artifact_returns_503(
    client, seeded_users, db_session, fitted_scaler_path, corrupted_artifact_path
):
    from app.ml import registry as model_registry
    from app.models_db import Framework, ModelStatus, ModelVersion

    v = ModelVersion(
        name="diabetes-risk", version="corrupt-v1", framework=Framework.sklearn,
        artifact_path=str(corrupted_artifact_path),
        artifact_sha256=model_registry.sha256_of_file(corrupted_artifact_path),
        scaler_artifact_path=str(fitted_scaler_path),
        scaler_sha256=model_registry.sha256_of_file(fitted_scaler_path),
        preprocessing_version="test-v1", status=ModelStatus.production,
    )
    db_session.add(v)
    db_session.commit()

    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "model_load_error"


def test_predict_model_inference_failure_returns_500_not_unhandled(
    client, seeded_users, db_session, fitted_scaler_path, broken_model_path
):
    from app.ml import registry as model_registry
    from app.models_db import Framework, ModelStatus, ModelVersion

    v = ModelVersion(
        name="diabetes-risk", version="broken-v1", framework=Framework.sklearn,
        artifact_path=str(broken_model_path),
        artifact_sha256=model_registry.sha256_of_file(broken_model_path),
        scaler_artifact_path=str(fitted_scaler_path),
        scaler_sha256=model_registry.sha256_of_file(fitted_scaler_path),
        preprocessing_version="test-v1", status=ModelStatus.production,
    )
    db_session.add(v)
    db_session.commit()

    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == "inference_error"
    assert "an unexpected error occurred" not in body["detail"]  # clean, specific message, not the catch-all


def test_predict_explicit_model_version_id(client, seeded_users, production_model_version, candidate_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(
        "/predict",
        json={"record": VALID_RECORD, "model_version_id": candidate_model_version.id},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["model_version_id"] == candidate_model_version.id


def test_predict_unknown_model_version_id_returns_503(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(
        "/predict", json={"record": VALID_RECORD, "model_version_id": "does-not-exist"}, headers=headers
    )
    assert resp.status_code == 503
