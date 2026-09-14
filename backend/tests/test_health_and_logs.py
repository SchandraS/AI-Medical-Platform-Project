from tests.conftest import VALID_RECORD, auth_headers


def test_health_is_always_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_reports_not_ready_without_production_model(client, seeded_users):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["database"] is True
    assert body["checks"]["production_model_registered"] is False


def test_ready_reports_ready_with_production_model(client, seeded_users, production_model_version):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_inference_logs_scoped_to_own_user_for_clinician(
    client, seeded_users, production_model_version
):
    clinician_headers = auth_headers(client, "clinician", "clinician123")
    client.post("/predict", json={"record": VALID_RECORD}, headers=clinician_headers)

    resp = client.get("/logs/inferences", headers=clinician_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
    assert all(l["user_id"] == seeded_users["clinician"].id for l in resp.json())


def test_inference_logs_visible_to_ml_engineer_across_users(
    client, seeded_users, production_model_version
):
    clinician_headers = auth_headers(client, "clinician", "clinician123")
    client.post("/predict", json={"record": VALID_RECORD}, headers=clinician_headers)

    mlops_headers = auth_headers(client, "mlops", "mlops123")
    resp = client.get("/logs/inferences", headers=mlops_headers)
    assert resp.status_code == 200
    assert any(l["user_id"] == seeded_users["clinician"].id for l in resp.json())


def test_inference_logs_requires_auth(client, seeded_users):
    resp = client.get("/logs/inferences")
    assert resp.status_code == 401


def test_metrics_summary_reflects_requests(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    client.post("/predict", json={"record": VALID_RECORD}, headers=headers)

    resp = client.get("/metrics/summary", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["inference_count"] >= 1
    assert body["active_model"]["id"] == production_model_version.id


def test_metrics_drift_503_without_reference_dataset(client, seeded_users, tmp_path, monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nonexistent"))
    get_settings.cache_clear()

    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.get("/metrics/drift", headers=headers)
    assert resp.status_code == 503
    get_settings.cache_clear()
