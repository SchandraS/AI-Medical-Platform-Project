"""Model versioning: listing, evaluation, promotion, rollback, RBAC, and the
'at most one production version' invariant."""
import pytest

from tests.conftest import auth_headers


def test_list_models_requires_auth(client, seeded_users):
    resp = client.get("/models")
    assert resp.status_code == 401


def test_list_models_returns_all_versions(client, seeded_users, production_model_version, candidate_model_version):
    headers = auth_headers(client, "viewer", "viewer123")
    resp = client.get("/models", headers=headers)
    assert resp.status_code == 200
    versions = {v["id"] for v in resp.json()}
    assert production_model_version.id in versions
    assert candidate_model_version.id in versions


def test_active_model_returns_production_version(client, seeded_users, production_model_version):
    headers = auth_headers(client, "viewer", "viewer123")
    resp = client.get("/models/active", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == production_model_version.id
    assert resp.json()["status"] == "production"


def test_active_model_503_when_none_deployed(client, seeded_users):
    headers = auth_headers(client, "viewer", "viewer123")
    resp = client.get("/models/active", headers=headers)
    assert resp.status_code == 503


def test_promote_requires_ml_engineer_role(client, seeded_users, production_model_version, candidate_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(
        f"/models/{candidate_model_version.id}/promote", json={"reason": "trying anyway"}, headers=headers
    )
    assert resp.status_code == 403


def test_promote_without_evaluation_metrics_is_rejected(
    client, seeded_users, production_model_version, candidate_model_version
):
    headers = auth_headers(client, "mlops", "mlops123")
    resp = client.post(
        f"/models/{candidate_model_version.id}/promote", json={"reason": "no eval yet"}, headers=headers
    )
    assert resp.status_code == 422
    assert "evaluate" in resp.json()["detail"].lower()


def test_promote_requires_reason(client, seeded_users, production_model_version, candidate_model_version, db_session):
    candidate_model_version.metrics = {"accuracy": 0.8}
    db_session.commit()
    headers = auth_headers(client, "mlops", "mlops123")
    resp = client.post(f"/models/{candidate_model_version.id}/promote", json={}, headers=headers)
    assert resp.status_code == 422


def test_promote_demotes_previous_production_and_logs_event(
    client, seeded_users, production_model_version, candidate_model_version, db_session
):
    from app.models_db import ModelStatus, PromotionEvent

    candidate_model_version.metrics = {"accuracy": 0.81, "f1": 0.8}
    db_session.commit()

    headers = auth_headers(client, "mlops", "mlops123")
    resp = client.post(
        f"/models/{candidate_model_version.id}/promote",
        json={"reason": "outperforms current production on holdout"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "production"

    db_session.refresh(production_model_version)
    db_session.refresh(candidate_model_version)
    assert production_model_version.status == ModelStatus.archived
    assert candidate_model_version.status == ModelStatus.production

    events = db_session.query(PromotionEvent).all()
    to_statuses = {e.to_status for e in events}
    assert "production" in to_statuses
    assert "archived" in to_statuses


def test_only_one_production_version_at_a_time(
    client, seeded_users, production_model_version, candidate_model_version, db_session
):
    from app.models_db import ModelStatus, ModelVersion

    candidate_model_version.metrics = {"accuracy": 0.81}
    db_session.commit()
    headers = auth_headers(client, "mlops", "mlops123")
    client.post(
        f"/models/{candidate_model_version.id}/promote", json={"reason": "promote"}, headers=headers
    )

    production_count = (
        db_session.query(ModelVersion)
        .filter(ModelVersion.name == "diabetes-risk", ModelVersion.status == ModelStatus.production)
        .count()
    )
    assert production_count == 1


def test_rollback_requires_ml_engineer_role(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post("/models/rollback", json={"reason": "bad predictions"}, headers=headers)
    assert resp.status_code == 403


def test_rollback_restores_previous_production_version(
    client, seeded_users, production_model_version, candidate_model_version, db_session
):
    from app.models_db import ModelStatus

    candidate_model_version.metrics = {"accuracy": 0.81}
    db_session.commit()
    headers = auth_headers(client, "mlops", "mlops123")

    promote_resp = client.post(
        f"/models/{candidate_model_version.id}/promote", json={"reason": "promote for test"}, headers=headers
    )
    assert promote_resp.status_code == 200

    rollback_resp = client.post(
        "/models/rollback", json={"reason": "unexpected drift in production"}, headers=headers
    )
    assert rollback_resp.status_code == 200, rollback_resp.text
    assert rollback_resp.json()["id"] == production_model_version.id

    db_session.refresh(production_model_version)
    db_session.refresh(candidate_model_version)
    assert production_model_version.status == ModelStatus.production
    assert candidate_model_version.status == ModelStatus.archived


def test_rollback_with_no_production_model_returns_409(client, seeded_users):
    headers = auth_headers(client, "mlops", "mlops123")
    resp = client.post("/models/rollback", json={"reason": "nothing to roll back"}, headers=headers)
    assert resp.status_code == 409


def test_promote_nonexistent_version_returns_404(client, seeded_users, production_model_version):
    headers = auth_headers(client, "mlops", "mlops123")
    resp = client.post("/models/does-not-exist/promote", json={"reason": "x"}, headers=headers)
    assert resp.status_code == 404


def test_evaluate_requires_ml_engineer_role(client, seeded_users, candidate_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(f"/models/{candidate_model_version.id}/evaluate", headers=headers)
    assert resp.status_code == 403


def test_evaluate_returns_503_when_dataset_unavailable(client, seeded_users, candidate_model_version, tmp_path, monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nonexistent"))
    get_settings.cache_clear()

    headers = auth_headers(client, "mlops", "mlops123")
    resp = client.post(f"/models/{candidate_model_version.id}/evaluate", headers=headers)
    assert resp.status_code == 503
    get_settings.cache_clear()


def test_promotion_history_lists_events(client, seeded_users, production_model_version, candidate_model_version, db_session):
    candidate_model_version.metrics = {"accuracy": 0.81}
    db_session.commit()
    headers = auth_headers(client, "mlops", "mlops123")
    client.post(f"/models/{candidate_model_version.id}/promote", json={"reason": "promote"}, headers=headers)

    resp = client.get("/models/promotions/history", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
