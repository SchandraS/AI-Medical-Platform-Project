from tests.conftest import auth_headers


def test_login_success_returns_token_and_role(client, seeded_users):
    resp = client.post("/auth/login", json={"username": "clinician", "password": "clinician123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"]
    assert body["role"] == "clinician"


def test_login_wrong_password_returns_401(client, seeded_users):
    resp = client.post("/auth/login", json={"username": "clinician", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user_returns_401(client, seeded_users):
    resp = client.post("/auth/login", json={"username": "nobody", "password": "x"})
    assert resp.status_code == 401


def test_login_rejects_extra_fields(client, seeded_users):
    resp = client.post(
        "/auth/login", json={"username": "clinician", "password": "clinician123", "extra": "field"}
    )
    assert resp.status_code == 422


def test_protected_endpoint_without_token_is_401(client, seeded_users):
    resp = client.get("/models")
    assert resp.status_code == 401


def test_protected_endpoint_with_garbage_token_is_401(client, seeded_users):
    resp = client.get("/models", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_role_hierarchy_ml_engineer_can_access_clinician_endpoints(
    client, seeded_users, production_model_version
):
    from tests.conftest import VALID_RECORD

    headers = auth_headers(client, "mlops", "mlops123")
    resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    assert resp.status_code == 200
