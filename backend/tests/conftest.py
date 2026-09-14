"""Shared pytest fixtures.

Design: tests run against an isolated in-memory SQLite DB (fast, no Docker
needed to run `pytest`), with the FastAPI DB dependency overridden. Model
artifacts used in most tests are lightweight fakes (a fitted-looking
StandardScaler + a deterministic fake sklearn-style classifier) written to
temp files via joblib, so the bulk of the suite never imports TensorFlow and
runs in seconds. A smaller set of tests marked with the `keras` fixture
exercises the real Keras artifact end-to-end when it's available (skipped
gracefully if the artifact/network isn't present, e.g. offline CI).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.ml import registry as model_registry
from app.ml.schema import FEATURE_ORDER
from app.models_db import Framework, ModelStatus, ModelVersion, Role, User
from app.security import hash_password


# ---------------------------------------------------------------------------
# Fake model: deterministic, no ML framework dependency, mirrors the real
# sklearn predict_proba(X) -> (n, 2) interface.
# ---------------------------------------------------------------------------
class FakeBinaryClassifier:
    """Predicts class 1 iff the (scaled) BMI feature is above its mean (0 in
    scaled space); otherwise class 0. Deterministic and cheap, but exercises
    the full array-shape / predict_proba contract the real RF model has."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        bmi_col = FEATURE_ORDER.index("BMI")
        scaled_bmi = X[:, bmi_col]
        p1 = 1 / (1 + np.exp(-scaled_bmi))  # sigmoid squashes to a valid probability
        return np.column_stack([1 - p1, p1])


class BrokenClassifier:
    """A model whose predict_proba always raises — used to test the
    InferenceError -> HTTP 500 path without corrupting a real artifact."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("simulated inference failure: incompatible input shape")


@pytest.fixture(scope="session")
def fitted_scaler_path(tmp_path_factory) -> Path:
    """A real StandardScaler fitted on synthetic data spanning each
    feature's valid range, with feature_names_in_ set to FEATURE_ORDER —
    mirrors the real artifact's contract exactly."""
    rng = np.random.default_rng(42)
    from app.ml.schema import FEATURE_MAP

    n = 500
    data = {}
    for name in FEATURE_ORDER:
        spec = FEATURE_MAP[name]
        if spec.binary:
            data[name] = rng.integers(0, 2, size=n).astype(float)
        else:
            data[name] = rng.uniform(spec.minimum, spec.maximum, size=n)
    import pandas as pd

    df = pd.DataFrame(data)[FEATURE_ORDER]
    scaler = StandardScaler().fit(df)

    d = tmp_path_factory.mktemp("artifacts")
    path = d / "scaler.joblib"
    joblib.dump(scaler, path)
    return path


@pytest.fixture(scope="session")
def fake_model_path(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("artifacts")
    path = d / "fake_model.joblib"
    joblib.dump(FakeBinaryClassifier(), path)
    return path


@pytest.fixture(scope="session")
def broken_model_path(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("artifacts")
    path = d / "broken_model.joblib"
    joblib.dump(BrokenClassifier(), path)
    return path


@pytest.fixture(scope="session")
def corrupted_artifact_path(tmp_path_factory) -> Path:
    """Not a valid pickle at all -- simulates disk/transfer corruption."""
    d = tmp_path_factory.mktemp("artifacts")
    path = d / "corrupted.joblib"
    path.write_bytes(b"this is not a valid joblib/pickle file")
    return path


@pytest.fixture()
def db_session(monkeypatch):
    # A single shared in-memory SQLite connection (via StaticPool) so the
    # request-logging middleware and batch worker -- which open their own
    # sessions via app.db.new_session() rather than the FastAPI dependency --
    # see the same tables and data as the test's own `db_session`.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    import app.db as db_module

    monkeypatch.setattr(db_module, "SessionLocal", TestingSessionLocal)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def seeded_users(db_session):
    users = {}
    for username, password, role in [
        ("viewer", "viewer123", Role.viewer),
        ("clinician", "clinician123", Role.clinician),
        ("mlops", "mlops123", Role.ml_engineer),
    ]:
        u = User(username=username, hashed_password=hash_password(password), role=role)
        db_session.add(u)
        users[username] = u
    db_session.commit()
    for u in users.values():
        db_session.refresh(u)
    return users


@pytest.fixture()
def production_model_version(db_session, fitted_scaler_path, fake_model_path):
    v = ModelVersion(
        name="diabetes-risk",
        version="fake-v1",
        framework=Framework.sklearn,
        artifact_path=str(fake_model_path),
        artifact_sha256=model_registry.sha256_of_file(fake_model_path),
        scaler_artifact_path=str(fitted_scaler_path),
        scaler_sha256=model_registry.sha256_of_file(fitted_scaler_path),
        preprocessing_version="test-v1",
        status=ModelStatus.production,
        metrics={"accuracy": 0.75, "f1": 0.74},
    )
    db_session.add(v)
    db_session.commit()
    db_session.refresh(v)
    return v


@pytest.fixture()
def candidate_model_version(db_session, fitted_scaler_path, fake_model_path):
    v = ModelVersion(
        name="diabetes-risk",
        version="fake-v2-candidate",
        framework=Framework.sklearn,
        artifact_path=str(fake_model_path),
        artifact_sha256=model_registry.sha256_of_file(fake_model_path),
        scaler_artifact_path=str(fitted_scaler_path),
        scaler_sha256=model_registry.sha256_of_file(fitted_scaler_path),
        preprocessing_version="test-v1",
        status=ModelStatus.candidate,
    )
    db_session.add(v)
    db_session.commit()
    db_session.refresh(v)
    return v


@pytest.fixture(autouse=True)
def _clear_model_registry_cache():
    model_registry.clear_cache()
    yield
    model_registry.clear_cache()


@pytest.fixture()
def client(db_session):
    from app.main import app

    def _get_db_override():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _get_db_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def auth_headers(client: TestClient, username: str, password: str) -> dict:
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


VALID_RECORD = {
    "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 28.0, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2,
    "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 1, "Age": 9, "Education": 5,
    "Income": 7,
}
