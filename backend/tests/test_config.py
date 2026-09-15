"""Guards the database_url driver-rewrite added for deploying to managed
Postgres providers (Render, Heroku, etc.), which typically hand out a bare
postgresql:// or postgres:// connection string. Without this rewrite,
SQLAlchemy would default to the psycopg2 driver -- not installed here, since
requirements.txt pins psycopg[binary]==3.2.3 (psycopg 3) -- and the app
would crash on startup with a driver-not-found error the moment it's
pointed at a real managed database instead of docker-compose's own.
"""
import pytest

from app.config import Settings


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "postgres://user:pass@host.render.com:5432/dbname",
            "postgresql+psycopg://user:pass@host.render.com:5432/dbname",
        ),
        (
            "postgresql://user:pass@host.render.com:5432/dbname",
            "postgresql+psycopg://user:pass@host.render.com:5432/dbname",
        ),
        (
            "postgresql+psycopg://manas:manas@db:5432/manas",
            "postgresql+psycopg://manas:manas@db:5432/manas",
        ),
    ],
)
def test_database_url_rewritten_to_psycopg_driver(monkeypatch, raw, expected):
    monkeypatch.setenv("DATABASE_URL", raw)
    assert Settings().database_url == expected


def test_database_url_default_already_has_psycopg_driver(monkeypatch):
    # Same reasoning as test_cors_origins_default_is_localhost_list below:
    # docker-compose.yml (and Render) both set DATABASE_URL for the running
    # app, so this test must clear it to genuinely test the field default
    # rather than whatever happens to be in the ambient environment.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert Settings().database_url.startswith("postgresql+psycopg://")


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Render's Blueprint fromService/property:host cross-service
        # reference: a bare hostname, no scheme, no JSON brackets/quotes.
        ("manas-frontend.onrender.com", ["https://manas-frontend.onrender.com"]),
        # Already a full URL, no scheme rewrite needed.
        ("https://manas-frontend.onrender.com", ["https://manas-frontend.onrender.com"]),
        # The existing JSON-array-of-URLs shape (docker-compose.yml's usage,
        # and the field's own local-dev default) must still work unchanged.
        ('["http://localhost:5173","http://localhost:8080"]', ["http://localhost:5173", "http://localhost:8080"]),
    ],
)
def test_cors_origins_accepts_bare_hostname_or_json_array(monkeypatch, raw, expected):
    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert Settings().cors_origins == expected


def test_cors_origins_default_is_localhost_list(monkeypatch):
    # Explicitly clear CORS_ORIGINS: docker-compose.yml sets it for the
    # running app (and did for this test process too, when run inside that
    # container) to a non-default value, so this test must not assume the
    # ambient environment has it unset -- it's testing the *field default*,
    # not "whatever's in the environment right now".
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert Settings().cors_origins == [
        "http://localhost:5173", "http://localhost:3000", "http://localhost:8080",
    ]
