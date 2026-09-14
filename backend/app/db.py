"""SQLAlchemy engine/session setup."""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def new_session():
    """Returns a session bound to the *current* SessionLocal factory.

    Code outside FastAPI's request/dependency cycle (the request-logging
    middleware, the background batch worker) can't use `Depends(get_db)`, so
    it calls this instead. Tests monkeypatch `app.db.SessionLocal` to point
    at an isolated in-memory DB; because this looks up SessionLocal at call
    time (module attribute), the override takes effect everywhere that calls
    new_session() -- unlike a `from app.db import SessionLocal` binding taken
    at import time, which would freeze in the original Postgres-pointed
    factory and try to hit a real database in every test.
    """
    return SessionLocal()
