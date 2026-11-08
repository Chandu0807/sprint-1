"""db.py - database engine and the per-request session."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from . import config

# SQLite (optional quick mode) needs this flag because the API uses several threads
_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, connect_args=_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
