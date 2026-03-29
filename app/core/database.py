"""
database.py
-----------
Database connection setup for the Forex Market Intelligence SaaS.

Uses SQLAlchemy ORM with PostgreSQL.  The connection string is read from
the ``DATABASE_URL`` environment variable — no credentials are hardcoded.

Usage
-----
>>> from app.core.database import SessionLocal, engine, Base
>>> Base.metadata.create_all(bind=engine)      # create tables
>>> with SessionLocal() as session:
...     session.add(record)
...     session.commit()
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL is None:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Example: postgresql://user:pass@localhost:5432/forex_saas"
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_session():
    """Yield a SQLAlchemy session and ensure it is closed afterwards."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
