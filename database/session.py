"""
Database connection manager and session lifecycle handling.
Defaults to local SQLite, automatically switches to PostgreSQL when DATABASE_URL is set.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fer_sessions.db")

# Configure engine arguments based on dialect
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes tables in database."""
    logger.info("Initializing database tables on %s", DATABASE_URL)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI/Flask database dependency generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
