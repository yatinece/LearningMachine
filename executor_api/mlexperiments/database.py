import logging
logger = logging.getLogger(__name__)
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .logger import get_config
import configparser
import os

# Use SQLite instead of PostgreSQL
# This will create a SQLite database file in the current directory


from .logger import get_config
config = get_config()

DATABASE_URL = config.get('DATABASE_URL')

# Create database engine with SQLite support
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()
def get_db():
    db = SessionLocal()
    try:
        logger.info(f"Connected to DB at: {db.bind.url}")
        yield db
    finally:
        db.close()