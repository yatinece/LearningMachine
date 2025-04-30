import configparser
from fastapi import FastAPI, Form
from pydantic import BaseModel
import logging
import pandas as pd
from datetime import datetime
import configparser
import logging
import logging.config
import os
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base

def read_config(file_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(file_path)

    return {
        'debug': config.getboolean('General', 'debug'),
        'log_level': config.get('General', 'log_level').upper(),
        'log_file_path': config.get('General', 'log_file_path', fallback='app.log'),
        'db_name': config.get('Database', 'db_name'),
        'db_host': config.get('Database', 'db_host'),
        'db_port': config.getint('Database', 'db_port'),
        'user_name': config.get('Project', 'user_name'),
        'project_name' : config.get('Project', 'project_name'),
    }


# -----------------------
# Logging configuration
# -----------------------

def setup_logging(log_level='INFO', log_file_path='app.log' , project_name ='learn_test'):
    logging_config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',  # Always capture full logs in file
                'formatter': 'standard',
                'filename': log_file_path,
                'mode': 'a',
            },
        },
        'loggers': {
            project_name: {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    }

    logging.config.dictConfig(logging_config)
    return logging.getLogger(__name__)

# -----------------------
# FastAPI app initialization
# -----------------------
app = FastAPI(title="Data Pipeline API")

# Database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./ml_sessions.db")
Base = declarative_base()
engine = sa.create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


if __name__ == "__main__":
    config = read_config()
    logger = setup_logging(config['log_level'], config['log_file_path'], config['project_name'])

    logger.info("Logger is using project name!")