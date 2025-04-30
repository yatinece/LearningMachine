import configparser
from datetime import datetime
import logging
import logging.config
from .config import create_config

def read_config(file_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(file_path)

    return {
        'debug': config.getboolean('General', 'debug'),
        'log_level': config.get('General', 'log_level').upper(),
        'log_file_path': config.get('General', 'log_file_path', fallback='app.log'),
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


if __name__ == "__main__":
    create_config()
    config = read_config()
    logger = setup_logging(config['log_level'], config['log_file_path'], config['project_name'])

    logger.info(f"Logger is using {config['project_name']}")