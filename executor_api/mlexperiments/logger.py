# Create an improved_logger.py file

import logging
import logging.config
import os
import configparser
from pathlib import Path

def get_config(file_path='./config.ini'):
    """Read configuration from config.ini file"""
    config = configparser.ConfigParser()
    
    # Check if the config file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    config.read(file_path)
    
    return {
        'debug': config.getboolean('General', 'debug', fallback=True),
        'log_level': config.get('General', 'log_level', fallback='INFO').upper(),
        'log_file_path': config.get('General', 'log_file_path', fallback='app.log'),
        'project_name': config.get('General', 'project_name', fallback='learn_test'),
        'DATABASE_URL': config.get('General', 'database_url', fallback='sqlite:///./project_management.db'),
        'OPTUNA_URL': config.get('General', 'optuna_url', fallback='sqlite:///./optuna_studies.db')
    }

def ensure_log_directory(log_file_path):
    """Ensure the log directory exists"""
    log_dir = os.path.dirname(log_file_path)
    if log_dir:  # Skip if the path doesn't have a directory component
        os.makedirs(log_dir, exist_ok=True)
    return log_file_path

def setup_logging(config=None):
    """
    Setup logging configuration for the entire application.
    
    Args:
        config: Dictionary with configuration options
               - log_level: The logging level (INFO, DEBUG, etc.)
               - log_file_path: Path to the log file
               - project_name: Name of the project/application
    
    Returns:
        The configured root logger
    """
    if config is None:
        config = get_config()
    
    log_level = config.get('log_level', 'INFO')
    log_file_path = config.get('log_file_path', 'app.log')
    project_name = config.get('project_name', 'learn_test')
    
    # Ensure the log directory exists
    ensure_log_directory(log_file_path)
    
    # Check if we can create/write to the log file
    try:
        with open(log_file_path, 'a') as f:
            f.write('')  # Test write access
    except Exception as e:
        print(f"WARNING: Could not write to log file {log_file_path}: {e}")
        print("Falling back to a default log file in the current directory")
        log_file_path = 'app.log'
    
    # Define the logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,  # This is important!
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
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
            project_name: {  # Project logger
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    
    # Apply the configuration
    logging.config.dictConfig(logging_config)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.info(f"Logging initialized. Project: {project_name}, Log level: {log_level}, Log file: {log_file_path}")
    
    return logger

if __name__ == "__main__":
    # Test the logging configuration
    try:
        config = get_config()
        logger = setup_logging(config)
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
    except Exception as e:
        print(f"Error setting up logging: {e}")