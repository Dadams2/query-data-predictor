
import logging.config
import sys

def setup_logging(log_level="INFO"):
    """
    Set up logging configuration for the application.
    """
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "default",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": "query_data_predictor.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "query_data_predictor": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False, # Do not pass messages to the root logger
            },
            # Quieten noisy third-party libraries
            "matplotlib": {
                "handlers": ["console", "file"],
                "level": "WARNING",
                "propagate": False,
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)

