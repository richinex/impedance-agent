# src/core/logging.py
import logging
import logging.config  # Add this import
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
import yaml

def setup_logging(
    level: str = "INFO",
    config_path: Optional[Path] = None
) -> None:
    """Configure logging with optional configuration file"""

    # Create basic logger first
    logger = logging.getLogger()
    if not logger.handlers:  # Only add handler if none exists
        handler = RichHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    # Default configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'rich.logging.RichHandler',
                'formatter': 'standard',
                'level': level,
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'impedance_analysis.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'standard',
                'level': level,
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': level,
                'propagate': True
            }
        }
    }

    try:
        # Load custom config if provided
        if config_path:
            with open(config_path) as f:
                config.update(yaml.safe_load(f))

        # Configure logging
        logging.config.dictConfig(config)
    except Exception as e:
        logger.warning(f"Could not configure advanced logging: {str(e)}. Using basic logging instead.")