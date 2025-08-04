"""
Logging utilities for the Smart Product Classifier.
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Optional


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging if not already configured
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


def setup_logging(
    config_file: Optional[str] = None,
    log_level: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        config_file: Path to logging configuration file
        log_level: Override log level
    """
    # Default configuration
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': 'logs/app.log',
                'mode': 'a'
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': 'logs/error.log',
                'mode': 'a'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'api': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'model': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'training': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }
    
    # Use custom config file if provided
    if config_file and Path(config_file).exists():
        try:
            logging.config.fileConfig(config_file)
            return
        except Exception as e:
            print(f"Failed to load logging config from {config_file}: {e}")
            print("Using default configuration")
    
    # Apply log level override
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
        default_config['loggers']['']['level'] = level
        for handler in default_config['handlers'].values():
            if handler['level'] == 'INFO':
                handler['level'] = level
    
    # Apply configuration
    logging.config.dictConfig(default_config)


class ContextLogger:
    """Logger with additional context information."""
    
    def __init__(self, logger: logging.Logger, context: dict = None):
        self.logger = logger
        self.context = context or {}
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} | {context_str}"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), **kwargs)
    
    def with_context(self, **context) -> 'ContextLogger':
        """Create new logger with additional context."""
        new_context = {**self.context, **context}
        return ContextLogger(self.logger, new_context)