"""
Logging configuration and setup.
"""

import json
import logging
import logging.config
import sys
from datetime import datetime
from typing import Any, Dict

import structlog
from pythonjsonlogger import jsonlogger


class JSONFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """
        Add additional fields to log record.
        
        Args:
            log_record: Log record dictionary
            record: LogRecord instance
            message_dict: Message dictionary
        """
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add level
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
        
        # Add logger name
        log_record['logger'] = record.name
        
        # Add module information
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno


class TextFormatter(logging.Formatter):
    """
    Custom text formatter with colors and structured information.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors and structure.
        
        Args:
            record: LogRecord instance
            
        Returns:
            str: Formatted log message
        """
        # Add timestamp
        timestamp = datetime.utcnow().isoformat()
        
        # Add color for level
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Format message
        formatted_message = (
            f"{timestamp} | "
            f"{level_color}{record.levelname:8}{reset_color} | "
            f"{record.name:20} | "
            f"{record.module}:{record.funcName}:{record.lineno} | "
            f"{record.getMessage()}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted_message += f"\n{self.formatException(record.exc_info)}"
        
        return formatted_message


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Setup application logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or text)
    """
    # Convert log level to uppercase
    log_level = log_level.upper()
    
    # Create handlers
    handlers = []
    
    if log_format.lower() == "json":
        # JSON handler for structured logging
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setFormatter(JSONFormatter(
            format="%(timestamp)s %(level)s %(logger)s %(module)s %(function)s %(line)s %(message)s"
        ))
        handlers.append(json_handler)
    else:
        # Text handler for human-readable logging
        text_handler = logging.StreamHandler(sys.stdout)
        text_handler.setFormatter(TextFormatter())
        handlers.append(text_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=handlers,
        force=True
    )
    
    # Configure specific loggers
    configure_logger_levels(log_level)
    
    # Setup structlog
    setup_structlog(log_format.lower() == "json")


def configure_logger_levels(log_level: str) -> None:
    """
    Configure logging levels for specific loggers.
    
    Args:
        log_level: Base logging level
    """
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Application loggers
    logging.getLogger("app").setLevel(getattr(logging, log_level))
    
    # Model and inference loggers
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)


def setup_structlog(use_json: bool = True) -> None:
    """
    Setup structlog for structured logging.
    
    Args:
        use_json: Whether to use JSON format
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if use_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        structlog.stdlib.BoundLogger: Structured logger instance
    """
    return structlog.get_logger(name)


class RequestLogger:
    """
    Logger for HTTP requests with additional context.
    """
    
    def __init__(self, logger_name: str = "app.requests"):
        self.logger = get_logger(logger_name)
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_agent: str = None,
        ip_address: str = None,
        **kwargs
    ) -> None:
        """
        Log HTTP request with context.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration in seconds
            user_agent: User agent string
            ip_address: Client IP address
            **kwargs: Additional context
        """
        self.logger.info(
            "HTTP Request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            user_agent=user_agent,
            ip_address=ip_address,
            **kwargs
        )
    
    def log_error(
        self,
        method: str,
        path: str,
        error: Exception,
        status_code: int = 500,
        **kwargs
    ) -> None:
        """
        Log HTTP request error.
        
        Args:
            method: HTTP method
            path: Request path
            error: Exception that occurred
            status_code: Response status code
            **kwargs: Additional context
        """
        self.logger.error(
            "HTTP Request Error",
            method=method,
            path=path,
            status_code=status_code,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )


class ModelLogger:
    """
    Logger for model inference and operations.
    """
    
    def __init__(self, logger_name: str = "app.model"):
        self.logger = get_logger(logger_name)
    
    def log_inference(
        self,
        model_name: str,
        input_shape: tuple,
        inference_time: float,
        predictions: list,
        **kwargs
    ) -> None:
        """
        Log model inference.
        
        Args:
            model_name: Name of the model
            input_shape: Shape of input data
            inference_time: Inference time in seconds
            predictions: Model predictions
            **kwargs: Additional context
        """
        self.logger.info(
            "Model Inference",
            model_name=model_name,
            input_shape=input_shape,
            inference_time_ms=round(inference_time * 1000, 2),
            predictions_count=len(predictions),
            top_prediction=predictions[0] if predictions else None,
            **kwargs
        )
    
    def log_model_load(self, model_name: str, model_path: str, load_time: float) -> None:
        """
        Log model loading.
        
        Args:
            model_name: Name of the model
            model_path: Path to model file
            load_time: Time taken to load model
        """
        self.logger.info(
            "Model Loaded",
            model_name=model_name,
            model_path=model_path,
            load_time_ms=round(load_time * 1000, 2)
        )