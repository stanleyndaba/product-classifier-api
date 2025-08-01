"""
Custom exception classes and exception handlers for the application.
"""

import logging
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class APIException(Exception):
    """
    Base API exception class.
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(APIException):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details
        )


class ModelException(APIException):
    """Exception raised for model-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_ERROR",
            details=details
        )


class ImageProcessingException(APIException):
    """Exception raised for image processing errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="IMAGE_PROCESSING_ERROR",
            details=details
        )


class RateLimitException(APIException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class FileUploadException(APIException):
    """Exception raised for file upload errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="FILE_UPLOAD_ERROR",
            details=details
        )


class ConfigurationException(APIException):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """
    Handle custom API exceptions.
    
    Args:
        request: The incoming request
        exc: The API exception
        
    Returns:
        JSONResponse: Error response
    """
    logger.error(
        f"API Exception: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "timestamp": request.state.start_time if hasattr(request.state, 'start_time') else None
            }
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTP exceptions.
    
    Args:
        request: The incoming request
        exc: The HTTP exception
        
    Returns:
        JSONResponse: Error response
    """
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "details": {},
                "timestamp": getattr(request.state, 'start_time', None)
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle request validation exceptions.
    
    Args:
        request: The incoming request
        exc: The validation exception
        
    Returns:
        JSONResponse: Error response
    """
    logger.warning(
        f"Validation Error: {exc.errors()}",
        extra={
            "errors": exc.errors(),
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "errors": exc.errors()
                },
                "timestamp": getattr(request.state, 'start_time', None)
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle general unexpected exceptions.
    
    Args:
        request: The incoming request
        exc: The exception
        
    Returns:
        JSONResponse: Error response
    """
    logger.error(
        f"Unexpected error: {type(exc).__name__} - {str(exc)}",
        extra={
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {},
                "timestamp": getattr(request.state, 'start_time', None)
            }
        }
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Setup exception handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        error_code: Error code identifier
        message: Error message
        status_code: HTTP status code
        details: Additional error details
        
    Returns:
        JSONResponse: Standardized error response
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "timestamp": None  # Will be set by middleware if available
            }
        }
    )