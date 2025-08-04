"""
Unified error handling for the Smart Product Classifier API.
"""

import traceback
from typing import Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .schemas import ErrorResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)


def setup_error_handlers(app: FastAPI):
    """Setup global error handlers for the FastAPI application."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        error_response = ErrorResponse(
            error="HTTPException",
            message=str(exc.detail),
            detail=f"Status code: {exc.status_code}"
        )
        
        logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        error_response = ErrorResponse(
            error="StarletteHTTPException",
            message=str(exc.detail),
            detail=f"Status code: {exc.status_code}"
        )
        
        logger.warning(f"Starlette HTTP {exc.status_code}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        error_response = ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            detail=str(error_details)
        )
        
        logger.warning(f"Validation error: {error_details}")
        
        return JSONResponse(
            status_code=422,
            content=error_response.dict()
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors."""
        error_response = ErrorResponse(
            error="ValueError",
            message=str(exc),
            detail="Invalid input value provided"
        )
        
        logger.error(f"Value error: {str(exc)}")
        
        return JSONResponse(
            status_code=400,
            content=error_response.dict()
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle file not found errors."""
        error_response = ErrorResponse(
            error="FileNotFoundError",
            message="Required file not found",
            detail=str(exc)
        )
        
        logger.error(f"File not found: {str(exc)}")
        
        return JSONResponse(
            status_code=404,
            content=error_response.dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        error_response = ErrorResponse(
            error=exc.__class__.__name__,
            message="An unexpected error occurred",
            detail=str(exc)
        )
        
        # Log the full traceback for debugging
        logger.error(f"Unexpected error: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


class APIError(Exception):
    """Custom API error class."""
    
    def __init__(self, message: str, status_code: int = 500, detail: str = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class ModelError(APIError):
    """Model-related errors."""
    
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status_code=503, detail=detail)


class ImageProcessingError(APIError):
    """Image processing errors."""
    
    def __init__(self, message: str, detail: str = None):
        super().__init__(message, status_code=400, detail=detail)


class RateLimitError(APIError):
    """Rate limiting errors."""
    
    def __init__(self, message: str = "Rate limit exceeded", detail: str = None):
        super().__init__(message, status_code=429, detail=detail)