"""
Dependency injection utilities for FastAPI.
"""

from functools import lru_cache
from typing import Generator

from fastapi import Depends

from ..services.model_service import ModelService
from .file_handler import FileHandler
from .rate_limiter import RateLimiter


# Global instances (singleton pattern)
_model_service: ModelService = None
_file_handler: FileHandler = None
_rate_limiter: RateLimiter = None


@lru_cache()
def get_model_service() -> ModelService:
    """
    Get or create model service instance.
    
    Returns:
        ModelService: Singleton model service instance
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


@lru_cache()
def get_file_handler() -> FileHandler:
    """
    Get or create file handler instance.
    
    Returns:
        FileHandler: Singleton file handler instance
    """
    global _file_handler
    if _file_handler is None:
        _file_handler = FileHandler()
    return _file_handler


@lru_cache()
def get_rate_limiter() -> RateLimiter:
    """
    Get or create rate limiter instance.
    
    Returns:
        RateLimiter: Singleton rate limiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter