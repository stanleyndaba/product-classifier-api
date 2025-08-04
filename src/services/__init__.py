"""
Services package for the Smart Product Classifier API.
Contains FastAPI endpoints, schemas, and middleware.
"""

from .api import app
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)

__all__ = [
    "app",
    "PredictionRequest",
    "PredictionResponse", 
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse"
]