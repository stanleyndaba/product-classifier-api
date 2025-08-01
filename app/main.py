"""
Main FastAPI application entry point.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .api.routes import prediction, health, batch
from .core.config import get_settings
from .core.exceptions import setup_exception_handlers
from .core.logging import setup_logging
from .core.middleware import setup_middleware
from .services.model_service import ModelService
from .utils.rate_limiter import RateLimiter

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


async def startup_event():
    """
    Initialize application resources on startup.
    """
    logger.info("Starting Smart Product Classifier API...")
    
    # Initialize model service
    model_service = ModelService()
    await model_service.load_model()
    app.state.model_service = model_service
    
    # Initialize rate limiter
    rate_limiter = RateLimiter()
    app.state.rate_limiter = rate_limiter
    
    logger.info("Application startup complete.")


async def shutdown_event():
    """
    Clean up application resources on shutdown.
    """
    logger.info("Shutting down Smart Product Classifier API...")
    
    # Clean up model resources
    if hasattr(app.state, 'model_service'):
        await app.state.model_service.cleanup()
    
    logger.info("Application shutdown complete.")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings.log_level)
    
    # Create FastAPI app
    app = FastAPI(
        title="Smart Product Classifier API",
        description="A production-ready Computer Vision API for real-time e-commerce product classification",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app, settings)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
    app.include_router(batch.router, prefix="/api/v1", tags=["Batch Processing"])
    
    return app


# Create the application instance
app = create_application()


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        dict: API information and status
    """
    return {
        "message": "Smart Product Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get application metrics and statistics.
    
    Returns:
        dict: Application metrics
    """
    # TODO: Implement metrics collection
    return {
        "requests_total": 0,
        "requests_per_second": 0,
        "average_response_time": 0,
        "model_inference_time": 0,
        "active_connections": 0
    }


def run_server():
    """
    Run the FastAPI server with uvicorn.
    """
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    run_server()