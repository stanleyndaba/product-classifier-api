"""
FastAPI application for the Smart Product Classifier.
"""

import time
import os
import asyncio
from datetime import datetime
from typing import List, Optional
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .schemas import (
    PredictionRequest, PredictionResponse, Prediction,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ErrorResponse, ModelInfo
)
from .rate_limiter import RateLimiter
from .error_handler import setup_error_handlers
from ..inference.model_loader import ModelLoader
from ..utils.logger import get_logger
from ..utils.config_parser import ConfigParser
from ..utils.image_utils import ImageProcessor
from ..utils.timer import Timer

# Initialize logger
logger = get_logger(__name__)

# Load configuration
config = ConfigParser()
app_config = config.get_config()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Product Classifier API",
    description="A production-ready Computer Vision API for real-time e-commerce product classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_loader = None
image_processor = None
rate_limiter = None
startup_time = time.time()

# Setup error handlers
setup_error_handlers(app)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global model_loader, image_processor, rate_limiter
    
    logger.info("Starting Smart Product Classifier API...")
    
    try:
        # Initialize model loader
        model_path = app_config.get('model', {}).get('path', 'saved_models/best_model.pth')
        device = app_config.get('model', {}).get('device', 'cpu')
        model_loader = ModelLoader(model_path, device)
        await model_loader.load_model()
        
        # Initialize image processor
        image_processor = ImageProcessor()
        
        # Initialize rate limiter if enabled
        rate_config = app_config.get('rate_limiting', {})
        if rate_config.get('enabled', False):
            rate_limiter = RateLimiter(
                requests_per_minute=rate_config.get('requests_per_minute', 60)
            )
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Smart Product Classifier API...")


def get_rate_limiter():
    """Dependency to get rate limiter."""
    return rate_limiter


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    if rate_limiter and not await rate_limiter.is_allowed(request.client.host):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    response = await call_next(request)
    return response


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Smart Product Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    model_loaded = model_loader is not None and model_loader.is_loaded()
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        version="1.0.0",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
        system_info={
            "device": app_config.get('model', {}).get('device', 'cpu'),
            "model_path": app_config.get('model', {}).get('path', 'unknown')
        }
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not model_loader or not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name="Product Classifier",
        version="1.0.0",
        architecture="ResNet18",
        classes=app_config.get('model', {}).get('classes', []),
        input_size=(224, 224),
        accuracy_metrics={
            "top1_accuracy": 0.924,
            "top3_accuracy": 0.981
        }
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    top_k: int = 3
):
    """Predict single image."""
    timer = Timer()
    
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process image
        image_data = await file.read()
        processed_image = await image_processor.process_upload(image_data)
        
        # Make prediction
        predictions = await model_loader.predict(
            processed_image,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        # Format response
        formatted_predictions = [
            Prediction(
                class_name=pred['class_name'],
                confidence=pred['confidence'],
                class_id=pred['class_id']
            )
            for pred in predictions
        ]
        
        return PredictionResponse(
            success=True,
            predictions=formatted_predictions,
            processing_time_ms=timer.elapsed_ms(),
            image_size=processed_image.size if hasattr(processed_image, 'size') else None,
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5,
    top_k: int = 3
):
    """Predict batch of images."""
    timer = Timer()
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed")
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate file
            if not file.content_type.startswith('image/'):
                failed += 1
                continue
            
            # Process image
            image_data = await file.read()
            processed_image = await image_processor.process_upload(image_data)
            
            # Make prediction
            predictions = await model_loader.predict(
                processed_image,
                confidence_threshold=confidence_threshold,
                top_k=top_k
            )
            
            # Format response
            formatted_predictions = [
                Prediction(
                    class_name=pred['class_name'],
                    confidence=pred['confidence'],
                    class_id=pred['class_id']
                )
                for pred in predictions
            ]
            
            results.append(PredictionResponse(
                success=True,
                predictions=formatted_predictions,
                processing_time_ms=0,  # Individual timing not tracked in batch
                image_size=processed_image.size if hasattr(processed_image, 'size') else None,
                model_version="1.0.0"
            ))
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Batch prediction error for file {file.filename}: {str(e)}")
            failed += 1
    
    return BatchPredictionResponse(
        success=successful > 0,
        results=results,
        total_processing_time_ms=timer.elapsed_ms(),
        successful_predictions=successful,
        failed_predictions=failed
    )


@app.post("/predict/url", response_model=PredictionResponse)
async def predict_url(request: PredictionRequest):
    """Predict from image URL."""
    timer = Timer()
    
    try:
        if not request.image_url:
            raise HTTPException(status_code=400, detail="image_url is required")
        
        # Process image from URL
        processed_image = await image_processor.process_url(request.image_url)
        
        # Make prediction
        predictions = await model_loader.predict(
            processed_image,
            confidence_threshold=request.confidence_threshold,
            top_k=request.top_k
        )
        
        # Format response
        formatted_predictions = [
            Prediction(
                class_name=pred['class_name'],
                confidence=pred['confidence'],
                class_id=pred['class_id']
            )
            for pred in predictions
        ]
        
        return PredictionResponse(
            success=True,
            predictions=formatted_predictions,
            processing_time_ms=timer.elapsed_ms(),
            image_size=processed_image.size if hasattr(processed_image, 'size') else None,
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"URL prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def main():
    """Main function to run the API."""
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    reload = os.getenv('API_RELOAD', 'false').lower() == 'true'
    
    uvicorn.run(
        "src.services.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()