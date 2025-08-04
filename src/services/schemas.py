"""
Pydantic models for request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single image prediction."""
    
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image file")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of top predictions to return")
    
    @validator('image_data', 'image_url')
    def validate_image_input(cls, v, values):
        if not values.get('image_data') and not v:
            raise ValueError('Either image_data or image_url must be provided')
        return v


class Prediction(BaseModel):
    """Single prediction result."""
    
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    class_id: int = Field(..., ge=0, description="Class index")


class PredictionResponse(BaseModel):
    """Response model for single image prediction."""
    
    success: bool = Field(..., description="Whether prediction was successful")
    predictions: List[Prediction] = Field(..., description="List of predictions ordered by confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    image_size: Optional[tuple] = Field(None, description="Original image dimensions (width, height)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    model_version: Optional[str] = Field(None, description="Model version used for prediction")


class BatchPredictionRequest(BaseModel):
    """Request model for batch image prediction."""
    
    images: List[Dict[str, Any]] = Field(..., min_items=1, max_items=20, description="List of image data")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of top predictions per image")
    
    @validator('images')
    def validate_images(cls, v):
        if len(v) > 20:
            raise ValueError('Maximum 20 images allowed per batch')
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch image prediction."""
    
    success: bool = Field(..., description="Whether batch prediction was successful")
    results: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    system_info: Optional[Dict[str, Any]] = Field(None, description="System information")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    architecture: str = Field(..., description="Model architecture")
    classes: List[str] = Field(..., description="List of class names")
    input_size: tuple = Field(..., description="Expected input image size (width, height)")
    accuracy_metrics: Optional[Dict[str, float]] = Field(None, description="Model accuracy metrics")
    training_date: Optional[datetime] = Field(None, description="Model training date")


class TrainingStatus(BaseModel):
    """Training job status response."""
    
    job_id: str = Field(..., description="Training job identifier")
    status: str = Field(..., description="Training status (pending, running, completed, failed)")
    progress: float = Field(..., ge=0.0, le=1.0, description="Training progress (0-1)")
    current_epoch: Optional[int] = Field(None, description="Current training epoch")
    total_epochs: Optional[int] = Field(None, description="Total training epochs")
    loss: Optional[float] = Field(None, description="Current training loss")
    accuracy: Optional[float] = Field(None, description="Current validation accuracy")
    eta_seconds: Optional[float] = Field(None, description="Estimated time to completion")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")