"""
Pydantic schemas for prediction requests and responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class Prediction(BaseModel):
    """
    Individual prediction result.
    """
    category: str = Field(..., description="Predicted product category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    category_id: int = Field(..., ge=0, description="Category ID number")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "electronics",
                "confidence": 0.8542,
                "category_id": 0
            }
        }


class Metadata(BaseModel):
    """
    Prediction metadata.
    """
    inference_time_ms: float = Field(..., ge=0, description="Inference time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    categories_count: int = Field(..., ge=1, description="Total number of categories")
    timestamp: float = Field(..., description="Unix timestamp of prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "inference_time_ms": 156.7,
                "model_version": "1.0.0",
                "categories_count": 12,
                "timestamp": 1635724800.0
            }
        }


class BatchMetadata(BaseModel):
    """
    Batch prediction metadata.
    """
    batch_size: int = Field(..., ge=1, description="Number of images in batch")
    total_inference_time_ms: float = Field(..., ge=0, description="Total batch processing time")
    average_time_per_image_ms: float = Field(..., ge=0, description="Average time per image")
    model_version: str = Field(..., description="Model version used")
    timestamp: float = Field(..., description="Unix timestamp of batch completion")
    files: Optional[List[Dict[str, Any]]] = Field(None, description="File information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_size": 5,
                "total_inference_time_ms": 623.4,
                "average_time_per_image_ms": 124.7,
                "model_version": "1.0.0",
                "timestamp": 1635724800.0,
                "files": [
                    {
                        "filename": "product1.jpg",
                        "content_type": "image/jpeg",
                        "size": 1024000
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for single image prediction.
    """
    success: bool = Field(..., description="Whether the prediction was successful")
    predictions: List[Prediction] = Field(..., description="List of predictions")
    metadata: Metadata = Field(..., description="Prediction metadata")
    
    @validator('predictions')
    def validate_predictions(cls, v):
        if not v:
            raise ValueError("At least one prediction must be returned")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {
                        "category": "electronics",
                        "confidence": 0.8542,
                        "category_id": 0
                    },
                    {
                        "category": "furniture",
                        "confidence": 0.1234,
                        "category_id": 2
                    }
                ],
                "metadata": {
                    "inference_time_ms": 156.7,
                    "model_version": "1.0.0",
                    "categories_count": 12,
                    "timestamp": 1635724800.0
                }
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch image prediction.
    """
    success: bool = Field(..., description="Whether the batch prediction was successful")
    predictions: List[List[Prediction]] = Field(..., description="List of predictions for each image")
    metadata: BatchMetadata = Field(..., description="Batch prediction metadata")
    
    @validator('predictions')
    def validate_batch_predictions(cls, v):
        if not v:
            raise ValueError("At least one prediction batch must be returned")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    [
                        {
                            "category": "electronics",
                            "confidence": 0.8542,
                            "category_id": 0
                        }
                    ],
                    [
                        {
                            "category": "clothing",
                            "confidence": 0.7231,
                            "category_id": 1
                        }
                    ]
                ],
                "metadata": {
                    "batch_size": 2,
                    "total_inference_time_ms": 298.4,
                    "average_time_per_image_ms": 149.2,
                    "model_version": "1.0.0",
                    "timestamp": 1635724800.0
                }
            }
        }


class PredictionRequest(BaseModel):
    """
    Request model for predictions (when using JSON instead of file upload).
    """
    image_url: Optional[str] = Field(None, description="URL of image to classify")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional prediction options"
    )
    
    @validator('image_url', 'image_base64')
    def validate_image_input(cls, v, values):
        image_url = values.get('image_url')
        image_base64 = values.get('image_base64')
        
        if not image_url and not image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")
        
        if image_url and image_base64:
            raise ValueError("Only one of image_url or image_base64 should be provided")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/product.jpg",
                "options": {
                    "confidence_threshold": 0.1,
                    "max_predictions": 3
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions.
    """
    file_ids: List[str] = Field(..., description="List of uploaded file IDs to process")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional batch processing options"
    )
    
    @validator('file_ids')
    def validate_file_ids(cls, v):
        if not v:
            raise ValueError("At least one file ID must be provided")
        if len(v) > 20:
            raise ValueError("Maximum 20 file IDs allowed per batch")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_ids": ["file_1635724800_0", "file_1635724800_1"],
                "options": {
                    "confidence_threshold": 0.1,
                    "priority": "normal"
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response model.
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: Optional[float] = Field(None, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid image format",
                "details": {
                    "supported_formats": ["JPEG", "PNG", "WebP"]
                },
                "timestamp": 1635724800.0
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str = Field(..., description="Health status")
    healthy: bool = Field(..., description="Whether service is healthy")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="API version")
    timestamp: float = Field(..., description="Health check timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "healthy": True,
                "model_loaded": True,
                "uptime": 3600.0,
                "version": "1.0.0",
                "timestamp": 1635724800.0
            }
        }


class CategoryInfo(BaseModel):
    """
    Category information model.
    """
    categories: List[str] = Field(..., description="List of category names")
    total_categories: int = Field(..., description="Total number of categories")
    category_mapping: Dict[str, int] = Field(..., description="Category name to ID mapping")
    
    class Config:
        json_schema_extra = {
            "example": {
                "categories": ["electronics", "clothing", "furniture"],
                "total_categories": 12,
                "category_mapping": {
                    "electronics": 0,
                    "clothing": 1,
                    "furniture": 2
                }
            }
        }


class ModelInfo(BaseModel):
    """
    Model information response.
    """
    model_path: str = Field(..., description="Path to model file")
    device: str = Field(..., description="Device model is running on")
    is_loaded: bool = Field(..., description="Whether model is loaded")
    load_time: Optional[float] = Field(None, description="Model load time in seconds")
    categories: List[str] = Field(..., description="Supported categories")
    image_dimensions: List[int] = Field(..., description="Expected image dimensions")
    confidence_threshold: float = Field(..., description="Confidence threshold")
    max_predictions: int = Field(..., description="Maximum predictions returned")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "models/product_classifier.pt",
                "device": "cpu",
                "is_loaded": True,
                "load_time": 2.34,
                "categories": ["electronics", "clothing"],
                "image_dimensions": [224, 224],
                "confidence_threshold": 0.1,
                "max_predictions": 3
            }
        }


class BenchmarkResult(BaseModel):
    """
    Model benchmark result.
    """
    iterations: int = Field(..., description="Number of benchmark iterations")
    average_time: float = Field(..., description="Average inference time in seconds")
    min_time: float = Field(..., description="Minimum inference time")
    max_time: float = Field(..., description="Maximum inference time")
    p50_time: float = Field(..., description="50th percentile inference time")
    p95_time: float = Field(..., description="95th percentile inference time")
    p99_time: float = Field(..., description="99th percentile inference time")
    throughput_fps: float = Field(..., description="Throughput in frames per second")
    total_time: float = Field(..., description="Total benchmark time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "iterations": 100,
                "average_time": 0.156,
                "min_time": 0.142,
                "max_time": 0.201,
                "p50_time": 0.154,
                "p95_time": 0.178,
                "p99_time": 0.195,
                "throughput_fps": 6.41,
                "total_time": 15.6
            }
        }