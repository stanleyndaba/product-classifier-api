"""
Prediction API routes for single image classification.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ...core.exceptions import ModelException, ValidationException, FileUploadException
from ...schemas.prediction import PredictionResponse, PredictionRequest, ErrorResponse
from ...services.model_service import ModelService
from ...utils.file_handler import FileHandler
from ...utils.dependencies import get_model_service, get_file_handler

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    },
    summary="Classify Product Image",
    description="Upload an image and get product category predictions with confidence scores."
)
async def predict_image(
    request: Request,
    file: UploadFile = File(..., description="Product image file (JPEG, PNG, WebP)"),
    model_service: ModelService = Depends(get_model_service),
    file_handler: FileHandler = Depends(get_file_handler)
) -> PredictionResponse:
    """
    Classify a product image and return top predictions.
    
    This endpoint accepts an image file and returns the top product category 
    predictions with confidence scores. The model supports 12 product categories:
    electronics, clothing, furniture, books, toys, sports, beauty, automotive, 
    home, garden, jewelry, and food.
    
    Args:
        file: Image file to classify (max 10MB)
        
    Returns:
        PredictionResponse: Predictions with confidence scores and metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate file upload
        await file_handler.validate_image_file(file)
        
        # Read image data
        image_data = await file_handler.read_image_file(file)
        
        # Make prediction
        result = await model_service.predict_single(image_data)
        
        # Create response
        response = PredictionResponse(
            predictions=result["predictions"],
            metadata=result["metadata"],
            success=True
        )
        
        return response
        
    except FileUploadException as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "File upload error",
                "message": e.message,
                "details": e.details
            }
        )
    
    except ValidationException as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation error", 
                "message": e.message,
                "details": e.details
            }
        )
    
    except ModelException as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model service error",
                "message": e.message,
                "details": e.details
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "details": {"error": str(e)}
            }
        )


@router.post(
    "/predict/timing",
    response_model=Dict[str, Any],
    summary="Classify with Detailed Timing",
    description="Get predictions with detailed timing breakdown for performance analysis."
)
async def predict_with_timing(
    request: Request,
    file: UploadFile = File(..., description="Product image file"),
    model_service: ModelService = Depends(get_model_service),
    file_handler: FileHandler = Depends(get_file_handler)
) -> Dict[str, Any]:
    """
    Classify image with detailed timing information.
    
    This endpoint provides the same predictions as /predict but includes
    detailed timing breakdown for preprocessing, inference, and postprocessing.
    Useful for performance analysis and optimization.
    
    Args:
        file: Image file to classify
        
    Returns:
        Dict[str, Any]: Predictions with detailed timing breakdown
    """
    try:
        # Validate and read file
        await file_handler.validate_image_file(file)
        image_data = await file_handler.read_image_file(file)
        
        # Make prediction with timing
        result = await model_service.predict_with_timing(image_data)
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e)
            }
        )


@router.get(
    "/categories",
    response_model=Dict[str, Any],
    summary="Get Supported Categories",
    description="Get list of all supported product categories."
)
async def get_categories(
    model_service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Get supported product categories.
    
    Returns the list of all product categories that the model can predict,
    along with category mappings and metadata.
    
    Returns:
        Dict[str, Any]: Category information
    """
    try:
        category_info = model_service.get_category_info()
        
        return {
            "success": True,
            **category_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get categories",
                "message": str(e)
            }
        )


@router.get(
    "/model/info",
    response_model=Dict[str, Any],
    summary="Get Model Information",
    description="Get detailed information about the loaded model."
)
async def get_model_info(
    model_service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Get model information and statistics.
    
    Returns detailed information about the loaded model including
    configuration, performance statistics, and health status.
    
    Returns:
        Dict[str, Any]: Model information and statistics
    """
    try:
        model_info = await model_service.get_model_info()
        
        return {
            "success": True,
            **model_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get model info",
                "message": str(e)
            }
        )


@router.post(
    "/model/benchmark",
    response_model=Dict[str, Any],
    summary="Benchmark Model Performance",
    description="Run performance benchmark on the model."
)
async def benchmark_model(
    request: Request,
    iterations: int = 100,
    model_service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Benchmark model performance.
    
    Runs a series of dummy predictions to measure model performance
    including throughput, latency percentiles, and other metrics.
    
    Args:
        iterations: Number of benchmark iterations (default: 100)
        
    Returns:
        Dict[str, Any]: Benchmark results and performance metrics
    """
    try:
        if not 1 <= iterations <= 1000:
            raise HTTPException(
                status_code=400,
                detail="Iterations must be between 1 and 1000"
            )
        
        benchmark_results = await model_service.benchmark_model(iterations)
        
        return {
            "success": True,
            "benchmark_results": benchmark_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Benchmark failed",
                "message": str(e)
            }
        )