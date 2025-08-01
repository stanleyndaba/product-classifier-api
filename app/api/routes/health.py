"""
Health check API routes for monitoring and status.
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from ...schemas.prediction import HealthResponse
from ...services.model_service import ModelService
from ...utils.dependencies import get_model_service

router = APIRouter()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and model service."
)
async def health_check(
    model_service: ModelService = Depends(get_model_service)
) -> HealthResponse:
    """
    Perform health check of the API service.
    
    Returns basic health information including:
    - Service status
    - Model loading status
    - Service uptime
    - API version
    
    Returns:
        HealthResponse: Health status information
    """
    try:
        # Get model service health
        model_health = await model_service.health_check()
        
        # Determine overall status
        healthy = model_health.get("healthy", False)
        status = "healthy" if healthy else "unhealthy"
        
        return HealthResponse(
            status=status,
            healthy=healthy,
            model_loaded=model_health.get("model_loaded", False),
            uptime=model_health.get("uptime", 0),
            version="1.0.0",
            timestamp=time.time()
        )
        
    except Exception as e:
        # Return unhealthy status if health check fails
        return HealthResponse(
            status="unhealthy",
            healthy=False,
            model_loaded=False,
            uptime=0,
            version="1.0.0",
            timestamp=time.time()
        )


@router.get(
    "/detailed",
    response_model=Dict[str, Any],
    summary="Detailed Health Check",
    description="Get detailed health information including model statistics."
)
async def detailed_health_check(
    model_service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Get detailed health information.
    
    Provides comprehensive health status including:
    - Model service details
    - Performance statistics
    - Resource usage
    - Configuration status
    
    Returns:
        Dict[str, Any]: Detailed health information
    """
    try:
        health_info = await model_service.health_check()
        
        return {
            "status": "healthy" if health_info.get("healthy") else "unhealthy",
            "timestamp": time.time(),
            "service": {
                "name": "Smart Product Classifier API",
                "version": "1.0.0",
                "healthy": health_info.get("healthy", False)
            },
            "model": {
                "loaded": health_info.get("model_loaded", False),
                "info": health_info.get("model_info", {}),
                "statistics": health_info.get("statistics", {})
            },
            "uptime": health_info.get("uptime", 0),
            "last_health_check": health_info.get("last_health_check", 0)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Health check failed",
                "message": str(e)
            }
        )


@router.get(
    "/ready",
    response_model=Dict[str, Any],
    summary="Readiness Check",
    description="Check if the service is ready to accept requests."
)
async def readiness_check(
    model_service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Check if service is ready to accept requests.
    
    This endpoint is useful for Kubernetes readiness probes
    and load balancer health checks.
    
    Returns:
        Dict[str, Any]: Readiness status
    """
    try:
        is_ready = model_service.is_ready()
        
        if not is_ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "ready": False,
                    "message": "Service not ready"
                }
            )
        
        return {
            "ready": True,
            "message": "Service is ready",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "message": f"Readiness check failed: {str(e)}"
            }
        )


@router.get(
    "/live",
    response_model=Dict[str, Any],
    summary="Liveness Check",
    description="Check if the service is alive and running."
)
async def liveness_check() -> Dict[str, Any]:
    """
    Simple liveness check.
    
    This endpoint always returns success if the service is running.
    Useful for Kubernetes liveness probes.
    
    Returns:
        Dict[str, Any]: Liveness status
    """
    return {
        "alive": True,
        "message": "Service is alive",
        "timestamp": time.time()
    }