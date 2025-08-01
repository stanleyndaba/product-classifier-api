"""
Model service for managing ML model lifecycle and predictions.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..core.config import get_model_config, get_settings
from ..core.exceptions import ModelException, ConfigurationException
from ..core.logging import ModelLogger
from ..models.base import ModelConfig, ModelRegistry
from ..models.classifier import ProductClassifier


class ModelService:
    """
    Service for managing machine learning models and predictions.
    
    Handles model loading, caching, health checks, and prediction routing.
    """
    
    def __init__(self):
        """Initialize model service."""
        self.settings = get_settings()
        self.model_config = get_model_config()
        self.logger = ModelLogger()
        
        # Model management
        self.registry = ModelRegistry()
        self.primary_model: Optional[ProductClassifier] = None
        self._is_healthy = False
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds
        
        # Performance tracking
        self._prediction_count = 0
        self._total_inference_time = 0.0
        self._last_prediction_time = None
    
    async def load_model(self) -> None:
        """
        Load the primary classification model.
        
        Raises:
            ModelException: If model loading fails
        """
        try:
            self.logger.logger.info("Loading primary classification model...")
            
            # Create model configuration
            config = ModelConfig(
                model_path=self.model_config["model_path"],
                device=self.model_config["device"],
                cache_size=self.model_config["cache_size"],
                image_dimensions=self.model_config["image_dimensions"],
                categories=self.model_config["categories"]
            )
            
            # Create and load model
            self.primary_model = ProductClassifier(config)
            await self.primary_model.load_model()
            
            # Register model
            await self.registry.register_model("primary", self.primary_model)
            
            # Mark as healthy
            self._is_healthy = True
            self._last_health_check = time.time()
            
            self.logger.logger.info(
                "Primary model loaded successfully",
                model_path=self.model_config["model_path"],
                device=self.model_config["device"],
                categories_count=len(self.model_config["categories"])
            )
            
        except Exception as e:
            self._is_healthy = False
            raise ModelException(
                f"Failed to load primary model: {str(e)}",
                details={
                    "model_path": self.model_config["model_path"],
                    "error": str(e)
                }
            )
    
    async def predict_single(self, image_data: Any) -> Dict[str, Any]:
        """
        Make prediction on a single image.
        
        Args:
            image_data: Input image data
            
        Returns:
            Dict[str, Any]: Prediction results with metadata
            
        Raises:
            ModelException: If prediction fails
        """
        if not self.is_ready():
            raise ModelException("Model service is not ready")
        
        try:
            start_time = time.time()
            
            # Make prediction
            predictions = await self.primary_model.predict(image_data)
            
            # Calculate timing
            inference_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(inference_time)
            
            # Prepare response
            result = {
                "predictions": predictions,
                "metadata": {
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model_version": "1.0.0",
                    "categories_count": len(self.model_config["categories"]),
                    "timestamp": time.time()
                }
            }
            
            self.logger.logger.info(
                "Single prediction completed",
                inference_time_ms=round(inference_time * 1000, 2),
                predictions_count=len(predictions),
                top_category=predictions[0]["category"] if predictions else None
            )
            
            return result
            
        except Exception as e:
            self.logger.logger.error(
                "Single prediction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise ModelException(f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, image_batch: List[Any]) -> Dict[str, Any]:
        """
        Make predictions on a batch of images.
        
        Args:
            image_batch: List of input images
            
        Returns:
            Dict[str, Any]: Batch prediction results with metadata
            
        Raises:
            ModelException: If batch prediction fails
        """
        if not self.is_ready():
            raise ModelException("Model service is not ready")
        
        batch_size = len(image_batch)
        
        if batch_size == 0:
            return {"predictions": [], "metadata": {"batch_size": 0}}
        
        if batch_size > self.settings.max_batch_size:
            raise ModelException(
                f"Batch size {batch_size} exceeds maximum {self.settings.max_batch_size}",
                details={"batch_size": batch_size, "max_batch_size": self.settings.max_batch_size}
            )
        
        if batch_size < self.settings.min_batch_size:
            raise ModelException(
                f"Batch size {batch_size} below minimum {self.settings.min_batch_size}",
                details={"batch_size": batch_size, "min_batch_size": self.settings.min_batch_size}
            )
        
        try:
            start_time = time.time()
            
            # Make batch prediction
            batch_predictions = await self.primary_model.predict_batch(image_batch)
            
            # Calculate timing
            total_time = time.time() - start_time
            avg_time_per_image = total_time / batch_size
            
            # Update statistics
            for _ in range(batch_size):
                self._update_stats(avg_time_per_image)
            
            # Prepare response
            result = {
                "predictions": batch_predictions,
                "metadata": {
                    "batch_size": batch_size,
                    "total_inference_time_ms": round(total_time * 1000, 2),
                    "average_time_per_image_ms": round(avg_time_per_image * 1000, 2),
                    "model_version": "1.0.0",
                    "timestamp": time.time()
                }
            }
            
            self.logger.logger.info(
                "Batch prediction completed",
                batch_size=batch_size,
                total_time_ms=round(total_time * 1000, 2),
                avg_time_per_image_ms=round(avg_time_per_image * 1000, 2)
            )
            
            return result
            
        except Exception as e:
            self.logger.logger.error(
                "Batch prediction failed",
                batch_size=batch_size,
                error=str(e),
                error_type=type(e).__name__
            )
            raise ModelException(f"Batch prediction failed: {str(e)}")
    
    async def predict_with_timing(self, image_data: Any) -> Dict[str, Any]:
        """
        Make prediction with detailed timing breakdown.
        
        Args:
            image_data: Input image data
            
        Returns:
            Dict[str, Any]: Prediction results with detailed timing
        """
        if not self.is_ready():
            raise ModelException("Model service is not ready")
        
        try:
            result = await self.primary_model.predict_with_timing(image_data)
            
            # Update statistics
            total_time = result["timing"]["total_time"] / 1000  # Convert to seconds
            self._update_stats(total_time)
            
            return result
            
        except Exception as e:
            raise ModelException(f"Prediction with timing failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """
        Check if model service is ready for predictions.
        
        Returns:
            bool: True if service is ready
        """
        return (
            self.primary_model is not None and
            self.primary_model.is_loaded() and
            self._is_healthy
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on model service.
        
        Returns:
            Dict[str, Any]: Health status and metrics
        """
        current_time = time.time()
        
        # Only perform actual health check if enough time has passed
        if current_time - self._last_health_check > self._health_check_interval:
            await self._perform_health_check()
            self._last_health_check = current_time
        
        return {
            "healthy": self._is_healthy,
            "model_loaded": self.primary_model is not None and self.primary_model.is_loaded(),
            "model_info": self.primary_model.get_model_info() if self.primary_model else None,
            "statistics": self.get_statistics(),
            "last_health_check": self._last_health_check,
            "uptime": current_time - self._last_health_check if self._last_health_check else 0
        }
    
    async def _perform_health_check(self) -> None:
        """
        Perform actual health check with model warmup.
        """
        try:
            if self.primary_model and self.primary_model.is_loaded():
                # Perform a quick warmup to ensure model is responsive
                await self.primary_model.warmup(1)
                self._is_healthy = True
            else:
                self._is_healthy = False
                
        except Exception as e:
            self.logger.logger.error(f"Health check failed: {str(e)}")
            self._is_healthy = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dict[str, Any]: Service statistics
        """
        avg_inference_time = (
            self._total_inference_time / self._prediction_count
            if self._prediction_count > 0 else 0
        )
        
        return {
            "predictions_total": self._prediction_count,
            "average_inference_time_ms": round(avg_inference_time * 1000, 2),
            "total_inference_time_ms": round(self._total_inference_time * 1000, 2),
            "last_prediction_time": self._last_prediction_time,
            "throughput_predictions_per_second": (
                1.0 / avg_inference_time if avg_inference_time > 0 else 0
            )
        }
    
    def _update_stats(self, inference_time: float) -> None:
        """
        Update prediction statistics.
        
        Args:
            inference_time: Inference time in seconds
        """
        self._prediction_count += 1
        self._total_inference_time += inference_time
        self._last_prediction_time = time.time()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.primary_model:
            return {"error": "No model loaded"}
        
        info = self.primary_model.get_model_info()
        info.update({
            "service_statistics": self.get_statistics(),
            "health_status": await self.health_check()
        })
        
        return info
    
    async def benchmark_model(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            iterations: Number of benchmark iterations
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        if not self.is_ready():
            raise ModelException("Model service is not ready for benchmarking")
        
        try:
            return await self.primary_model.benchmark(iterations)
            
        except Exception as e:
            raise ModelException(f"Benchmarking failed: {str(e)}")
    
    def get_categories(self) -> List[str]:
        """
        Get list of supported categories.
        
        Returns:
            List[str]: Category names
        """
        if self.primary_model:
            return self.primary_model.get_categories()
        return self.model_config["categories"]
    
    def get_category_info(self) -> Dict[str, Any]:
        """
        Get category information.
        
        Returns:
            Dict[str, Any]: Category information
        """
        categories = self.get_categories()
        
        return {
            "categories": categories,
            "total_categories": len(categories),
            "category_mapping": {
                name: idx for idx, name in enumerate(categories)
            }
        }
    
    async def reload_model(self) -> None:
        """
        Reload the primary model.
        
        Raises:
            ModelException: If model reload fails
        """
        try:
            self.logger.logger.info("Reloading primary model...")
            
            # Clean up existing model
            if self.primary_model:
                await self.primary_model.cleanup()
                await self.registry.unregister_model("primary")
            
            # Reset statistics
            self._prediction_count = 0
            self._total_inference_time = 0.0
            self._last_prediction_time = None
            
            # Load new model
            await self.load_model()
            
            self.logger.logger.info("Model reloaded successfully")
            
        except Exception as e:
            self._is_healthy = False
            raise ModelException(f"Model reload failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """
        Clean up model service resources.
        """
        try:
            self.logger.logger.info("Cleaning up model service...")
            
            # Clean up all models
            await self.registry.cleanup_all()
            
            if self.primary_model:
                await self.primary_model.cleanup()
                self.primary_model = None
            
            self._is_healthy = False
            
            self.logger.logger.info("Model service cleanup completed")
            
        except Exception as e:
            self.logger.logger.error(f"Cleanup failed: {str(e)}")
            raise ModelException(f"Cleanup failed: {str(e)}")