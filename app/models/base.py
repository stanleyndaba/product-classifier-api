"""
Base model class and configuration for machine learning models.
"""

import abc
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.jit import ScriptModule

from ..core.exceptions import ModelException, ConfigurationException
from ..core.logging import ModelLogger


@dataclass
class ModelConfig:
    """
    Configuration for model loading and inference.
    """
    model_path: str
    device: str = "cpu"
    cache_size: int = 1
    image_dimensions: tuple = (224, 224)
    categories: List[str] = None
    confidence_threshold: float = 0.1
    max_predictions: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.categories is None:
            self.categories = [
                "electronics", "clothing", "furniture", "books", "toys",
                "sports", "beauty", "automotive", "home", "garden",
                "jewelry", "food"
            ]
        
        if not Path(self.model_path).exists():
            raise ConfigurationException(
                f"Model file not found: {self.model_path}",
                details={"model_path": self.model_path}
            )
        
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ConfigurationException(
                f"Unsupported device: {self.device}",
                details={"device": self.device, "supported": ["cpu", "cuda", "mps"]}
            )


class BaseModel(abc.ABC):
    """
    Abstract base class for machine learning models.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[Union[nn.Module, ScriptModule]] = None
        self.device = torch.device(config.device)
        self.logger = ModelLogger()
        self._is_loaded = False
        self._load_time = None
    
    @abc.abstractmethod
    async def load_model(self) -> None:
        """
        Load the model from disk.
        
        Raises:
            ModelException: If model loading fails
        """
        pass
    
    @abc.abstractmethod
    async def predict(self, input_data: Any) -> List[Dict[str, Any]]:
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            List[Dict[str, Any]]: Prediction results
            
        Raises:
            ModelException: If prediction fails
        """
        pass
    
    @abc.abstractmethod
    async def predict_batch(self, input_batch: List[Any]) -> List[List[Dict[str, Any]]]:
        """
        Make predictions on a batch of input data.
        
        Args:
            input_batch: Batch of input data
            
        Returns:
            List[List[Dict[str, Any]]]: Batch prediction results
            
        Raises:
            ModelException: If batch prediction fails
        """
        pass
    
    async def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format and content.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if input is valid
            
        Raises:
            ModelException: If input validation fails
        """
        return True
    
    async def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input data before inference.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Any: Preprocessed data
        """
        return input_data
    
    async def postprocess(self, model_output: Any) -> List[Dict[str, Any]]:
        """
        Postprocess model output into structured results.
        
        Args:
            model_output: Raw model output
            
        Returns:
            List[Dict[str, Any]]: Structured prediction results
        """
        return []
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            bool: True if model is loaded
        """
        return self._is_loaded and self.model is not None
    
    def get_load_time(self) -> Optional[float]:
        """
        Get model loading time.
        
        Returns:
            Optional[float]: Load time in seconds, None if not loaded
        """
        return self._load_time
    
    async def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        self.logger.logger.info("Model cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_path": self.config.model_path,
            "device": str(self.device),
            "is_loaded": self.is_loaded(),
            "load_time": self.get_load_time(),
            "categories": self.config.categories,
            "image_dimensions": self.config.image_dimensions,
            "confidence_threshold": self.config.confidence_threshold,
            "max_predictions": self.config.max_predictions
        }
    
    async def warmup(self, num_iterations: int = 5) -> Dict[str, float]:
        """
        Warm up the model with dummy predictions.
        
        Args:
            num_iterations: Number of warmup iterations
            
        Returns:
            Dict[str, float]: Warmup statistics
        """
        if not self.is_loaded():
            raise ModelException("Model must be loaded before warmup")
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, *self.config.image_dimensions,
            device=self.device
        )
        
        warmup_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    if isinstance(self.model, ScriptModule):
                        _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)
                
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                
            except Exception as e:
                self.logger.logger.error(f"Warmup iteration {i} failed: {e}")
                raise ModelException(f"Model warmup failed: {e}")
        
        avg_time = sum(warmup_times) / len(warmup_times)
        min_time = min(warmup_times)
        max_time = max(warmup_times)
        
        self.logger.logger.info(
            f"Model warmup completed: {num_iterations} iterations, "
            f"avg: {avg_time:.3f}s, min: {min_time:.3f}s, max: {max_time:.3f}s"
        )
        
        return {
            "iterations": num_iterations,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_time": sum(warmup_times)
        }
    
    async def benchmark(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            num_iterations: Number of benchmark iterations
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        if not self.is_loaded():
            raise ModelException("Model must be loaded before benchmarking")
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, *self.config.image_dimensions,
            device=self.device
        )
        
        inference_times = []
        
        # Warmup
        await self.warmup(5)
        
        # Benchmark
        for i in range(num_iterations):
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    if isinstance(self.model, ScriptModule):
                        output = self.model(dummy_input)
                    else:
                        output = self.model(dummy_input)
                
                # Ensure computation is complete
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
            except Exception as e:
                self.logger.logger.error(f"Benchmark iteration {i} failed: {e}")
                raise ModelException(f"Model benchmarking failed: {e}")
        
        # Calculate statistics
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        
        # Calculate percentiles
        sorted_times = sorted(inference_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        throughput = 1.0 / avg_time
        
        benchmark_results = {
            "iterations": num_iterations,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "p50_time": p50,
            "p95_time": p95,
            "p99_time": p99,
            "throughput_fps": throughput,
            "total_time": sum(inference_times)
        }
        
        self.logger.logger.info(
            f"Model benchmark completed: {num_iterations} iterations, "
            f"avg: {avg_time:.3f}s, throughput: {throughput:.1f} FPS"
        )
        
        return benchmark_results


class ModelRegistry:
    """
    Registry for managing multiple model instances.
    """
    
    def __init__(self):
        """Initialize model registry."""
        self.models: Dict[str, BaseModel] = {}
        self.logger = ModelLogger()
    
    async def register_model(self, name: str, model: BaseModel) -> None:
        """
        Register a model instance.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
        self.logger.logger.info(f"Model registered: {name}")
    
    async def unregister_model(self, name: str) -> None:
        """
        Unregister and cleanup a model.
        
        Args:
            name: Model name
        """
        if name in self.models:
            await self.models[name].cleanup()
            del self.models[name]
            self.logger.logger.info(f"Model unregistered: {name}")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """
        Get a registered model.
        
        Args:
            name: Model name
            
        Returns:
            Optional[BaseModel]: Model instance or None
        """
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
            List[str]: Model names
        """
        return list(self.models.keys())
    
    async def cleanup_all(self) -> None:
        """
        Cleanup all registered models.
        """
        for name in list(self.models.keys()):
            await self.unregister_model(name)
        
        self.logger.logger.info("All models cleaned up")