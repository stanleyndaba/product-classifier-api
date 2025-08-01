"""
Product classification model implementation.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import asyncio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import ScriptModule
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from .base import BaseModel, ModelConfig
from .preprocessing import ImagePreprocessor
from ..core.exceptions import ModelException, ValidationException


@dataclass
class ClassificationResult:
    """
    Result of a classification prediction.
    """
    category: str
    confidence: float
    category_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "category_id": self.category_id
        }


class ProductClassifier(BaseModel):
    """
    Product classification model using ResNet-18 architecture.
    
    This model classifies product images into predefined categories
    with confidence scores and supports both single and batch inference.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize product classifier.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.preprocessor = ImagePreprocessor(
            target_size=config.image_dimensions,
            device=str(self.device)
        )
        self._class_names = config.categories
        self._num_classes = len(self._class_names)
    
    async def load_model(self) -> None:
        """
        Load the product classification model.
        
        Raises:
            ModelException: If model loading fails
        """
        try:
            start_time = time.time()
            
            # Load the TorchScript model
            model_path = Path(self.config.model_path)
            
            if not model_path.exists():
                raise ModelException(
                    f"Model file not found: {model_path}",
                    details={"model_path": str(model_path)}
                )
            
            # Load model based on file extension
            if model_path.suffix == '.pt':
                # TorchScript model
                self.model = torch.jit.load(str(model_path), map_location=self.device)
                self.model.eval()
            elif model_path.suffix in ['.pth', '.ckpt']:
                # PyTorch state dict - create ResNet-18 architecture
                self.model = self._create_resnet18_model()
                state_dict = torch.load(str(model_path), map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
            else:
                raise ModelException(
                    f"Unsupported model format: {model_path.suffix}",
                    details={"supported_formats": [".pt", ".pth", ".ckpt"]}
                )
            
            self._load_time = time.time() - start_time
            self._is_loaded = True
            
            # Log successful loading
            self.logger.log_model_load(
                model_name="ProductClassifier",
                model_path=str(model_path),
                load_time=self._load_time
            )
            
            # Warm up the model
            await self.warmup(3)
            
        except Exception as e:
            raise ModelException(
                f"Failed to load model: {str(e)}",
                details={
                    "model_path": self.config.model_path,
                    "device": str(self.device),
                    "error": str(e)
                }
            )
    
    def _create_resnet18_model(self) -> nn.Module:
        """
        Create ResNet-18 model architecture.
        
        Returns:
            nn.Module: ResNet-18 model
        """
        import torchvision.models as models
        
        # Create ResNet-18 with custom number of classes
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self._num_classes)
        
        return model
    
    async def validate_input(self, input_data: Any) -> bool:
        """
        Validate input image data.
        
        Args:
            input_data: Input image data (PIL Image, numpy array, or tensor)
            
        Returns:
            bool: True if input is valid
            
        Raises:
            ValidationException: If input validation fails
        """
        try:
            if isinstance(input_data, Image.Image):
                # PIL Image
                if input_data.mode not in ['RGB', 'RGBA', 'L']:
                    raise ValidationException(
                        f"Unsupported image mode: {input_data.mode}",
                        details={"supported_modes": ["RGB", "RGBA", "L"]}
                    )
                
                width, height = input_data.size
                if width < 32 or height < 32:
                    raise ValidationException(
                        "Image too small (minimum 32x32)",
                        details={"width": width, "height": height, "minimum": 32}
                    )
                
                if width > 4096 or height > 4096:
                    raise ValidationException(
                        "Image too large (maximum 4096x4096)",
                        details={"width": width, "height": height, "maximum": 4096}
                    )
            
            elif isinstance(input_data, np.ndarray):
                # Numpy array
                if input_data.ndim not in [2, 3]:
                    raise ValidationException(
                        f"Invalid image dimensions: {input_data.ndim}",
                        details={"dimensions": input_data.ndim, "expected": [2, 3]}
                    )
                
                if input_data.ndim == 3 and input_data.shape[2] not in [1, 3, 4]:
                    raise ValidationException(
                        f"Invalid number of channels: {input_data.shape[2]}",
                        details={"channels": input_data.shape[2], "expected": [1, 3, 4]}
                    )
            
            elif isinstance(input_data, torch.Tensor):
                # PyTorch tensor
                if input_data.ndim not in [3, 4]:
                    raise ValidationException(
                        f"Invalid tensor dimensions: {input_data.ndim}",
                        details={"dimensions": input_data.ndim, "expected": [3, 4]}
                    )
            
            else:
                raise ValidationException(
                    f"Unsupported input type: {type(input_data)}",
                    details={"supported_types": ["PIL.Image", "numpy.ndarray", "torch.Tensor"]}
                )
            
            return True
            
        except ValidationException:
            raise
        except Exception as e:
            raise ValidationException(
                f"Input validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def preprocess(self, input_data: Any) -> torch.Tensor:
        """
        Preprocess input image for model inference.
        
        Args:
            input_data: Input image data
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return await self.preprocessor.preprocess(input_data)
    
    async def predict(self, input_data: Any) -> List[Dict[str, Any]]:
        """
        Make prediction on a single image.
        
        Args:
            input_data: Input image data
            
        Returns:
            List[Dict[str, Any]]: Top predictions with confidence scores
            
        Raises:
            ModelException: If prediction fails
        """
        if not self.is_loaded():
            raise ModelException("Model must be loaded before prediction")
        
        try:
            # Validate and preprocess input
            await self.validate_input(input_data)
            preprocessed_input = await self.preprocess(input_data)
            
            # Add batch dimension if needed
            if preprocessed_input.ndim == 3:
                preprocessed_input = preprocessed_input.unsqueeze(0)
            
            # Model inference
            start_time = time.time()
            
            with torch.no_grad():
                if isinstance(self.model, ScriptModule):
                    logits = self.model(preprocessed_input)
                else:
                    logits = self.model(preprocessed_input)
            
            inference_time = time.time() - start_time
            
            # Postprocess results
            predictions = await self.postprocess(logits)
            
            # Log inference
            self.logger.log_inference(
                model_name="ProductClassifier",
                input_shape=preprocessed_input.shape,
                inference_time=inference_time,
                predictions=predictions
            )
            
            return predictions
            
        except (ValidationException, ModelException):
            raise
        except Exception as e:
            raise ModelException(
                f"Prediction failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def predict_batch(self, input_batch: List[Any]) -> List[List[Dict[str, Any]]]:
        """
        Make predictions on a batch of images.
        
        Args:
            input_batch: List of input images
            
        Returns:
            List[List[Dict[str, Any]]]: Batch of predictions
            
        Raises:
            ModelException: If batch prediction fails
        """
        if not self.is_loaded():
            raise ModelException("Model must be loaded before prediction")
        
        try:
            batch_size = len(input_batch)
            
            if batch_size == 0:
                return []
            
            # Validate and preprocess all inputs
            preprocessed_batch = []
            for i, input_data in enumerate(input_batch):
                try:
                    await self.validate_input(input_data)
                    preprocessed = await self.preprocess(input_data)
                    preprocessed_batch.append(preprocessed)
                except Exception as e:
                    raise ModelException(
                        f"Failed to preprocess batch item {i}: {str(e)}",
                        details={"batch_index": i, "error": str(e)}
                    )
            
            # Stack into batch tensor
            batch_tensor = torch.stack(preprocessed_batch)
            
            # Model inference
            start_time = time.time()
            
            with torch.no_grad():
                if isinstance(self.model, ScriptModule):
                    batch_logits = self.model(batch_tensor)
                else:
                    batch_logits = self.model(batch_tensor)
            
            inference_time = time.time() - start_time
            
            # Postprocess each result in batch
            batch_predictions = []
            for i in range(batch_size):
                logits = batch_logits[i:i+1]  # Keep batch dimension
                predictions = await self.postprocess(logits)
                batch_predictions.append(predictions)
            
            # Log batch inference
            self.logger.logger.info(
                "Batch Inference",
                model_name="ProductClassifier",
                batch_size=batch_size,
                input_shape=batch_tensor.shape,
                inference_time_ms=round(inference_time * 1000, 2),
                avg_time_per_image=round(inference_time / batch_size * 1000, 2)
            )
            
            return batch_predictions
            
        except ModelException:
            raise
        except Exception as e:
            raise ModelException(
                f"Batch prediction failed: {str(e)}",
                details={"batch_size": len(input_batch), "error": str(e)}
            )
    
    async def postprocess(self, model_output: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Postprocess model output into structured predictions.
        
        Args:
            model_output: Raw model output logits
            
        Returns:
            List[Dict[str, Any]]: Structured prediction results
        """
        try:
            # Apply softmax to get probabilities
            probabilities = F.softmax(model_output, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(
                probabilities[0], 
                k=min(self.config.max_predictions, self._num_classes)
            )
            
            # Convert to list of results
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                confidence = prob.item()
                category_id = idx.item()
                
                # Only include predictions above threshold
                if confidence >= self.config.confidence_threshold:
                    category = self._class_names[category_id]
                    
                    result = ClassificationResult(
                        category=category,
                        confidence=confidence,
                        category_id=category_id
                    )
                    
                    predictions.append(result.to_dict())
            
            # Ensure at least one prediction if all are below threshold
            if not predictions and len(top_probs) > 0:
                best_prob = top_probs[0].item()
                best_idx = top_indices[0].item()
                best_category = self._class_names[best_idx]
                
                result = ClassificationResult(
                    category=best_category,
                    confidence=best_prob,
                    category_id=best_idx
                )
                
                predictions.append(result.to_dict())
            
            return predictions
            
        except Exception as e:
            raise ModelException(
                f"Postprocessing failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_categories(self) -> List[str]:
        """
        Get list of supported categories.
        
        Returns:
            List[str]: Category names
        """
        return self._class_names.copy()
    
    def get_category_id(self, category_name: str) -> Optional[int]:
        """
        Get category ID by name.
        
        Args:
            category_name: Category name
            
        Returns:
            Optional[int]: Category ID or None if not found
        """
        try:
            return self._class_names.index(category_name)
        except ValueError:
            return None
    
    def get_category_name(self, category_id: int) -> Optional[str]:
        """
        Get category name by ID.
        
        Args:
            category_id: Category ID
            
        Returns:
            Optional[str]: Category name or None if invalid ID
        """
        if 0 <= category_id < len(self._class_names):
            return self._class_names[category_id]
        return None
    
    async def predict_with_timing(self, input_data: Any) -> Dict[str, Any]:
        """
        Make prediction with detailed timing information.
        
        Args:
            input_data: Input image data
            
        Returns:
            Dict[str, Any]: Predictions with timing details
        """
        total_start = time.time()
        
        # Preprocessing timing
        preprocess_start = time.time()
        await self.validate_input(input_data)
        preprocessed_input = await self.preprocess(input_data)
        preprocess_time = time.time() - preprocess_start
        
        # Add batch dimension if needed
        if preprocessed_input.ndim == 3:
            preprocessed_input = preprocessed_input.unsqueeze(0)
        
        # Inference timing
        inference_start = time.time()
        with torch.no_grad():
            if isinstance(self.model, ScriptModule):
                logits = self.model(preprocessed_input)
            else:
                logits = self.model(preprocessed_input)
        inference_time = time.time() - inference_start
        
        # Postprocessing timing
        postprocess_start = time.time()
        predictions = await self.postprocess(logits)
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - total_start
        
        return {
            "predictions": predictions,
            "timing": {
                "total_time": round(total_time * 1000, 2),
                "preprocessing_time": round(preprocess_time * 1000, 2),
                "inference_time": round(inference_time * 1000, 2),
                "postprocessing_time": round(postprocess_time * 1000, 2)
            },
            "metadata": {
                "input_shape": list(preprocessed_input.shape),
                "device": str(self.device),
                "model_loaded": self.is_loaded()
            }
        }