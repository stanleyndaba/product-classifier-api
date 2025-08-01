"""
Image preprocessing utilities for machine learning models.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from typing import Any, Tuple, Union, Optional
import io
import asyncio

from ..core.exceptions import ImageProcessingException


class ImagePreprocessor:
    """
    Image preprocessing pipeline for product classification.
    
    Handles various input formats (PIL, OpenCV, numpy, tensor) and provides
    robust preprocessing with error handling and validation.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        device: str = "cpu",
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image dimensions (height, width)
            device: Device for tensor operations
            normalize: Whether to apply ImageNet normalization
            mean: Normalization mean values (RGB)
            std: Normalization standard deviation values (RGB)
        """
        self.target_size = target_size
        self.device = device
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Create transform pipeline
        self._create_transforms()
    
    def _create_transforms(self) -> None:
        """Create torchvision transform pipeline."""
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        self.transform = transforms.Compose(transform_list)
        
        # Alternative transform without normalization for debugging
        self.transform_no_norm = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])
    
    async def preprocess(self, input_data: Any) -> torch.Tensor:
        """
        Preprocess input image data.
        
        Args:
            input_data: Input image (PIL Image, numpy array, cv2 image, or tensor)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
            
        Raises:
            ImageProcessingException: If preprocessing fails
        """
        try:
            # Convert to PIL Image first
            pil_image = await self.to_pil_image(input_data)
            
            # Validate image
            await self.validate_image(pil_image)
            
            # Apply transforms
            tensor = self.transform(pil_image)
            
            # Move to device
            if self.device != "cpu":
                tensor = tensor.to(self.device)
            
            return tensor
            
        except ImageProcessingException:
            raise
        except Exception as e:
            raise ImageProcessingException(
                f"Image preprocessing failed: {str(e)}",
                details={"error": str(e), "input_type": type(input_data).__name__}
            )
    
    async def to_pil_image(self, input_data: Any) -> Image.Image:
        """
        Convert various input formats to PIL Image.
        
        Args:
            input_data: Input image data
            
        Returns:
            Image.Image: PIL Image
            
        Raises:
            ImageProcessingException: If conversion fails
        """
        try:
            if isinstance(input_data, Image.Image):
                # Already PIL Image
                return self._ensure_rgb(input_data)
            
            elif isinstance(input_data, np.ndarray):
                # NumPy array (OpenCV format)
                return await self._numpy_to_pil(input_data)
            
            elif isinstance(input_data, torch.Tensor):
                # PyTorch tensor
                return await self._tensor_to_pil(input_data)
            
            elif isinstance(input_data, bytes):
                # Bytes data
                return await self._bytes_to_pil(input_data)
            
            elif isinstance(input_data, str):
                # File path
                return await self._load_from_path(input_data)
            
            else:
                raise ImageProcessingException(
                    f"Unsupported input type: {type(input_data)}",
                    details={
                        "input_type": type(input_data).__name__,
                        "supported_types": ["PIL.Image", "numpy.ndarray", "torch.Tensor", "bytes", "str"]
                    }
                )
                
        except ImageProcessingException:
            raise
        except Exception as e:
            raise ImageProcessingException(
                f"Failed to convert to PIL Image: {str(e)}",
                details={"error": str(e), "input_type": type(input_data).__name__}
            )
    
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """
        Ensure image is in RGB format.
        
        Args:
            image: PIL Image
            
        Returns:
            Image.Image: RGB PIL Image
        """
        if image.mode == 'RGB':
            return image
        elif image.mode == 'RGBA':
            # Create white background and paste RGBA image
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            return rgb_image
        elif image.mode in ['L', 'P']:
            # Convert grayscale or palette to RGB
            return image.convert('RGB')
        else:
            # For other modes, convert to RGB
            return image.convert('RGB')
    
    async def _numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            array: NumPy array
            
        Returns:
            Image.Image: PIL Image
        """
        try:
            # Handle different array shapes
            if array.ndim == 2:
                # Grayscale image
                image = Image.fromarray(array, mode='L')
            elif array.ndim == 3:
                if array.shape[2] == 1:
                    # Single channel
                    image = Image.fromarray(array.squeeze(), mode='L')
                elif array.shape[2] == 3:
                    # RGB image (OpenCV uses BGR, so convert)
                    if self._is_bgr_format(array):
                        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(array, mode='RGB')
                elif array.shape[2] == 4:
                    # RGBA image
                    image = Image.fromarray(array, mode='RGBA')
                else:
                    raise ImageProcessingException(
                        f"Unsupported number of channels: {array.shape[2]}",
                        details={"channels": array.shape[2], "shape": array.shape}
                    )
            else:
                raise ImageProcessingException(
                    f"Unsupported array dimensions: {array.ndim}",
                    details={"dimensions": array.ndim, "shape": array.shape}
                )
            
            return self._ensure_rgb(image)
            
        except Exception as e:
            raise ImageProcessingException(
                f"Failed to convert numpy array to PIL: {str(e)}",
                details={"error": str(e), "shape": array.shape, "dtype": str(array.dtype)}
            )
    
    def _is_bgr_format(self, array: np.ndarray) -> bool:
        """
        Heuristic to detect if array is in BGR format (typical for OpenCV).
        
        Args:
            array: NumPy array
            
        Returns:
            bool: True if likely BGR format
        """
        # Simple heuristic: if the array values suggest it might be from OpenCV
        # This is not foolproof but covers most cases
        return array.dtype == np.uint8 and array.max() <= 255
    
    async def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert PyTorch tensor to PIL Image.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Image.Image: PIL Image
        """
        try:
            # Handle different tensor shapes
            if tensor.ndim == 4:
                # Batch tensor, take first image
                tensor = tensor[0]
            
            if tensor.ndim != 3:
                raise ImageProcessingException(
                    f"Expected 3D tensor, got {tensor.ndim}D",
                    details={"dimensions": tensor.ndim, "shape": list(tensor.shape)}
                )
            
            # Move to CPU and convert to numpy
            numpy_array = tensor.detach().cpu().numpy()
            
            # Handle different channel arrangements
            if numpy_array.shape[0] == 1:
                # Single channel (C, H, W) -> (H, W)
                numpy_array = numpy_array.squeeze(0)
                image = Image.fromarray((numpy_array * 255).astype(np.uint8), mode='L')
            elif numpy_array.shape[0] == 3:
                # RGB channels (C, H, W) -> (H, W, C)
                numpy_array = np.transpose(numpy_array, (1, 2, 0))
                
                # Denormalize if values are in [-1, 1] or [0, 1] range
                if numpy_array.max() <= 1.0:
                    numpy_array = (numpy_array * 255).astype(np.uint8)
                
                image = Image.fromarray(numpy_array, mode='RGB')
            else:
                raise ImageProcessingException(
                    f"Unsupported number of channels: {numpy_array.shape[0]}",
                    details={"channels": numpy_array.shape[0], "shape": list(tensor.shape)}
                )
            
            return self._ensure_rgb(image)
            
        except Exception as e:
            raise ImageProcessingException(
                f"Failed to convert tensor to PIL: {str(e)}",
                details={"error": str(e), "shape": list(tensor.shape)}
            )
    
    async def _bytes_to_pil(self, data: bytes) -> Image.Image:
        """
        Convert bytes data to PIL Image.
        
        Args:
            data: Image bytes
            
        Returns:
            Image.Image: PIL Image
        """
        try:
            image = Image.open(io.BytesIO(data))
            return self._ensure_rgb(image)
            
        except Exception as e:
            raise ImageProcessingException(
                f"Failed to load image from bytes: {str(e)}",
                details={"error": str(e), "data_size": len(data)}
            )
    
    async def _load_from_path(self, path: str) -> Image.Image:
        """
        Load image from file path.
        
        Args:
            path: Image file path
            
        Returns:
            Image.Image: PIL Image
        """
        try:
            # Try PIL first
            try:
                image = Image.open(path)
                return self._ensure_rgb(image)
            except Exception:
                # Fallback to OpenCV
                cv_image = cv2.imread(path)
                if cv_image is None:
                    raise ImageProcessingException(
                        f"Could not load image from path: {path}",
                        details={"path": path}
                    )
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_image)
                
        except ImageProcessingException:
            raise
        except Exception as e:
            raise ImageProcessingException(
                f"Failed to load image from path: {str(e)}",
                details={"error": str(e), "path": path}
            )
    
    async def validate_image(self, image: Image.Image) -> None:
        """
        Validate PIL Image.
        
        Args:
            image: PIL Image to validate
            
        Raises:
            ImageProcessingException: If validation fails
        """
        try:
            # Check image size
            width, height = image.size
            
            if width < 16 or height < 16:
                raise ImageProcessingException(
                    "Image too small (minimum 16x16)",
                    details={"width": width, "height": height, "minimum": 16}
                )
            
            if width > 8192 or height > 8192:
                raise ImageProcessingException(
                    "Image too large (maximum 8192x8192)",
                    details={"width": width, "height": height, "maximum": 8192}
                )
            
            # Check if image is corrupted
            try:
                image.verify()
            except Exception:
                # Re-open if verify closes the image
                if hasattr(image, 'fp') and image.fp is None:
                    raise ImageProcessingException(
                        "Image verification failed - possibly corrupted",
                        details={"width": width, "height": height}
                    )
                
        except ImageProcessingException:
            raise
        except Exception as e:
            raise ImageProcessingException(
                f"Image validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_image_info(self, image: Image.Image) -> dict:
        """
        Get information about an image.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: Image information
        """
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "has_transparency": image.mode in ['RGBA', 'LA'] or 'transparency' in image.info
        }
    
    async def resize_image(
        self,
        image: Image.Image,
        size: Tuple[int, int],
        maintain_aspect_ratio: bool = True,
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Resize image with optional aspect ratio preservation.
        
        Args:
            image: PIL Image
            size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            fill_color: Fill color for padding (RGB)
            
        Returns:
            Image.Image: Resized image
        """
        try:
            if maintain_aspect_ratio:
                # Calculate size maintaining aspect ratio
                image.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Create new image with target size and fill color
                new_image = Image.new('RGB', size, fill_color)
                
                # Paste resized image centered
                x = (size[0] - image.size[0]) // 2
                y = (size[1] - image.size[1]) // 2
                new_image.paste(image, (x, y))
                
                return new_image
            else:
                # Direct resize without maintaining aspect ratio
                return image.resize(size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            raise ImageProcessingException(
                f"Image resize failed: {str(e)}",
                details={"error": str(e), "target_size": size}
            )
    
    async def preprocess_batch(self, input_batch: list) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            input_batch: List of input images
            
        Returns:
            torch.Tensor: Batch of preprocessed image tensors
            
        Raises:
            ImageProcessingException: If batch preprocessing fails
        """
        try:
            processed_images = []
            
            for i, input_data in enumerate(input_batch):
                try:
                    processed = await self.preprocess(input_data)
                    processed_images.append(processed)
                except Exception as e:
                    raise ImageProcessingException(
                        f"Failed to preprocess batch item {i}: {str(e)}",
                        details={"batch_index": i, "error": str(e)}
                    )
            
            # Stack into batch tensor
            return torch.stack(processed_images)
            
        except ImageProcessingException:
            raise
        except Exception as e:
            raise ImageProcessingException(
                f"Batch preprocessing failed: {str(e)}",
                details={"error": str(e), "batch_size": len(input_batch)}
            )