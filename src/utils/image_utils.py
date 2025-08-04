"""
Image processing utilities for the Smart Product Classifier.
"""

import io
import base64
import asyncio
import aiohttp
from typing import Union, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Image processing utilities for the product classifier."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.transform = self._get_transform()
    
    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    async def process_upload(self, image_data: bytes) -> torch.Tensor:
        """
        Process uploaded image data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Processed image tensor
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Apply transformations
            tensor = self.transform(image)
            
            # Add batch dimension
            return tensor.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    async def process_url(self, image_url: str, timeout: int = 10) -> torch.Tensor:
        """
        Process image from URL.
        
        Args:
            image_url: URL to image file
            timeout: Download timeout in seconds
            
        Returns:
            Processed image tensor
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=timeout) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download image: HTTP {response.status}")
                    
                    image_data = await response.read()
                    return await self.process_upload(image_data)
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading image from {image_url}")
            raise ValueError("Image download timeout")
        except Exception as e:
            logger.error(f"Error processing image from URL {image_url}: {e}")
            raise ValueError(f"Failed to process image from URL: {e}")
    
    def process_base64(self, base64_data: str) -> torch.Tensor:
        """
        Process base64 encoded image.
        
        Args:
            base64_data: Base64 encoded image string
            
        Returns:
            Processed image tensor
        """
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Process as uploaded image
            return asyncio.run(self.process_upload(image_data))
            
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            raise ValueError(f"Invalid base64 image data: {e}")
    
    def process_file(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Process image from file path.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Processed image tensor
        """
        try:
            image = Image.open(file_path).convert('RGB')
            tensor = self.transform(image)
            return tensor.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")
            raise ValueError(f"Failed to process image file: {e}")
    
    def validate_image(self, image_data: bytes, max_size: int = 10 * 1024 * 1024) -> bool:
        """
        Validate image data.
        
        Args:
            image_data: Raw image bytes
            max_size: Maximum file size in bytes
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file size
            if len(image_data) > max_size:
                logger.warning(f"Image too large: {len(image_data)} bytes")
                return False
            
            # Try to open image
            image = Image.open(io.BytesIO(image_data))
            
            # Check dimensions
            width, height = image.size
            if width < 32 or height < 32:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            
            if width > 4096 or height > 4096:
                logger.warning(f"Image too large: {width}x{height}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Invalid image: {e}")
            return False
    
    def get_image_info(self, image_data: bytes) -> dict:
        """
        Get image information.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with image information
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            return {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'file_size': len(image_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}


class OpenCVProcessor:
    """OpenCV-based image processing utilities."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def preprocess_opencv(self, image_path: str) -> np.ndarray:
        """
        Preprocess image using OpenCV.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.target_size)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # Convert to CHW format
            image = np.transpose(image, (2, 0, 1))
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image with OpenCV: {e}")
            raise
    
    def apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentations.
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image array
        """
        try:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
            
            # Random brightness and contrast
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-20, 20)    # brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            return image
            
        except Exception as e:
            logger.error(f"Error applying augmentations: {e}")
            return image
    
    def detect_objects(self, image: np.ndarray) -> list:
        """
        Simple object detection using contours.
        
        Args:
            image: Input image array
            
        Returns:
            List of detected object bounding boxes
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract bounding boxes
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 50:  # Filter small objects
                    boxes.append((x, y, x + w, y + h))
            
            return boxes
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []


def create_thumbnail(image_data: bytes, size: Tuple[int, int] = (128, 128)) -> bytes:
    """
    Create thumbnail from image data.
    
    Args:
        image_data: Raw image bytes
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image bytes
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating thumbnail: {e}")
        raise ValueError(f"Failed to create thumbnail: {e}")


def convert_to_jpeg(image_data: bytes, quality: int = 90) -> bytes:
    """
    Convert image to JPEG format.
    
    Args:
        image_data: Raw image bytes
        quality: JPEG quality (1-100)
        
    Returns:
        JPEG image bytes
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        # Save as JPEG
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error converting to JPEG: {e}")
        raise ValueError(f"Failed to convert to JPEG: {e}")