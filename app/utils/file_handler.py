"""
File handling utilities for image uploads and validation.
"""

import io
import os
from typing import Any, List
from PIL import Image
from fastapi import UploadFile

from ..core.config import get_settings
from ..core.exceptions import FileUploadException, ValidationException


class FileHandler:
    """
    Handler for file upload operations and validation.
    """
    
    def __init__(self):
        """Initialize file handler with settings."""
        self.settings = get_settings()
    
    async def validate_image_file(self, file: UploadFile) -> None:
        """
        Validate uploaded image file.
        
        Args:
            file: Uploaded file
            
        Raises:
            FileUploadException: If file validation fails
        """
        # Check file exists
        if not file:
            raise FileUploadException("No file provided")
        
        # Check filename
        if not file.filename:
            raise FileUploadException("No filename provided")
        
        # Check content type
        if file.content_type not in self.settings.allowed_image_types:
            raise FileUploadException(
                f"Unsupported file type: {file.content_type}",
                details={
                    "content_type": file.content_type,
                    "allowed_types": self.settings.allowed_image_types
                }
            )
        
        # Check file size
        if hasattr(file, 'size') and file.size:
            if file.size > self.settings.max_image_size:
                raise FileUploadException(
                    f"File too large: {file.size} bytes",
                    details={
                        "file_size": file.size,
                        "max_size": self.settings.max_image_size
                    }
                )
    
    async def read_image_file(self, file: UploadFile) -> Image.Image:
        """
        Read and validate image file content.
        
        Args:
            file: Uploaded file
            
        Returns:
            Image.Image: PIL Image object
            
        Raises:
            FileUploadException: If file reading fails
        """
        try:
            # Read file content
            content = await file.read()
            
            # Check content size
            if len(content) > self.settings.max_image_size:
                raise FileUploadException(
                    f"File content too large: {len(content)} bytes",
                    details={
                        "content_size": len(content),
                        "max_size": self.settings.max_image_size
                    }
                )
            
            # Try to open as image
            try:
                image = Image.open(io.BytesIO(content))
                image.verify()  # Verify it's a valid image
                
                # Reopen for actual use (verify closes the file)
                image = Image.open(io.BytesIO(content))
                
                return image
                
            except Exception as e:
                raise FileUploadException(
                    f"Invalid image file: {str(e)}",
                    details={"filename": file.filename, "error": str(e)}
                )
                
        except FileUploadException:
            raise
        except Exception as e:
            raise FileUploadException(
                f"Failed to read file: {str(e)}",
                details={"filename": file.filename, "error": str(e)}
            )
        
        finally:
            # Reset file position for potential future reads
            await file.seek(0)
    
    def get_file_info(self, file: UploadFile) -> dict:
        """
        Get information about uploaded file.
        
        Args:
            file: Uploaded file
            
        Returns:
            dict: File information
        """
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": getattr(file, 'size', None)
        }