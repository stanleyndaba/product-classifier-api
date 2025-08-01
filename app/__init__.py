"""
Smart Product Classifier API

A production-ready Computer Vision API for real-time e-commerce product classification.
Built with FastAPI, PyTorch, OpenCV, and Docker.
"""

__version__ = "1.0.0"
__author__ = "Smart Product Classifier Team"
__email__ = "team@smartproductclassifier.com"

from .main import app

__all__ = ["app"]