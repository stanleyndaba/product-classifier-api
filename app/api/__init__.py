"""
API routes and endpoints for the Smart Product Classifier.
"""

from .routes import prediction, health, batch

__all__ = ["prediction", "health", "batch"]