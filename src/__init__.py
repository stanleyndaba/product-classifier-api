"""
Smart Product Classifier
A production-ready Computer Vision API for real-time e-commerce product classification.
"""

__version__ = "1.0.0"
__author__ = "Stanley Ndaba"
__email__ = "your.email@example.com"

from pathlib import Path
import sys

# Add src to Python path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
]