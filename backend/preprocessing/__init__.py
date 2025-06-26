"""
Vietnamese Text Preprocessing Package

This package provides utilities for preprocessing Vietnamese text including:
- Text tokenization using multiple methods
- Text cleaning and normalization
- Stop words removal
- Text statistics and analysis
"""

from .tokenizer import VietnameseTokenizer
from .utils import TextPreprocessor, VietnameseStopWords, TextStatistics

__version__ = "1.0.0"
__author__ = "Your Name"

# Package-level exports
__all__ = [
    'VietnameseTokenizer',
    'TextPreprocessor', 
    'VietnameseStopWords',
    'TextStatistics'
]

# Package initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())