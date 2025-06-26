"""
Vietnamese Text Similarity Measures Package

This package provides various text similarity calculation methods including:
- String-based similarity (Levenshtein, Jaro, LCS, etc.)
- Set-based similarity (Jaccard, Dice, Overlap, etc.)
- Vector-based similarity (TF-IDF Cosine, Count Vectors, etc.)
- Semantic-based similarity (Word2Vec, FastText, PhoBERT, etc.)

Usage:
    from similarity_measures import SimilarityCalculator
    
    calculator = SimilarityCalculator()
    score = calculator.calculate('text1', 'text2', method='cosine_tfidf')
"""

from .string_based import StringBasedSimilarity
from .set_based import SetBasedSimilarity
from .vector_based import VectorBasedSimilarity
from .semantic_based import SemanticBasedSimilarity

import logging
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Vietnamese Text Similarity Measures"

# Package-level exports
__all__ = [
    'StringBasedSimilarity',
    'SetBasedSimilarity', 
    'VectorBasedSimilarity',
    'SemanticBasedSimilarity',
    'SimilarityCalculator',
    'SimilarityBenchmark',
    'get_available_methods',
    'validate_method'
]

# Configure logging for the package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class SimilarityCalculator:
    """
    Unified interface for all similarity calculation methods
    
    This class provides a single entry point to access all similarity measures
    and handles method routing, validation, and error handling.
    """
    
    def __init__(self, enable_semantic_models: bool = False):
        """
        Initialize the SimilarityCalculator
        
        Args:
            enable_semantic_models: Whether to load semantic models (Word2Vec, PhoBERT, etc.)
                                   Set to False for faster initialization if only using basic methods
        """
        self.logger = logging.getLogger(__name__ + '.SimilarityCalculator')
        
        # Initialize similarity calculators
        self.string_similarity = StringBasedSimilarity()
        self.set_similarity = SetBasedSimilarity()
        self.vector_similarity = VectorBasedSimilarity()
        self.semantic_similarity = SemanticBasedSimilarity()
        
        # Load semantic models if requested
        self.semantic_models_enabled = enable_semantic_models
        if enable_semantic_models:
            self._initialize_semantic_models()
        
        # Method categories mapping
        self.method_categories = {
            'string_based': ['levenshtein', 'jaro', 'sequence_matcher', 'lcs'],
            'set_based': ['jaccard', 'dice', 'overlap', 'cosine_sets'],
            'vector_based': ['cosine_tfidf', 'cosine_count', 'euclidean', 'manhattan'],
            'semantic_based': ['simple', 'word2vec', 'fasttext', 'phobert']
        }
        
        # Flatten all methods for quick lookup
        self.all_methods = set()
        for methods in self.method_categories.values():
            self.all_methods.update(methods)
    
    def _initialize_semantic_models(self):
        """Initialize semantic models (optional, for better performance)"""
        try:
            # Try to load PhoBERT model
            self.semantic_similarity.load_phobert_model()
            self.logger.info("Semantic models initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize semantic models: {e}")
    
    def calculate(self, text1: str, text2: str, method: str = 'cosine_tfidf', 
                 tokens1: Optional[List[str]] = None, 
                 tokens2: Optional[List[str]] = None) -> float:
        """
        Calculate similarity between two texts using specified method
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method to use
            tokens1: Pre-tokenized first text (optional)
            tokens2: Pre-tokenized second text (optional)
            
        Returns:
            Similarity score (0.0 to 1.0)
            
        Raises:
            ValueError: If method is not supported
        """
        if not self.validate_method(method):
            raise ValueError(f"Unsupported similarity method: {method}")
        
        try:
            # Route to appropriate similarity calculator
            if method in self.method_categories['string_based']:
                return self.string_similarity.calculate_similarity(text1, text2, method)
            
            elif method in self.method_categories['set_based']:
                if tokens1 is None or tokens2 is None:
                    # Import tokenizer here to avoid circular imports
                    from ..preprocessing.tokenizer import VietnameseTokenizer
                    tokenizer = VietnameseTokenizer()
                    tokens1 = tokenizer.tokenize(text1)
                    tokens2 = tokenizer.tokenize(text2)
                return self.set_similarity.calculate_similarity(tokens1, tokens2, method)
            
            elif method in self.method_categories['vector_based']:
                return self.vector_similarity.calculate_similarity(text1, text2, method)
            
            elif method in self.method_categories['semantic_based']:
                if tokens1 is None or tokens2 is None:
                    from ..preprocessing.tokenizer import VietnameseTokenizer
                    tokenizer = VietnameseTokenizer()
                    tokens1 = tokenizer.tokenize(text1)
                    tokens2 = tokenizer.tokenize(text2)
                return self.semantic_similarity.calculate_similarity(text1, text2, tokens1, tokens2, method)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity with method {method}: {e}")
            return 0.0
    
    def calculate_multiple(self, text1: str, text2: str, 
                          methods: List[str], 
                          tokens1: Optional[List[str]] = None,
                          tokens2: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate similarity using multiple methods
        
        Args:
            text1: First text
            text2: Second text
            methods: List of similarity methods
            tokens1: Pre-tokenized first text (optional)
            tokens2: Pre-tokenized second text (optional)
            
        Returns:
            Dictionary mapping method names to similarity scores
        """
        results = {}
        
        for method in methods:
            try:
                score = self.calculate(text1, text2, method, tokens1, tokens2)
                results[method] = score
            except Exception as e:
                self.logger.error(f"Error with method {method}: {e}")
                results[method] = None
        
        return results
    
    def calculate_all_methods(self, text1: str, text2: str,
                             tokens1: Optional[List[str]] = None,
                             tokens2: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate similarity using all available methods, grouped by category
        
        Returns:
            Nested dictionary: {category: {method: score}}
        """
        results = {}
        
        for category, methods in self.method_categories.items():
            results[category] = {}
            
            # Skip semantic methods if models not enabled
            if category == 'semantic_based' and not self.semantic_models_enabled:
                # Only use simple semantic method
                methods = ['simple']
            
            for method in methods:
                try:
                    score = self.calculate(text1, text2, method, tokens1, tokens2)
                    results[category][method] = score
                except Exception as e:
                    self.logger.error(f"Error with {category}.{method}: {e}")
                    results[category][method] = None
        
        return results
    
    def get_best_method(self, text1: str, text2: str, 
                       exclude_methods: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        Find the method that gives the highest similarity score
        
        Args:
            text1: First text
            text2: Second text
            exclude_methods: Methods to exclude from comparison
            
        Returns:
            Tuple of (best_method, best_score)
        """
        exclude_methods = exclude_methods or []
        all_results = self.calculate_all_methods(text1, text2)
        
        best_method = None
        best_score = -1.0
        
        for category, methods in all_results.items():
            for method, score in methods.items():
                if method in exclude_methods or score is None:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_method = method
        
        return best_method, best_score
    
    def validate_method(self, method: str) -> bool:
        """Validate if a similarity method is supported"""
        return method in self.all_methods
    
    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get information about a specific method"""
        if not self.validate_method(method):
            return {'valid': False, 'error': 'Method not found'}
        
        # Find category
        category = None
        for cat, methods in self.method_categories.items():
            if method in methods:
                category = cat
                break
        
        method_descriptions = {
            # String-based
            'levenshtein': 'Edit distance based similarity',
            'jaro': 'Jaro similarity for short strings',
            'sequence_matcher': 'Python difflib sequence matching',
            'lcs': 'Longest Common Subsequence similarity',
            
            # Set-based
            'jaccard': 'Jaccard coefficient (intersection over union)',
            'dice': 'Dice coefficient (2*intersection over sum)',
            'overlap': 'Overlap coefficient',
            'cosine_sets': 'Cosine similarity for sets',
            
            # Vector-based
            'cosine_tfidf': 'Cosine similarity with TF-IDF vectors',
            'cosine_count': 'Cosine similarity with count vectors',
            'euclidean': 'Euclidean distance based similarity',
            'manhattan': 'Manhattan distance based similarity',
            
            # Semantic-based
            'simple': 'Simple semantic similarity (no models)',
            'word2vec': 'Word2Vec embeddings similarity',
            'fasttext': 'FastText embeddings similarity',
            'phobert': 'PhoBERT transformer similarity'
        }
        
        return {
            'valid': True,
            'method': method,
            'category': category,
            'description': method_descriptions.get(method, 'No description available'),
            'requires_tokens': category in ['set_based', 'semantic_based'],
            'requires_models': method in ['word2vec', 'fasttext', 'phobert']
        }

class SimilarityBenchmark:
    """Benchmark and compare different similarity methods"""
    
    def __init__(self):
        self.calculator = SimilarityCalculator()
        self.logger = logging.getLogger(__name__ + '.SimilarityBenchmark')
    
    def benchmark_methods(self, text_pairs: List[Tuple[str, str]], 
                         methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark similarity methods on a set of text pairs
        
        Args:
            text_pairs: List of (text1, text2) tuples
            methods: Methods to benchmark (None for all)
            
        Returns:
            Benchmark results with timing and statistics
        """
        import time
        
        if methods is None:
            methods = list(self.calculator.all_methods)
        
        results = {
            'methods': {},
            'text_pairs_count': len(text_pairs),
            'benchmark_timestamp': time.time()
        }
        
        for method in methods:
            if not self.calculator.validate_method(method):
                continue
            
            method_results = {
                'scores': [],
                'execution_times': [],
                'errors': 0
            }
            
            for text1, text2 in text_pairs:
                start_time = time.time()
                
                try:
                    score = self.calculator.calculate(text1, text2, method)
                    method_results['scores'].append(score)
                except Exception as e:
                    self.logger.error(f"Error in {method} for pair: {e}")
                    method_results['errors'] += 1
                    method_results['scores'].append(None)
                
                execution_time = time.time() - start_time
                method_results['execution_times'].append(execution_time)
            
            # Calculate statistics
            valid_scores = [s for s in method_results['scores'] if s is not None]
            valid_times = method_results['execution_times']
            
            method_results['statistics'] = {
                'avg_score': sum(valid_scores) / len(valid_scores) if valid_scores else 0,
                'min_score': min(valid_scores) if valid_scores else 0,
                'max_score': max(valid_scores) if valid_scores else 0,
                'avg_execution_time': sum(valid_times) / len(valid_times),
                'total_execution_time': sum(valid_times),
                'success_rate': (len(valid_scores) / len(text_pairs)) * 100
            }
            
            results['methods'][method] = method_results
        
        return results

# Package-level convenience functions
def get_available_methods() -> Dict[str, List[str]]:
    """Get all available similarity methods grouped by category"""
    calculator = SimilarityCalculator()
    return calculator.method_categories.copy()

def validate_method(method: str) -> bool:
    """Check if a similarity method is valid"""
    calculator = SimilarityCalculator()
    return calculator.validate_method(method)

def calculate_similarity(text1: str, text2: str, method: str = 'cosine_tfidf') -> float:
    """
    Convenience function to calculate similarity
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    calculator = SimilarityCalculator()
    return calculator.calculate(text1, text2, method)

# Package initialization
logger.info(f"Similarity Measures Package v{__version__} initialized")
logger.info(f"Available methods: {len(get_available_methods())} categories")

# Optional: Check if required dependencies are available
try:
    import sklearn
    logger.info("scikit-learn available - vector methods enabled")
except ImportError:
    logger.warning("scikit-learn not available - vector methods may not work")

try:
    import transformers
    logger.info("transformers available - PhoBERT methods enabled")
except ImportError:
    logger.warning("transformers not available - PhoBERT methods disabled")