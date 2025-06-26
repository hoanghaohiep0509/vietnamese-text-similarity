import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

class VectorBasedSimilarity:
    """Vector-based similarity measures"""
    
    @staticmethod
    def create_tfidf_vectors(texts: List[str]) -> np.ndarray:
        """Create TF-IDF vectors from texts"""
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(texts)
    
    @staticmethod
    def create_count_vectors(texts: List[str]) -> np.ndarray:
        """Create count vectors from texts"""
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(texts)
    
    @staticmethod
    def cosine_similarity_tfidf(text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors"""
        if not text1.strip() and not text2.strip():
            return 1.0
        if not text1.strip() or not text2.strip():
            return 0.0
        
        texts = [text1, text2]
        tfidf_matrix = VectorBasedSimilarity.create_tfidf_vectors(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return float(similarity_matrix[0][1])
    
    @staticmethod
    def cosine_similarity_count(text1: str, text2: str) -> float:
        """Calculate cosine similarity using count vectors"""
        if not text1.strip() and not text2.strip():
            return 1.0
        if not text1.strip() or not text2.strip():
            return 0.0
        
        texts = [text1, text2]
        count_matrix = VectorBasedSimilarity.create_count_vectors(texts)
        similarity_matrix = cosine_similarity(count_matrix)
        
        return float(similarity_matrix[0][1])
    
    @staticmethod
    def euclidean_similarity(text1: str, text2: str) -> float:
        """Calculate Euclidean distance-based similarity"""
        if not text1.strip() and not text2.strip():
            return 1.0
        if not text1.strip() or not text2.strip():
            return 0.0
        
        texts = [text1, text2]
        tfidf_matrix = VectorBasedSimilarity.create_tfidf_vectors(texts)
        
        vector1 = tfidf_matrix[0].toarray().flatten()
        vector2 = tfidf_matrix[1].toarray().flatten()
        
        euclidean_distance = np.linalg.norm(vector1 - vector2)
        # Convert distance to similarity (0-1 range)
        max_distance = math.sqrt(2)  # Maximum possible distance for normalized vectors
        similarity = 1 - (euclidean_distance / max_distance)
        
        return max(0.0, similarity)
    
    @staticmethod
    def manhattan_similarity(text1: str, text2: str) -> float:
        """Calculate Manhattan distance-based similarity"""
        if not text1.strip() and not text2.strip():
            return 1.0
        if not text1.strip() or not text2.strip():
            return 0.0
        
        texts = [text1, text2]
        tfidf_matrix = VectorBasedSimilarity.create_tfidf_vectors(texts)
        
        vector1 = tfidf_matrix[0].toarray().flatten()
        vector2 = tfidf_matrix[1].toarray().flatten()
        
        manhattan_distance = np.sum(np.abs(vector1 - vector2))
        # Convert distance to similarity
        max_distance = 2.0  # Maximum possible Manhattan distance for normalized vectors
        similarity = 1 - (manhattan_distance / max_distance)
        
        return max(0.0, similarity)
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str, method: str = "cosine_tfidf") -> float:
        """Main method to calculate vector-based similarity"""
        if method == "cosine_tfidf":
            return VectorBasedSimilarity.cosine_similarity_tfidf(text1, text2)
        elif method == "cosine_count":
            return VectorBasedSimilarity.cosine_similarity_count(text1, text2)
        elif method == "euclidean":
            return VectorBasedSimilarity.euclidean_similarity(text1, text2)
        elif method == "manhattan":
            return VectorBasedSimilarity.manhattan_similarity(text1, text2)
        else:
            raise ValueError(f"Unknown vector similarity method: {method}")