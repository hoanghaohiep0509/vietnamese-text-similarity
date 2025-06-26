from typing import List, Set
import math

class SetBasedSimilarity:
    """Set-based similarity measures"""
    
    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity coefficient"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def dice_similarity(set1: Set, set2: Set) -> float:
        """Calculate Dice similarity coefficient"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        return (2.0 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0
    
    @staticmethod
    def overlap_similarity(set1: Set, set2: Set) -> float:
        """Calculate Overlap similarity coefficient"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        min_size = min(len(set1), len(set2))
        
        return intersection / min_size if min_size > 0 else 0.0
    
    @staticmethod
    def cosine_similarity_sets(set1: Set, set2: Set) -> float:
        """Calculate Cosine similarity for sets"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        magnitude = math.sqrt(len(set1) * len(set2))
        
        return intersection / magnitude if magnitude > 0 else 0.0
    
    @staticmethod
    def calculate_similarity(tokens1: List[str], tokens2: List[str], method: str = "jaccard") -> float:
        """Main method to calculate set-based similarity"""
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        if method == "jaccard":
            return SetBasedSimilarity.jaccard_similarity(set1, set2)
        elif method == "dice":
            return SetBasedSimilarity.dice_similarity(set1, set2)
        elif method == "overlap":
            return SetBasedSimilarity.overlap_similarity(set1, set2)
        elif method == "cosine_sets":
            return SetBasedSimilarity.cosine_similarity_sets(set1, set2)
        else:
            raise ValueError(f"Unknown set similarity method: {method}")