import difflib
from typing import List, Tuple
import re

class StringBasedSimilarity:
    """String-based similarity measures"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return StringBasedSimilarity.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity (normalized)"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = StringBasedSimilarity.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def jaro_similarity(s1: str, s2: str) -> float:
        """Calculate Jaro similarity"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        match_distance = (max(len1, len2) // 2) - 1
        
        if match_distance < 0:
            match_distance = 0
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Find transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        return (matches / len1 + matches / len2 + 
                (matches - transpositions / 2) / matches) / 3.0
    
    @staticmethod
    def sequence_matcher_similarity(s1: str, s2: str) -> float:
        """Calculate similarity using difflib.SequenceMatcher"""
        return difflib.SequenceMatcher(None, s1, s2).ratio()
    
    @staticmethod
    def longest_common_subsequence(s1: str, s2: str) -> float:
        """Calculate LCS-based similarity"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return lcs_length / max(m, n)
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str, method: str = "levenshtein") -> float:
        """Main method to calculate string-based similarity"""
        if method == "levenshtein":
            return StringBasedSimilarity.levenshtein_similarity(text1, text2)
        elif method == "jaro":
            return StringBasedSimilarity.jaro_similarity(text1, text2)
        elif method == "sequence_matcher":
            return StringBasedSimilarity.sequence_matcher_similarity(text1, text2)
        elif method == "lcs":
            return StringBasedSimilarity.longest_common_subsequence(text1, text2)
        else:
            raise ValueError(f"Unknown string similarity method: {method}")