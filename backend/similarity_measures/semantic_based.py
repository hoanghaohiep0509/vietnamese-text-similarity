import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SemanticBasedSimilarity:
    """Semantic-based similarity measures using embeddings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.word2vec_model = None
        self.fasttext_model = None
        self.phobert_tokenizer = None
        self.phobert_model = None
        
    def load_word2vec_model(self, model_path: Optional[str] = None):
        """Load Word2Vec model for Vietnamese"""
        try:
            from gensim.models import Word2Vec, KeyedVectors
            
            if model_path:
                self.word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            else:
                # Use a simple pre-trained model or train a basic one
                self.logger.info("No Word2Vec model path provided, using basic implementation")
                self.word2vec_model = None
                
        except Exception as e:
            self.logger.error(f"Error loading Word2Vec model: {e}")
            self.word2vec_model = None
    
    def load_fasttext_model(self, model_path: Optional[str] = None):
        """Load FastText model for Vietnamese"""
        try:
            import fasttext
            
            if model_path:
                self.fasttext_model = fasttext.load_model(model_path)
            else:
                self.logger.info("No FastText model path provided")
                self.fasttext_model = None
                
        except Exception as e:
            self.logger.error(f"Error loading FastText model: {e}")
            self.fasttext_model = None
    
    def load_phobert_model(self):
        """Load PhoBERT model for Vietnamese"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
            self.logger.info("PhoBERT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading PhoBERT model: {e}")
            self.phobert_tokenizer = None
            self.phobert_model = None
    
    def get_word_embedding(self, word: str, method: str = "word2vec") -> Optional[np.ndarray]:
        """Get word embedding using specified method"""
        try:
            if method == "word2vec" and self.word2vec_model:
                if word in self.word2vec_model:
                    return self.word2vec_model[word]
            elif method == "fasttext" and self.fasttext_model:
                return self.fasttext_model.get_word_vector(word)
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting embedding for word '{word}': {e}")
            return None
    
    def get_sentence_embedding_average(self, tokens: List[str], method: str = "word2vec") -> Optional[np.ndarray]:
        """Get sentence embedding by averaging word embeddings"""
        embeddings = []
        
        for token in tokens:
            embedding = self.get_word_embedding(token, method)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def get_phobert_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get sentence embedding using PhoBERT"""
        try:
            if not self.phobert_tokenizer or not self.phobert_model:
                self.logger.warning("PhoBERT model not loaded")
                return None
            
            import torch
            
            # Tokenize and encode
            inputs = self.phobert_tokenizer(text, return_tensors="pt", 
                                          padding=True, truncation=True, max_length=256)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return embeddings.numpy()
            
        except Exception as e:
            self.logger.error(f"Error getting PhoBERT embedding: {e}")
            return None
    
    def cosine_similarity_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Reshape for sklearn cosine_similarity
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def euclidean_similarity_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance-based similarity between embeddings"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert distance to similarity (normalize by vector length)
            max_distance = np.linalg.norm(embedding1) + np.linalg.norm(embedding2)
            
            if max_distance == 0:
                return 1.0
            
            similarity = 1 - (distance / max_distance)
            return max(0.0, float(similarity))
            
        except Exception as e:
            self.logger.error(f"Error calculating Euclidean similarity: {e}")
            return 0.0
    
    def word2vec_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity using Word2Vec embeddings"""
        if not self.word2vec_model:
            self.logger.warning("Word2Vec model not loaded")
            return 0.0
        
        emb1 = self.get_sentence_embedding_average(tokens1, "word2vec")
        emb2 = self.get_sentence_embedding_average(tokens2, "word2vec")
        
        return self.cosine_similarity_embeddings(emb1, emb2)
    
    def fasttext_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity using FastText embeddings"""
        if not self.fasttext_model:
            self.logger.warning("FastText model not loaded")
            return 0.0
        
        emb1 = self.get_sentence_embedding_average(tokens1, "fasttext")
        emb2 = self.get_sentence_embedding_average(tokens2, "fasttext")
        
        return self.cosine_similarity_embeddings(emb1, emb2)
    
    def phobert_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using PhoBERT embeddings"""
        if not self.phobert_model:
            self.logger.warning("PhoBERT model not loaded")
            return 0.0
        
        emb1 = self.get_phobert_embedding(text1)
        emb2 = self.get_phobert_embedding(text2)
        
        return self.cosine_similarity_embeddings(emb1, emb2)
    
    def simple_semantic_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Simple semantic similarity without pre-trained models"""
        # This is a fallback method when no pre-trained models are available
        # It uses basic word overlap with some weighting
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Jaccard similarity as base
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Weight by sentence length similarity
        len_sim = 1 - abs(len(tokens1) - len(tokens2)) / max(len(tokens1), len(tokens2), 1)
        
        # Combine similarities
        return (jaccard * 0.7 + len_sim * 0.3)
    
    def calculate_similarity(self, text1: str, text2: str, tokens1: List[str], 
                           tokens2: List[str], method: str = "simple") -> float:
        """Main method to calculate semantic-based similarity"""
        try:
            if method == "word2vec":
                return self.word2vec_similarity(tokens1, tokens2)
            elif method == "fasttext":
                return self.fasttext_similarity(tokens1, tokens2)
            elif method == "phobert":
                return self.phobert_similarity(text1, text2)
            elif method == "simple":
                return self.simple_semantic_similarity(tokens1, tokens2)
            else:
                self.logger.warning(f"Unknown semantic method {method}, using simple")
                return self.simple_semantic_similarity(tokens1, tokens2)
                
        except Exception as e:
            self.logger.error(f"Error in semantic similarity calculation: {e}")
            return self.simple_semantic_similarity(tokens1, tokens2)