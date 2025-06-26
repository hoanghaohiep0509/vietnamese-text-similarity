# -*- coding: utf-8 -*-
"""
Vietnamese Tokenizer Module
This module provides a tokenizer for Vietnamese text using various methods including Underthesea, PyVi, RDRSegmenter (VnCoreNLP), and a simple whitespace-based tokenizer. 

Vietnamese Text Tokenizer
Supports multiple tokenization methods for Vietnamese text including:
- Simple (không tách): Basic whitespace tokenization
- VnCoreNLP: Vietnamese Core NLP toolkit
- Underthesea: Vietnamese NLP library
- PyVi: Fast Vietnamese text processing
- RDRSegmenter: Rule-based segmentation
"""  

import logging
import re
import os
import sys
from typing import List, Optional, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VietnameseTokenizer:
    """Vietnamese text tokenizer with multiple methods"""
    
    def __init__(self):
        """Initialize tokenizer with available methods"""
        self.available_methods = ['simple', 'vncorenlp', 'underthesea', 'pyvi', 'rdrsegmenter']
        self.tokenizers = {}
        self.method_info = {}
        self._init_tokenizers()
    
    def _init_tokenizers(self):
        """Initialize all available tokenizers"""
        print("🚀 Initializing Vietnamese Tokenizers...")
        
        # Initialize tokenizer availability
        self.tokenizers = {
            'simple': True,  # Always available
            'vncorenlp': False,
            'underthesea': False,
            'pyvi': False,
            'rdrsegmenter': False
        }
        
        # 1. Simple Tokenizer (always available)
        self._init_simple_tokenizer()
        
        # 2. VnCoreNLP Tokenizer
        self._init_vncorenlp_tokenizer()
        
        # 3. Underthesea Tokenizer
        self._init_underthesea_tokenizer()
        
        # 4. PyVi Tokenizer
        self._init_pyvi_tokenizer()
        
        # 5. RDRSegmenter Tokenizer
        self._init_rdrsegmenter_tokenizer()
        
        # Update available methods
        self.available_methods = [
            method for method in self.available_methods 
            if self.tokenizers.get(method, False)
        ]
        
        print(f"✅ Available tokenization methods: {self.available_methods}")
        print(f"📊 Tokenizers status: {self.tokenizers}")
        
        logger.info(f"Initialized tokenizers: {self.available_methods}")
    
    def _init_simple_tokenizer(self):
        """Initialize Simple tokenizer (always available)"""
        try:
            self.tokenizers['simple'] = True
            self.method_info['simple'] = {
                'name': 'Simple Tokenizer (Không tách)',
                'description': 'Tách từ đơn giản dựa trên khoảng trắng và dấu câu',
                'status': 'available',
                'pros': ['Tốc độ rất nhanh', 'Không cần thư viện', 'Luôn khả dụng'],
                'cons': ['Độ chính xác thấp', 'Không hiểu ngữ cảnh tiếng Việt']
            }
            print("✅ Simple Tokenizer: LOADED")
        except Exception as e:
            print(f"❌ Simple Tokenizer: FAILED - {e}")

    
    def _init_vncorenlp_tokenizer(self):
        """Initialize VnCoreNLP tokenizer"""
        try:
            import vncorenlp
            
            # Try to initialize VnCoreNLP
            vncorenlp_dir = './vncorenlp'
            if not os.path.exists(vncorenlp_dir):
                os.makedirs(vncorenlp_dir)
            
            self.vncorenlp = vncorenlp.VnCoreNLP(
                annotators=["wseg"], 
                max_heap_size='-Xmx500m'
            )
            return True
        except Exception as e:
            print(f"❌ VnCoreNLP: FAILED - {e}")
            return False


            self.tokenizers['vncorenlp'] = True
            self.method_info['vncorenlp'] = {
                'name': 'VnCoreNLP',
                'description': 'Bộ công cụ xử lý ngôn ngữ tiếng Việt toàn diện',
                'status': 'available',
                'pros': ['Độ chính xác cao', 'Hỗ trợ nhiều NLP tasks', 'Được phát triển tại VN'],
                'cons': ['Cần tải models', 'Tốn bộ nhớ', 'Setup phức tạp']
            }
            print("✅ VnCoreNLP: LOADED")
            
        except ImportError as e:
            print(f"⚠️ VnCoreNLP: NOT INSTALLED - {e}")
            self.vncorenlp = None
            self.method_info['vncorenlp'] = {
                'name': 'VnCoreNLP',
                'status': 'not_installed',
                'error': 'Chưa cài đặt thư viện vncorenlp'
            }
        except Exception as e:
            print(f"❌ VnCoreNLP: FAILED - {e}")
            self.vncorenlp = None
            self.method_info['vncorenlp'] = {
                'name': 'VnCoreNLP',
                'status': 'error',
                'error': str(e)
            }   
    
    def _init_underthesea_tokenizer(self):
        """Initialize Underthesea tokenizer"""
        try:
            import underthesea
            
            # Test tokenization
            test_result = underthesea.word_tokenize("Việt Nam")
            
            self.underthesea = underthesea
            self.tokenizers['underthesea'] = True
            self.method_info['underthesea'] = {
                'name': 'Underthesea',
                'description': 'Thư viện NLP tiếng Việt phổ biến và mạnh mẽ',
                'status': 'available',
                'pros': ['Độ chính xác cao', 'Nhiều tính năng NLP', 'Cộng đồng tích cực'],
                'cons': ['Tốc độ chậm hơn', 'Cần cài đặt dependencies']
            }
            print("✅ Underthesea: LOADED")
            
        except ImportError as e:
            print(f"⚠️ Underthesea: NOT INSTALLED - {e}")
            self.underthesea = None
            self.method_info['underthesea'] = {
                'name': 'Underthesea',
                'status': 'not_installed',
                'error': 'Chưa cài đặt thư viện underthesea'
            }
        except Exception as e:
            print(f"❌ Underthesea: FAILED - {e}")
            self.underthesea = None
            self.method_info['underthesea'] = {
                'name': 'Underthesea',
                'status': 'error',
                'error': str(e)
            }
    
    def _init_pyvi_tokenizer(self):
        """Initialize PyVi tokenizer"""
        try:
            from pyvi import ViTokenizer
            
            # Test tokenization
            test_result = ViTokenizer.tokenize("Việt Nam")
            
            self.pyvi_tokenizer = ViTokenizer
            self.tokenizers['pyvi'] = True
            self.method_info['pyvi'] = {
                'name': 'PyVi',
                'description': 'Toolkit xử lý tiếng Việt nhanh và nhẹ',
                'status': 'available',
                'pros': ['Tốc độ nhanh', 'Nhẹ và đơn giản', 'Dễ tích hợp'],
                'cons': ['Độ chính xác thấp hơn', 'Ít tính năng hơn']
            }
            print("✅ PyVi: LOADED")
            
        except ImportError as e:
            print(f"⚠️ PyVi: NOT INSTALLED - {e}")
            self.pyvi_tokenizer = None
            self.method_info['pyvi'] = {
                'name': 'PyVi',
                'status': 'not_installed',
                'error': 'Chưa cài đặt thư viện pyvi'
            }
        except Exception as e:
            print(f"❌ PyVi: FAILED - {e}")
            self.pyvi_tokenizer = None
            self.method_info['pyvi'] = {
                'name': 'PyVi',
                'status': 'error',
                'error': str(e)
            }
    
    def _init_rdrsegmenter_tokenizer(self):
        """Initialize RDRSegmenter tokenizer"""
        try:
            # RDRSegmenter có thể được tích hợp trong VnCoreNLP hoặc riêng biệt
            # Thử import riêng trước
            try:
                import py_vncorenlp
                self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
                source = "py_vncorenlp"
            except:
                # Fallback to vncorenlp if available
                if hasattr(self, 'vncorenlp') and self.vncorenlp:
                    self.rdrsegmenter = self.vncorenlp
                    source = "vncorenlp"
                else:
                    raise ImportError("No RDRSegmenter implementation found")
            
            self.tokenizers['rdrsegmenter'] = True
            self.method_info['rdrsegmenter'] = {
                'name': 'RDRSegmenter',
                'description': 'Bộ tách từ dựa trên Ripple Down Rules',
                'status': 'available',
                'source': source,
                'pros': ['Dựa trên ML', 'Hiệu quả tốt', 'Nghiên cứu khoa học'],
                'cons': ['Cần setup', 'Phụ thuộc VnCoreNLP']
            }
            print("✅ RDRSegmenter: LOADED")
            
        except Exception as e:
            print(f"⚠️ RDRSegmenter: FAILED - {e}")
            self.rdrsegmenter = None
            self.method_info['rdrsegmenter'] = {
                'name': 'RDRSegmenter',
                'status': 'not_available',
                'error': str(e)
            }
    
    def get_available_methods(self) -> List[str]:
        """Get list of available tokenization methods"""
        return self.available_methods.copy()
    
    def get_method_info(self, method: str = None) -> Dict[str, Any]:
        """Get information about tokenization method(s)"""
        if method:
            return self.method_info.get(method, {
                'name': method,
                'status': 'unknown',
                'error': 'Method not found'
            })
        return self.method_info.copy()
    
    def tokenize(self, text: str, method: str = 'simple') -> List[str]:
        """
        Tokenize Vietnamese text using specified method
        
        Args:
            text (str): Input text to tokenize
            method (str): Tokenization method
        
        Returns:
            List[str]: List of tokens
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        try:
            if method == 'simple':
                return self._tokenize_simple(text)
            elif method == 'vncorenlp':
                return self._tokenize_vncorenlp(text)
            elif method == 'underthesea':
                return self._tokenize_underthesea(text)
            elif method == 'pyvi':
                return self._tokenize_pyvi(text)
            elif method == 'rdrsegmenter':
                return self._tokenize_rdrsegmenter(text)
            else:
                logger.warning(f"Unknown method {method}, using simple")
                return self._tokenize_simple(text)
                
        except Exception as e:
            logger.error(f"Error in tokenization with {method}: {e}")
            logger.info("Falling back to simple tokenization")
            return self._tokenize_simple(text)
    
    def _tokenize_simple(self, text: str) -> List[str]:
        """
        Simple tokenization (Không tách - basic splitting)
        
        Args:
            text (str): Input text
        Returns:
            List[str]: List of tokens
        """
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Split by whitespace and preserve punctuation
            tokens = re.findall(r'\S+', text)
            
            # Normalize to lowercase
            tokens = [token.lower() for token in tokens if token.strip()]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error in simple tokenization: {e}")
            return text.split()
    
    def _tokenize_vncorenlp(self, text: str) -> List[str]:
        """
        Tokenize using VnCoreNLP
        
        Args:
            text (str): Input text
        Returns:
            List[str]: List of tokens
        """
        if not self.tokenizers.get('vncorenlp', False) or not self.vncorenlp:
            raise ValueError("VnCoreNLP tokenizer not available")
        
        try:
            # Use VnCoreNLP word segmentation
            result = self.vncorenlp.tokenize(text)
            
            if not result:
                return self._tokenize_simple(text)
            
            # Extract tokens from the result
            tokens = []
            for sentence in result:
                if isinstance(sentence, list):
                    tokens.extend(sentence)
                elif isinstance(sentence, str):
                    tokens.append(sentence)
            
            # Normalize tokens
            tokens = [token.strip().lower() for token in tokens if token.strip()]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error in VnCoreNLP tokenization: {e}")
            raise
    
    def _tokenize_underthesea(self, text: str) -> List[str]:
        """
        Tokenize using Underthesea
        
        Args:
            text (str): Input text
        Returns:
            List[str]: List of tokens
        """
        if not self.tokenizers.get('underthesea', False) or not self.underthesea:
            raise ValueError("Underthesea tokenizer not available")
        
        try:
            # Use Underthesea word tokenization
            tokens = self.underthesea.word_tokenize(text)
            
            # Normalize tokens
            tokens = [token.strip().lower() for token in tokens if token.strip()]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error in Underthesea tokenization: {e}")
            raise
    
    def _tokenize_pyvi(self, text: str) -> List[str]:
        """
        Tokenize using PyVi
        
        Args:
            text (str): Input text
        Returns:
            List[str]: List of tokens
        """
        if not self.tokenizers.get('pyvi', False) or not self.pyvi_tokenizer:
            raise ValueError("PyVi tokenizer not available")
        
        try:
            # Use PyVi tokenizer
            tokenized_text = self.pyvi_tokenizer.tokenize(text)
            
            # Split the tokenized text and handle underscores
            tokens = tokenized_text.split()
            
            # Process tokens - PyVi uses underscores to connect word parts
            final_tokens = []
            for token in tokens:
                # Replace underscores with spaces and split
                if '_' in token:
                    # Keep compound words together but remove underscores
                    compound_word = token.replace('_', ' ')
                    final_tokens.append(compound_word.lower())
                else:
                    final_tokens.append(token.lower())
            
            return [token for token in final_tokens if token.strip()]
            
        except Exception as e:
            logger.error(f"Error in PyVi tokenization: {e}")
            raise
    
    def _tokenize_rdrsegmenter(self, text: str) -> List[str]:
        """
        Tokenize using RDRSegmenter
        
        Args:
            text (str): Input text
        Returns:
            List[str]: List of tokens
        """
        if not self.tokenizers.get('rdrsegmenter', False) or not self.rdrsegmenter:
            raise ValueError("RDRSegmenter tokenizer not available")
        
        try:
            # Use RDRSegmenter (through VnCoreNLP or py_vncorenlp)
            if hasattr(self.rdrsegmenter, 'tokenize'):
                result = self.rdrsegmenter.tokenize(text)
            elif hasattr(self.rdrsegmenter, 'annotate'):
                result = self.rdrsegmenter.annotate(text)
            else:
                # Fallback
                result = self.rdrsegmenter.segment(text)
            
            if not result:
                return self._tokenize_simple(text)
            
            # Extract tokens from the result
            tokens = []
            for sentence in result:
                if isinstance(sentence, list):
                    tokens.extend(sentence)
                elif isinstance(sentence, str):
                    tokens.append(sentence)
            
            # Normalize tokens
            tokens = [token.strip().lower() for token in tokens if token.strip()]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error in RDRSegmenter tokenization: {e}")
            raise
    
    def tokenize_batch(self, texts: List[str], method: str = 'simple') -> List[List[str]]:
        """
        Tokenize multiple texts
        
        Args:
            texts (List[str]): List of texts to tokenize
            method (str): Tokenization method
            
        Returns:
            List[List[str]]: List of token lists
        """
        results = []
        for text in texts:
            try:
                tokens = self.tokenize(text, method)
                results.append(tokens)
            except Exception as e:
                logger.error(f"Error tokenizing text '{text[:50]}...': {e}")
                results.append([])
        
        return results
    
    def compare_methods(self, text: str, methods: List[str] = None) -> Dict[str, List[str]]:
        """
        Compare tokenization results across different methods
        
        Args:
            text (str): Text to tokenize
            methods (List[str]): Methods to compare (default: all available)
            
        Returns:
            Dict[str, List[str]]: Results for each method
        """
        if methods is None:
            methods = self.available_methods
        
        results = {}
        for method in methods:
            if method in self.available_methods:
                try:
                    results[method] = self.tokenize(text, method)
                except Exception as e:
                    results[method] = f"Error: {e}"
            else:
                results[method] = "Method not available"
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tokenizer statistics
        
        Returns:
            Dict[str, Any]: Statistics about available tokenizers
        """
        return {
            'total_methods': len(self.tokenizers),
            'available_methods': len(self.available_methods),
            'methods_list': self.available_methods,
            'tokenizers_status': self.tokenizers.copy(),
            'method_info': self.method_info.copy(),
            'default_method': 'simple',
            'recommended_method': self._get_recommended_method()
        }
    
    def _get_recommended_method(self) -> str:
        """Get recommended tokenization method based on availability"""
        priority = ['underthesea', 'vncorenlp', 'pyvi', 'rdrsegmenter', 'simple']
        
        for method in priority:
            if method in self.available_methods:
                return method
        
        return 'simple'
    
    def validate_method(self, method: str) -> bool:
        """
        Validate if a tokenization method is available
        
        Args:
            method (str): Method name
            
        Returns:
            bool: True if method is available
        """
        return method in self.available_methods
    

def test_tokenizer():
    """Test function for the tokenizer"""
    print("🧪 Testing VietnameseTokenizer...")
    print("=" * 60)
    
    tokenizer = VietnameseTokenizer()
    
    test_text = "Việt Nam là một quốc gia xinh đẹp ở Đông Nam Á. Chúng ta yêu quê hương!"
    
    print(f"📝 Test text: {test_text}")
    print("=" * 60)
    
    # Test all available methods
    for method in tokenizer.get_available_methods():
        try:
            print(f"\n🔸 Method: {method.upper()}")
            tokens = tokenizer.tokenize(test_text, method)
            print(f"   Tokens ({len(tokens)}): {tokens}")
            
            # Get method info
            info = tokenizer.get_method_info(method)
            if 'description' in info:
                print(f"   Description: {info['description']}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Tokenizer Statistics:")
    stats = tokenizer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🔍 Method Comparison:")
    comparison = tokenizer.compare_methods("Hà Nội là thủ đô của Việt Nam.")
    for method, result in comparison.items():
        print(f"   {method}: {result}")




if __name__ == "__main__":
    test_tokenizer()