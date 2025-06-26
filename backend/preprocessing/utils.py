import re
import string
import unicodedata
from typing import List, Dict, Set, Tuple, Optional
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)

class VietnameseStopWords:
    """Vietnamese stop words handler"""
    
    # Common Vietnamese stop words
    STOP_WORDS = {
        'à', 'ạ', 'ah', 'ái', 'ào', 'ạo', 'bằng', 'bị', 'bởi', 'bộ', 'bù', 'có', 'cả', 'các', 'cái', 'của', 'cho', 'chúng', 'chúng_ta', 'chúng_tôi',
        'còn', 'cũng', 'da', 'dưới', 'để', 'được', 'đã', 'đang', 'đây', 'đó', 'đã', 'đến', 'đều', 'đi', 'đó', 'đó', 'đó', 'đó',
        'gì', 'giờ', 'hay', 'hơn', 'hoặc', 'khi', 'không', 'là', 'lại', 'lần', 'lúc', 'mà', 'mình', 'một', 'này', 'nào', 'nếu', 'những',
        'như', 'nhưng', 'ở', 'rồi', 'sẽ', 'tại', 'thì', 'thế', 'tôi', 'tới', 'trên', 'trong', 'và', 'với', 'về', 'vì', 'vào', 'vậy'
    }
    
    @classmethod
    def is_stop_word(cls, word: str) -> bool:
        """Check if a word is a stop word"""
        return word.lower().strip() in cls.STOP_WORDS
    
    @classmethod
    def remove_stop_words(cls, tokens: List[str]) -> List[str]:
        """Remove stop words from token list"""
        return [token for token in tokens if not cls.is_stop_word(token)]
    
    @classmethod
    def add_stop_words(cls, words: List[str]) -> None:
        """Add custom stop words"""
        cls.STOP_WORDS.update(word.lower().strip() for word in words)
    
    @classmethod
    def get_stop_words(cls) -> Set[str]:
        """Get all stop words"""
        return cls.STOP_WORDS.copy()

class TextPreprocessor:
    """Advanced text preprocessing utilities for Vietnamese"""
    
    def __init__(self, remove_stop_words: bool = True, 
                 normalize_unicode: bool = True,
                 remove_punctuation: bool = True):
        self.remove_stop_words = remove_stop_words
        self.normalize_unicode = normalize_unicode
        self.remove_punctuation = remove_punctuation
        self.stop_words_handler = VietnameseStopWords()
    
    def normalize_unicode_text(self, text: str) -> str:
        """Normalize Unicode characters in Vietnamese text"""
        if not text:
            return ""
        
        # Normalize Unicode (NFC normalization)
        text = unicodedata.normalize('NFC', text)
        
        # Fix common Vietnamese character issues
        replacements = {
            'à': 'à', 'á': 'á', 'ả': 'ả', 'ã': 'ã', 'ạ': 'ạ',
            'ă': 'ă', 'ằ': 'ằ', 'ắ': 'ắ', 'ẳ': 'ẳ', 'ẵ': 'ẵ', 'ặ': 'ặ',
            'â': 'â', 'ầ': 'ầ', 'ấ': 'ấ', 'ẩ': 'ẩ', 'ẫ': 'ẫ', 'ậ': 'ậ',
            'è': 'è', 'é': 'é', 'ẻ': 'ẻ', 'ẽ': 'ẽ', 'ẹ': 'ẹ',
            'ê': 'ê', 'ề': 'ề', 'ế': 'ế', 'ể': 'ể', 'ễ': 'ễ', 'ệ': 'ệ',
            'ì': 'ì', 'í': 'í', 'ỉ': 'ỉ', 'ĩ': 'ĩ', 'ị': 'ị',
            'ò': 'ò', 'ó': 'ó', 'ỏ': 'ỏ', 'õ': 'õ', 'ọ': 'ọ',
            'ô': 'ô', 'ồ': 'ồ', 'ố': 'ố', 'ổ': 'ổ', 'ỗ': 'ỗ', 'ộ': 'ộ',
            'ơ': 'ơ', 'ờ': 'ờ', 'ớ': 'ớ', 'ở': 'ở', 'ỡ': 'ỡ', 'ợ': 'ợ',
            'ù': 'ù', 'ú': 'ú', 'ủ': 'ủ', 'ũ': 'ũ', 'ụ': 'ụ',
            'ư': 'ư', 'ừ': 'ừ', 'ứ': 'ứ', 'ử': 'ử', 'ữ': 'ữ', 'ự': 'ự',
            'ỳ': 'ỳ', 'ý': 'ý', 'ỷ': 'ỷ', 'ỹ': 'ỹ', 'ỵ': 'ỵ',
            'đ': 'đ', 'Đ': 'Đ'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove Vietnamese phone numbers"""
        phone_patterns = [
            r'\b(0|\+84)[0-9]{8,10}\b',  # Vietnamese phone format
            r'\b[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'  # General phone format
        ]
        
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_extra_punctuation(self, text: str) -> str:
        """Remove or normalize extra punctuation"""
        # Remove repeated punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[-]{2,}', '-', text)
        
        # Remove punctuation at the end (optional)
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def process_numbers(self, text: str, method: str = 'keep') -> str:
        """Process numbers in text
        
        Args:
            method: 'keep', 'remove', 'replace_with_token'
        """
        if method == 'remove':
            text = re.sub(r'\b\d+\b', '', text)
        elif method == 'replace_with_token':
            text = re.sub(r'\b\d+\b', '<NUMBER>', text)
        # 'keep' - do nothing
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Main text cleaning pipeline"""
        if not text:
            return ""
        
        # Step 1: Unicode normalization
        if self.normalize_unicode:
            text = self.normalize_unicode_text(text)
        
        # Step 2: Remove HTML, URLs, emails, phones
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        
        # Step 3: Convert to lowercase
        text = text.lower()
        
        # Step 4: Handle punctuation
        text = self.remove_extra_punctuation(text)
        
        # Step 5: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        return text
    
    def process_tokens(self, tokens: List[str]) -> List[str]:
        """Process a list of tokens"""
        processed_tokens = []
        
        for token in tokens:
            # Skip empty tokens
            if not token.strip():
                continue
            
            # Clean individual token
            clean_token = self.clean_text(token)
            
            if not clean_token:
                continue
            
            # Remove stop words if enabled
            if self.remove_stop_words and self.stop_words_handler.is_stop_word(clean_token):
                continue
            
            processed_tokens.append(clean_token)
        
        return processed_tokens

class TextStatistics:
    """Calculate various text statistics"""
    
    @staticmethod
    def calculate_basic_stats(text: str, tokens: List[str]) -> Dict[str, any]:
        """Calculate basic text statistics"""
        return {
            'character_count': len(text),
            'character_count_no_spaces': len(text.replace(' ', '')),
            'word_count': len(tokens),
            'sentence_count': len(re.split(r'[.!?]+', text)) - 1,
            'paragraph_count': len(text.split('\n\n')),
            'average_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'longest_word': max(tokens, key=len) if tokens else '',
            'shortest_word': min(tokens, key=len) if tokens else ''
        }
    
    @staticmethod
    def calculate_readability_stats(tokens: List[str]) -> Dict[str, float]:
        """Calculate readability statistics"""
        if not tokens:
            return {'lexical_diversity': 0.0, 'word_frequency_stats': {}}
        
        word_freq = Counter(tokens)
        unique_words = len(word_freq)
        total_words = len(tokens)
        
        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0.0
        
        # Most common words
        most_common = word_freq.most_common(10)
        
        return {
            'lexical_diversity': lexical_diversity,
            'unique_word_count': unique_words,
            'total_word_count': total_words,
            'most_common_words': most_common,
            'hapax_legomena': sum(1 for count in word_freq.values() if count == 1)  # Words appearing only once
        }
    
    @staticmethod
    def calculate_similarity_stats(tokens1: List[str], tokens2: List[str]) -> Dict[str, any]:
        """Calculate statistics for text comparison"""
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return {
            'common_words': list(intersection),
            'common_word_count': len(intersection),
            'unique_to_text1': list(set1 - set2),
            'unique_to_text2': list(set2 - set1),
            'total_unique_words': len(union),
            'overlap_ratio': len(intersection) / len(union) if union else 0.0,
            'length_ratio': len(tokens1) / len(tokens2) if tokens2 else float('inf')
        }

class AdvancedTextPreprocessor:
    """Advanced preprocessing with configurable pipeline"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.stats = TextStatistics()
        self.pipeline_steps = []
    
    def add_step(self, step_name: str, step_function, **kwargs):
        """Add a preprocessing step to the pipeline"""
        self.pipeline_steps.append({
            'name': step_name,
            'function': step_function,
            'kwargs': kwargs
        })
    
    def remove_step(self, step_name: str):
        """Remove a preprocessing step"""
        self.pipeline_steps = [step for step in self.pipeline_steps if step['name'] != step_name]
    
    def process_pipeline(self, text: str) -> Tuple[str, Dict[str, any]]:
        """Process text through the configured pipeline"""
        processed_text = text
        stats = {'original_length': len(text)}
        
        for step in self.pipeline_steps:
            try:
                processed_text = step['function'](processed_text, **step['kwargs'])
                stats[f"{step['name']}_length"] = len(processed_text)
            except Exception as e:
                logger.error(f"Error in preprocessing step {step['name']}: {e}")
        
        stats['final_length'] = len(processed_text)
        stats['reduction_ratio'] = (len(text) - len(processed_text)) / len(text) if text else 0
        
        return processed_text, stats
    
    def get_default_pipeline(self) -> 'AdvancedTextPreprocessor':
        """Set up default preprocessing pipeline"""
        self.add_step('clean', self.preprocessor.clean_text)
        self.add_step('normalize_whitespace', self.preprocessor.normalize_whitespace)
        return self