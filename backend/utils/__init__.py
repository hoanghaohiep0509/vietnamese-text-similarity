"""
Utils Package for Vietnamese Text Similarity Project

This package provides utility functions and classes including:
- API endpoints and Flask application setup
- File handling operations (CSV, JSON, etc.)
- Common utilities and helper functions
- Configuration management
- Logging utilities

Usage:
    from utils import TextSimilarityAPI, FileHandler
    
    api = TextSimilarityAPI()
    api.run()
"""

from .api import TextSimilarityAPI, api
from .file_handler import FileHandler

import logging
import os
from typing import Dict, Any, Optional

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Utilities for Vietnamese Text Similarity Project"

# Package-level exports
__all__ = [
    'TextSimilarityAPI',
    'api',
    'FileHandler',
    'setup_logging',
    'get_project_config',
    'validate_config'
]

# Configure package logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def setup_logging(level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration for the entire project
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure basic logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # Add file handler if specified
    handlers = [console_handler]
    if log_file:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Get root logger and configure
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    logger.info(f"Logging configured: level={level}, file={log_file}")

def get_project_config() -> Dict[str, Any]:
    """
    Get project configuration settings
    
    Returns:
        Dictionary containing project configuration
    """
    config = {
        'project_name': 'Vietnamese Text Similarity',
        'version': __version__,
        'api': {
            'host': '127.0.0.1',
            'port': 5000,
            'debug': True,
            'cors_enabled': True
        },
        'similarity': {
            'default_tokenizer': 'underthesea',
            'default_similarity_method': 'cosine_tfidf',
            'enable_semantic_models': False,
            'max_text_length': 10000
        },
        'file_handling': {
            'max_file_size_mb': 50,
            'allowed_extensions': ['.txt', '.csv', '.json'],
            'upload_folder': 'uploads',
            'results_folder': 'results'
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/app.log',
            'max_file_size_mb': 10,
            'backup_count': 5
        },
        'performance': {
            'cache_enabled': True,
            'cache_size': 1000,
            'request_timeout': 30
        }
    }
    
    # Override with environment variables if available
    config = _load_env_config(config)
    
    return config

def _load_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration from environment variables"""
    import os
    
    # API configuration
    if os.getenv('API_HOST'):
        config['api']['host'] = os.getenv('API_HOST')
    
    if os.getenv('API_PORT'):
        try:
            config['api']['port'] = int(os.getenv('API_PORT'))
        except ValueError:
            logger.warning("Invalid API_PORT environment variable")
    
    if os.getenv('API_DEBUG'):
        config['api']['debug'] = os.getenv('API_DEBUG').lower() in ['true', '1', 'yes']
    
    # Similarity configuration
    if os.getenv('DEFAULT_TOKENIZER'):
        config['similarity']['default_tokenizer'] = os.getenv('DEFAULT_TOKENIZER')
    
    if os.getenv('DEFAULT_SIMILARITY_METHOD'):
        config['similarity']['default_similarity_method'] = os.getenv('DEFAULT_SIMILARITY_METHOD')
    
    if os.getenv('ENABLE_SEMANTIC_MODELS'):
        config['similarity']['enable_semantic_models'] = os.getenv('ENABLE_SEMANTIC_MODELS').lower() in ['true', '1', 'yes']
    
    # Logging configuration
    if os.getenv('LOG_LEVEL'):
        config['logging']['level'] = os.getenv('LOG_LEVEL').upper()
    
    if os.getenv('LOG_FILE'):
        config['logging']['file'] = os.getenv('LOG_FILE')
    
    return config

def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate project configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate API configuration
    if 'api' not in config:
        errors.append("Missing 'api' configuration section")
    else:
        api_config = config['api']
        
        if 'port' in api_config:
            try:
                port = int(api_config['port'])
                if not (1 <= port <= 65535):
                    errors.append("API port must be between 1 and 65535")
            except (ValueError, TypeError):
                errors.append("API port must be a valid integer")
    
    # Validate similarity configuration
    if 'similarity' not in config:
        errors.append("Missing 'similarity' configuration section")
    else:
        sim_config = config['similarity']
        
        valid_tokenizers = ['underthesea', 'pyvi', 'simple']
        if sim_config.get('default_tokenizer') not in valid_tokenizers:
            errors.append(f"Invalid default tokenizer. Must be one of: {valid_tokenizers}")
        
        # Import and validate similarity methods
        try:
            from similarity_measures import get_available_methods
            available_methods = get_available_methods()
            all_methods = []
            for methods in available_methods.values():
                all_methods.extend(methods)
            
            default_method = sim_config.get('default_similarity_method')
            if default_method and default_method not in all_methods:
                errors.append(f"Invalid default similarity method: {default_method}")
        
        except ImportError:
            logger.warning("Could not validate similarity methods - similarity_measures not available")
    
    # Validate file handling configuration
    if 'file_handling' in config:
        fh_config = config['file_handling']
        
        if 'max_file_size_mb' in fh_config:
            try:
                size = float(fh_config['max_file_size_mb'])
                if size <= 0:
                    errors.append("Max file size must be positive")
            except (ValueError, TypeError):
                errors.append("Max file size must be a valid number")
    
    # Validate logging configuration
    if 'logging' in config:
        log_config = config['logging']
        
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_config.get('level') not in valid_levels:
            errors.append(f"Invalid log level. Must be one of: {valid_levels}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

class ProjectManager:
    """Manage project-wide settings and initialization"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize project manager
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_project_config()
        
        if config_override:
            self._update_config(self.config, config_override)
        
        # Validate configuration
        is_valid, errors = validate_config(self.config)
        if not is_valid:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")
        
        # Setup logging
        log_config = self.config.get('logging', {})
        setup_logging(
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('file'),
        )
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("Project Manager initialized successfully")
    
    def _update_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """Recursively update configuration"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _create_directories(self) -> None:
        """Create necessary project directories"""
        directories = []
        
        # File handling directories
        fh_config = self.config.get('file_handling', {})
        if 'upload_folder' in fh_config:
            directories.append(fh_config['upload_folder'])
        if 'results_folder' in fh_config:
            directories.append(fh_config['results_folder'])
        
        # Logging directory
        log_config = self.config.get('logging', {})
        if 'file' in log_config:
            log_dir = os.path.dirname(log_config['file'])
            if log_dir:
                directories.append(log_dir)
        
        # Create directories
        for directory in directories:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Could not create directory {directory}: {e}")
    
    def get_api_instance(self) -> 'TextSimilarityAPI':
        """Get configured API instance"""
        return TextSimilarityAPI(config=self.config)
    
    def get_file_handler(self) -> 'FileHandler':
        """Get configured file handler"""
        return FileHandler()

# Package initialization
try:
    # Try to initialize with default configuration
    _default_manager = ProjectManager()
    logger.info(f"Utils Package v{__version__} initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize Utils Package: {e}")
    # Continue without default manager
    _default_manager = None

# Convenience function to get default manager
def get_default_manager() -> Optional[ProjectManager]:
    """Get the default project manager instance"""
    return _default_manager