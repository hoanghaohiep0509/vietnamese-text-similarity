from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import logging
from typing import Dict, List, Any, Optional
import time
from functools import wraps
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.tokenizer import VietnameseTokenizer
from similarity_measures import SimilarityCalculator
from .file_handler import FileHandler

# Configure logging
logger = logging.getLogger(__name__)

def timing_decorator(f):
    """Decorator to measure execution time"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Add timing info to response if it's a tuple or dict
        if isinstance(result, tuple) and len(result) == 2:
            response, status_code = result
            if isinstance(response, dict):
                response['execution_time'] = round(execution_time, 4)
            return response, status_code
        
        return result
    return wrapper

class TextSimilarityAPI:
    """Enhanced API class for text similarity analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration if none provided
        if config is None:
            config = {
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
                }
            }
        
        self.config = config
        self.app = Flask(__name__)
        
        # Configure CORS
        CORS(self.app, 
             origins="*",
             methods=["GET", "POST", "OPTIONS"],
             allow_headers=["Content-Type", "Authorization"])
        
        self.app.config['JSON_AS_ASCII'] = False  # Support Vietnamese characters in JSON responses
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  # Pretty print JSON responses
        self.app.config['JSON_SORT_KEYS'] = False  # Do not sort keys in JSON responses
        self.app.config['TRAP_HTTP_EXCEPTIONS'] = True  # Enable error handling
        self.app.config['TRAP_BAD_REQUEST_ERRORS'] = True  # Enable bad request error handling
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.get('api', {}).get('max_content_length', 16 * 1024 * 1024)  # 16 MB
        self.app.config['PROPAGATE_EXCEPTIONS'] = True  # Propagate exceptions to error handlers

        # Initialize components
        self.tokenizer = VietnameseTokenizer()
        
        # Initialize similarity calculator with semantic models if enabled
        semantic_enabled = self.config.get('similarity', {}).get('enable_semantic_models', False)
        self.similarity_calculator = SimilarityCalculator(enable_semantic_models=semantic_enabled)
        
        self.file_handler = FileHandler()
        
        # Request statistics
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': time.time()
        }
        
        self.datasets = self.load_datasets()  # Thay đổi tên method
        
        # Setup routes
        self.setup_routes()
        self.setup_error_handlers()
        
        logger.info("TextSimilarityAPI initialized")
    
    def setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested endpoint does not exist'
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/', methods=['GET'])
        def root():
            """Root endpoint"""
            return jsonify({
                'message': 'Vietnamese Text Similarity API',
                'status': 'running',
                'version': '1.0.0',
                'endpoints': [
                    '/api/health',
                    '/api/dataset', 
                    '/api/methods',
                    '/api/similarity'
                ]
            })
        
        @self.app.route('/api', methods=['GET'])
        def test():
            """Simple test endpoint"""
            return jsonify({
                'status': 'success',
                'message': 'API is working!',
                'timestamp': time.time()
            })

        # Log requests and responses
        @self.app.before_request
        def before_request():
            """Log requests and update statistics"""
            self.request_stats['total_requests'] += 1
            logger.info(f"Incoming request: {request.method} {request.path}")
        
        @self.app.after_request
        def after_request(response):
            """Log responses and update statistics"""
            if response.status_code < 400:
                self.request_stats['successful_requests'] += 1
            else:
                self.request_stats['failed_requests'] += 1
            
            logger.info(f"Response: {response.status_code}")
            return response
        
        @self.app.route('/api/health', methods=['GET'])
        @timing_decorator
        def health_check():
            """Health check endpoint"""
            uptime = time.time() - self.request_stats['start_time']
            
            return jsonify({
                'status': 'healthy',
                'message': 'Text Similarity API is running',
                'uptime_seconds': round(uptime, 2),
                'statistics': self.request_stats.copy()
            })
        
        @self.app.route('/api/tokenize', methods=['POST'])
        @timing_decorator
        def tokenize_text():
            """Tokenize text endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'text' not in data:
                    return jsonify({'error': 'Text is required'}), 400
                
                text = data['text']
                method = data.get('method', self.config.get('similarity', {}).get('default_tokenizer', 'underthesea'))
                
                # Check text length
                max_length = self.config.get('similarity', {}).get('max_text_length', 10000)
                if len(text) > max_length:
                    return jsonify({
                        'error': f'Text too long. Maximum length is {max_length} characters'
                    }), 400
                
                tokens = self.tokenizer.tokenize(text, method)
                
                return jsonify({
                    'tokens': tokens,
                    'count': len(tokens),
                    'method': method
                })
                
            except Exception as e:
                logger.error(f"Error in tokenization: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/similarity', methods=['POST'])
        @timing_decorator
        def calculate_similarity():
            """Calculate similarity endpoint"""
            try:
                data = request.get_json()
                
                # Validate input
                required_fields = ['text1', 'text2']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'{field} is required'}), 400
                
                text1 = data['text1']
                text2 = data['text2']
                similarity_method = data.get('similarity_method', 
                    self.config.get('similarity', {}).get('default_similarity_method', 'cosine_tfidf'))
                tokenize_method = data.get('tokenize_method', 
                    self.config.get('similarity', {}).get('default_tokenizer', 'underthesea'))
                
                # Check text lengths
                max_length = self.config.get('similarity', {}).get('max_text_length', 10000)
                if len(text1) > max_length or len(text2) > max_length:
                    return jsonify({
                        'error': f'Text too long. Maximum length is {max_length} characters'
                    }), 400
                
                # Validate similarity method
                if not self.similarity_calculator.validate_method(similarity_method):
                    available_methods = self.similarity_calculator.method_categories
                    return jsonify({
                        'error': f'Invalid similarity method: {similarity_method}',
                        'available_methods': available_methods
                    }), 400
                
                # Tokenize texts
                tokens1 = self.tokenizer.tokenize(text1, tokenize_method)
                tokens2 = self.tokenizer.tokenize(text2, tokenize_method)
                
                # Calculate similarity
                similarity_score = self.similarity_calculator.calculate(
                    text1, text2, similarity_method, tokens1, tokens2
                )
                
                return jsonify({
                    'similarity_score': similarity_score,
                    'text1_tokens': tokens1,
                    'text2_tokens': tokens2,
                    'similarity_method': similarity_method,
                    'tokenize_method': tokenize_method,
                    'text1_length': len(tokens1),
                    'text2_length': len(tokens2)
                })
                
            except Exception as e:
                logger.error(f"Error in similarity calculation: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/categories', methods=['GET'])
        def get_categories():
            """Get all unique categories from dataset"""
            try:
                categories = list(set(d.get('category') for d in self.datasets))
                return jsonify({
                    'status': 'success',
                    'data': categories
                })
            except Exception as e:
                logger.error(f"Error getting categories: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500

        @self.app.route('/api/labels', methods=['GET'])
        def get_labels():
            """Get all unique labels from dataset"""
            try:
                labels = list(set(d.get('label') for d in self.datasets))
                return jsonify({
                    'status': 'success',
                    'data': labels
                })
            except Exception as e:
                logger.error(f"Error getting labels: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500

        @self.app.route('/api/methods', methods=['GET'])
        def get_available_methods():
            """Get available methods"""
            try:
                tokenizer_methods = self.tokenizer.get_available_methods()
                
                # Debug log
                logger.info(f"Available tokenizer methods: {tokenizer_methods}")
                
                return jsonify({
                    'success': True,
                    'tokenize_methods': tokenizer_methods,
                    'similarity_methods': self.similarity_calculator.method_categories
                })
            except Exception as e:
                logger.error(f"Error getting methods: {e}")
                # Fallback với tất cả methods
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'tokenize_methods': ['simple', 'underthesea', 'pyvi', 'rdrsegmenter', 'vncorenlp'],
                    'similarity_methods': {}
                })

        @self.app.route('/api/dataset', methods=['GET'])
        @timing_decorator
        def get_dataset():
            """Get dataset from CSV file"""
            try:
                # Lọc theo category và label nếu được chỉ định
                category = request.args.get('category')
                label = request.args.get('label')
                
                # Lấy dữ liệu từ datasets đã load
                filtered_data = self.datasets.copy()
                
                if category:
                    filtered_data = [d for d in filtered_data if d.get('category') == category]
                if label:
                    filtered_data = [d for d in filtered_data if d.get('label') == label]
                
                return jsonify({
                    'status': 'success',
                    'data': filtered_data,
                    'total': len(filtered_data)
                })
                
            except Exception as e:
                logger.error(f"Error in get_dataset: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500

        @self.app.route('/api/datasets/<int:dataset_id>', methods=['GET'])
        @timing_decorator
        def get_dataset_by_id(dataset_id):
            """Get specific dataset by ID"""
            try:
                dataset = next((d for d in self.datasets if d['id'] == dataset_id), None)
                
                if not dataset:
                    return jsonify({'error': 'Dataset not found'}), 404
                
                return jsonify({
                    'status': 'success',
                    'data': dataset
                })
                
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/batch_similarity', methods=['POST'])
        @timing_decorator
        def batch_similarity():
            """Calculate similarity for multiple text pairs"""
            try:
                data = request.get_json()
                
                if 'text_pairs' not in data:
                    return jsonify({'error': 'text_pairs is required'}), 400
                
                text_pairs = data['text_pairs']
                similarity_method = data.get('similarity_method', 
                    self.config.get('similarity', {}).get('default_similarity_method', 'cosine_tfidf'))
                tokenize_method = data.get('tokenize_method', 
                    self.config.get('similarity', {}).get('default_tokenizer', 'underthesea'))
                
                results = []
                
                for i, pair in enumerate(text_pairs):
                    if 'text1' not in pair or 'text2' not in pair:
                        continue
                    
                    try:
                        # Tokenize texts
                        tokens1 = self.tokenizer.tokenize(pair['text1'], tokenize_method)
                        tokens2 = self.tokenizer.tokenize(pair['text2'], tokenize_method)
                        
                        # Calculate similarity
                        similarity_score = self.similarity_calculator.calculate(
                            pair['text1'], pair['text2'], similarity_method, tokens1, tokens2
                        )
                        
                        result = {
                            'pair_id': pair.get('id', i),
                            'similarity_score': similarity_score,
                            'text1_length': len(tokens1),
                            'text2_length': len(tokens2),
                            'expected_label': pair.get('label'),
                            'category': pair.get('category')
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error processing pair {i}: {e}")
                        results.append({
                            'pair_id': pair.get('id', i),
                            'error': str(e)
                        })
                
                return jsonify({
                    'results': results,
                    'similarity_method': similarity_method,
                    'tokenize_method': tokenize_method,
                    'total_processed': len(results)
                })
                
            except Exception as e:
                logger.error(f"Error in batch similarity: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/compare_methods', methods=['POST'])
        @timing_decorator
        def compare_methods():
            """Compare all similarity methods on given text pairs"""
            try:
                data = request.get_json()
                
                # Can work with single pair or multiple pairs
                if 'text1' in data and 'text2' in data:
                    # Single pair
                    text_pairs = [{'text1': data['text1'], 'text2': data['text2']}]
                elif 'text_pairs' in data:
                    # Multiple pairs
                    text_pairs = data['text_pairs']
                else:
                    return jsonify({'error': 'Either text1/text2 or text_pairs is required'}), 400
                
                tokenize_method = data.get('tokenize_method', 'underthesea')
                
                # Get all available methods
                all_methods = []
                for methods in self.similarity_calculator.method_categories.values():
                    all_methods.extend(methods)
                
                results = {}
                
                for method in all_methods:
                    method_results = []
                    
                    for i, pair in enumerate(text_pairs):
                        try:
                            start_time = time.time()
                            
                            # Tokenize
                            tokens1 = self.tokenizer.tokenize(pair['text1'], tokenize_method)
                            tokens2 = self.tokenizer.tokenize(pair['text2'], tokenize_method)
                            
                            # Calculate similarity
                            similarity_score = self.similarity_calculator.calculate(
                                pair['text1'], pair['text2'], method, tokens1, tokens2
                            )
                            
                            execution_time = time.time() - start_time
                            
                            method_results.append({
                                'pair_id': pair.get('id', i),
                                'similarity_score': similarity_score,
                                'execution_time': execution_time,
                                'expected_label': pair.get('label')
                            })
                            
                        except Exception as e:
                            logger.error(f"Error with method {method} on pair {i}: {e}")
                            method_results.append({
                                'pair_id': pair.get('id', i),
                                'error': str(e)
                            })
                    
                    results[method] = {
                        'results': method_results,
                        'category': self.get_method_category(method),
                        'avg_execution_time': sum(r.get('execution_time', 0) for r in method_results) / len(method_results) if method_results else 0
                    }
                
                return jsonify({
                    'comparison_results': results,
                    'tokenize_method': tokenize_method,
                    'total_pairs': len(text_pairs),
                    'total_methods': len(all_methods)
                })
                
            except Exception as e:
                logger.error(f"Error in compare methods: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_statistics():
            """Get API usage statistics"""
            uptime = time.time() - self.request_stats['start_time']
            
            return jsonify({
                'uptime_seconds': round(uptime, 2),
                'total_requests': self.request_stats['total_requests'],
                'successful_requests': self.request_stats['successful_requests'],
                'failed_requests': self.request_stats['failed_requests'],
                'success_rate': round(
                    (self.request_stats['successful_requests'] / max(self.request_stats['total_requests'], 1)) * 100, 2
                ),
                'available_methods': {
                    'tokenizers': ['underthesea', 'pyvi', 'simple', 'rdrsegmenter', 'vncorenlp'],
                    'similarity_methods': self.similarity_calculator.method_categories
                }
            })
        
        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """Get API configuration"""
            return jsonify({
                'config': {
                    'max_text_length': self.config.get('similarity', {}).get('max_text_length', 10000),
                    'default_tokenizer': self.config.get('similarity', {}).get('default_tokenizer', 'underthesea'),
                    'default_similarity_method': self.config.get('similarity', {}).get('default_similarity_method', 'cosine_tfidf'),
                    'semantic_models_enabled': self.config.get('similarity', {}).get('enable_semantic_models', False)
                },
                'version': '1.0.0',
                'api_info': {
                    'name': 'Vietnamese Text Similarity API',
                    'description': 'API for calculating text similarity using various methods',
                    'endpoints': self.get_available_endpoints()
                }
            })

    def load_datasets(self):
        """Load datasets from CSV file"""
        try:
            # Đường dẫn đến file CSV
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'data', 'test_dataset', 'dataset.csv'
            )
            
            if not os.path.exists(csv_path):
                logger.error(f"Dataset file not found: {csv_path}")
                return []
            
            # Đọc CSV
            df = pd.read_csv(csv_path)
            
            # Chuyển DataFrame thành list of dictionaries
            datasets = df.to_dict('records')
            logger.info(f"Successfully loaded {len(datasets)} datasets from CSV")
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return []

    def get_method_category(self, method):
        """Get category for a similarity method"""
        for category, methods in self.similarity_calculator.method_categories.items():
            if method in methods:
                return category
        return 'unknown'
    
    def get_available_endpoints(self):
        """Get list of available API endpoints"""
        return [
            {'method': 'GET', 'path': '/api/health', 'description': 'Health check'},
            {'method': 'GET', 'path': '/api/methods', 'description': 'Available methods'},
            {'method': 'GET', 'path': '/api/dataset', 'description': 'Get test datasets'},
            {'method': 'GET', 'path': '/api/datasets/<id>', 'description': 'Get specific dataset'},
            {'method': 'POST', 'path': '/api/similarity', 'description': 'Calculate similarity'},
            {'method': 'POST', 'path': '/api/tokenize', 'description': 'Tokenize text'},
            {'method': 'POST', 'path': '/api/batch_similarity', 'description': 'Batch similarity calculation'},
            {'method': 'POST', 'path': '/api/compare_methods', 'description': 'Compare all methods'},
            {'method': 'GET', 'path': '/api/stats', 'description': 'API statistics'},
            {'method': 'GET', 'path': '/api/config', 'description': 'API configuration'}
        ]
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None, 
            debug: Optional[bool] = None):
        """Run the Flask application"""
        
        # Use config values as defaults
        api_config = self.config.get('api', {})
        
        if host is None:
            host = api_config.get('host', '0.0.0.0')
        if port is None:
            port = api_config.get('port', 5000)
        if debug is None:
            debug = api_config.get('debug', True)
        
        logger.info(f"Starting Text Similarity API on {host}:{port} (debug={debug})")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            raise

# Remove the problematic default instance creation
api = None