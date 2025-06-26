"""
Main entry point for Vietnamese Text Similarity Backend

This module serves as the main entry point for the backend application,
handling initialization, configuration, and server startup.
"""
import pandas as pd
import sys
import os
import logging
import argparse
from typing import Optional, Dict, Any
import signal
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from utils import ProjectManager, get_project_config, setup_logging
    from utils.api import TextSimilarityAPI
    from utils.file_handler import FileHandler
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

# Global variables
app_instance = None
project_manager = None

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        cleanup_and_exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def cleanup_and_exit(exit_code: int = 0):
    """Cleanup resources and exit"""
    global app_instance, project_manager
    
    logger.info("Performing cleanup...")
    
    try:
        # Add any cleanup operations here
        if app_instance:
            logger.info("Shutting down API server...")
        
        if project_manager:
            logger.info("Cleaning up project manager...")
        
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        exit_code = 1
    
    finally:
        sys.exit(exit_code)

def validate_environment():
    """Validate the environment and dependencies"""
    logger.info("Validating environment...")
    
    errors = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        errors.append("Python 3.7 or higher is required")
    
    # Check required packages
    required_packages = [
        ('flask', 'flask'),
        ('flask_cors', 'flask_cors'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn')  # import name vs package name
    ]
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            logger.debug(f"✓ {package_name} is available")
        except ImportError:
            errors.append(f"Required package '{package_name}' is not installed. Try: pip install {package_name}")
    
    # Check optional packages
    optional_packages = {
        'underthesea': 'Vietnamese tokenization',
        'pyvi': 'Alternative Vietnamese tokenization'
    }
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            logger.debug(f"✓ {package} is available ({description})")
        except ImportError:
            warnings.append(f"Optional package '{package}' not available: {description}")
    
    # Report results
    if errors:
        logger.error("Environment validation failed:")
        for error in errors:
            logger.error(f"  ❌ {error}")
        return False
    
    if warnings:
        logger.warning("Environment validation completed with warnings:")
        for warning in warnings:
            logger.warning(f"  ⚠️  {warning}")
    else:
        logger.info("✓ Environment validation passed")
    
    return True

def check_data_directories():
    """Check and create necessary data directories"""
    logger.info("Checking data directories...")
    
    directories = [
        'data',
        'data/test_dataset',
        'data/results',
        'logs',
        'uploads',
        'temp'
    ]
    
    for directory in directories:
        dir_path = os.path.join(os.path.dirname(current_dir), directory)
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")
            else:
                logger.debug(f"Directory exists: {dir_path}")
        except Exception as e:
            logger.error(f"Could not create directory {dir_path}: {e}")
            return False
    
    logger.info("✓ Data directories verified")
    return True

def load_configuration(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate configuration"""
    logger.info("Loading configuration...")
    
    try:
        # Get default configuration
        config = get_project_config()
        
        # Load from file if specified
        if config_file and os.path.exists(config_file):
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Merge configurations
            def merge_config(base, override):
                for key, value in override.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        merge_config(base[key], value)
                    else:
                        base[key] = value
            
            merge_config(config, file_config)
            logger.info(f"Configuration loaded from file: {config_file}")
        
        logger.info("✓ Configuration loaded successfully")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def initialize_components(config: Dict[str, Any]):
    """Initialize all backend components"""
    global project_manager, app_instance
    
    logger.info("Initializing backend components...")
    
    try:
        # Initialize project manager
        project_manager = ProjectManager(config)
        logger.info("✓ Project manager initialized")
        
        # Initialize API
        app_instance = TextSimilarityAPI(config)
        logger.info("✓ API server initialized")
        
        # Test components
        test_components()
        
        logger.info("✓ All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

def test_components():
    """Test that all components are working"""
    logger.info("Testing components...")
    
    try:
        # Test tokenizer
        from preprocessing.tokenizer import VietnameseTokenizer
        tokenizer = VietnameseTokenizer()
        test_tokens = tokenizer.tokenize("Đây là một câu tiếng Việt để test.")
        logger.debug(f"Tokenizer test result: {len(test_tokens)} tokens")
        
        # Test similarity calculator
        from similarity_measures import SimilarityCalculator
        calculator = SimilarityCalculator()
        test_score = calculator.calculate("Xin chào", "Chào bạn", "jaccard")
        logger.debug(f"Similarity test result: {test_score}")
        
        # Test file handler
        file_handler = FileHandler()
        logger.debug("File handler initialized")
        
        logger.info("✓ Component testing completed")
        
    except Exception as e:
        logger.error(f"Component testing failed: {e}")
        raise

def run_server(config: Dict[str, Any], host: Optional[str] = None, 
               port: Optional[int] = None, debug: Optional[bool] = None):
    """Run the API server"""
    global app_instance
    
    if not app_instance:
        raise RuntimeError("API instance not initialized")
    
    logger.info("Starting Vietnamese Text Similarity API Server...")
    
    # Print startup information
    api_config = config.get('api', {})
    final_host = host or api_config.get('host', '0.0.0.0')  
    final_port = port or api_config.get('port', 5000)
    final_debug = debug if debug is not None else api_config.get('debug', True)
    
    print("\n" + "="*60)
    print("VIETNAMESE TEXT SIMILARITY API")
    print("="*60)
    print(f"Server: http://localhost:{final_port}")  # HIỂN THị localhost CHO USER
    print(f"Debug Mode: {'Enabled' if final_debug else 'Disabled'}")
    print(f"Health Check: http://localhost:{final_port}/api/health")
    print(f"API Methods: http://localhost:{final_port}/api/methods")
    print("="*60)
    print("Available Endpoints:")
    print("   GET  /                   - API Information") 
    print("   GET  /api/dataset        - Get test datasets") 
    print("   POST /api/similarity     - Calculate text similarity")
    print("   POST /api/tokenize       - Tokenize Vietnamese text")
    print("   POST /api/batch_similarity - Batch similarity calculation")
    print("   POST /api/compare_methods  - Compare all methods")
    print("   GET  /api/health         - Health check")
    print("   GET  /api/methods        - Available methods")
    print("   GET  /api/config         - API configuration")
    print("   GET  /api/stats          - Usage statistics")
    print("="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        app_instance.run(host=final_host, port=final_port, debug=final_debug)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

def create_sample_config():
    """Create a sample configuration file"""
    config = get_project_config()
    
    config_file = os.path.join(os.path.dirname(current_dir), 'config.json')
    
    try:
        import json
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Sample configuration created: {config_file}")
        print("You can modify this file and use it with --config option")
        
    except Exception as e:
        logger.error(f"Error creating sample config: {e}")

def main():
    """Main function"""
    global logger
    
    parser = argparse.ArgumentParser(
        description='Vietnamese Text Similarity Backend Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Start with default settings
  python main.py --host 0.0.0.0 --port 8080  # Start on all interfaces, port 8080
  python main.py --debug                  # Start in debug mode
  python main.py --config config.json    # Use custom configuration
  python main.py --create-config         # Create sample configuration file
        """
    )
    
    parser.add_argument('--host', type=str, help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create sample configuration file and exit')
    parser.add_argument('--validate-only', action='store_true', 
                       help='Only validate environment and configuration, then exit')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_sample_config()
        return
    
    # Setup initial logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Vietnamese Text Similarity Backend...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    try:
        # Setup signal handlers
        setup_signal_handlers()
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Check data directories
        if not check_data_directories():
            logger.error("Data directory setup failed")
            sys.exit(1)
        
        # Load configuration
        config = load_configuration(args.config)
        
        # Handle validation-only mode
        if args.validate_only:
            logger.info("Validation completed successfully")
            return
        
        # Initialize components
        initialize_components(config)
        
        # Determine debug mode
        debug_mode = None
        if args.debug:
            debug_mode = True
        elif args.no_debug:
            debug_mode = False
        
        # Run server
        run_server(config, args.host, args.port, debug_mode)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        cleanup_and_exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        cleanup_and_exit(1)

app = Flask(__name__)
CORS(app)

@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
       
        df = pd.read_csv('../data/test_dataset/dataset.csv')
        categories = df['category'].unique().tolist()
        print(f"Found categories: {categories}")  # Thêm log để debug
        return jsonify({
            'status': 'success',
            'data': categories
        })
    except Exception as e:
        print(f"Error loading categories: {e}")  # Thêm log để debug
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/labels', methods=['GET'])
def get_labels():
    try:
        # Sử dụng đường dẫn tương đối từ thư mục backend
        df = pd.read_csv('../data/test_dataset/dataset.csv')
        labels = df['label'].unique().tolist()
        print(f"Found labels: {labels}")  # Thêm log để debug
        return jsonify({
            'status': 'success',
            'data': labels
        })
    except Exception as e:
        print(f"Error loading labels: {e}")  # Thêm log để debug
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    try:
        category = request.args.get('category')
        label = request.args.get('label')
        
        df = pd.read_csv('../data/test_dataset/dataset.csv')
        
        if category:
            df = df[df['category'] == category]
        if label:
            df = df[df['label'] == label]
        # Convert DataFrame to list of dictionaries using to_dict('records')
        result = df.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    main()