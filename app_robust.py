"""
Robust Vector RAG Database Application
Main application with comprehensive error handling, recovery mechanisms, and enhanced security
"""

from flask import Flask, request, jsonify, render_template
import os
import sys
import logging
import atexit
from datetime import datetime
from typing import Dict, Any

# Enhanced error handling and recovery
from utils.error_handler import (
    ErrorCategories,
    ErrorRecovery,
    DependencyManager,
    FileCleanupManager,
    handle_errors,
    safe_operation
)

# Import security modules with fallback handling
try:
    from security import (
        validate_input, 
        require_auth, 
        JWTAuth,
        encrypt_api_key,
        decrypt_api_key
    )
    from security.file_validator_enhanced import enhanced_validate_file_upload
    from security.cors_config import SecureCORS
    from config.security import SecurityConfig
    SECURITY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Security modules not available: {e}")
    SECURITY_AVAILABLE = False

# Import application modules with fallback
try:
    from vector_db_robust import VectorDatabase
    VECTOR_DB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Robust vector DB not available, using fallback: {e}")
    try:
        from vector_db import VectorDatabase
        VECTOR_DB_AVAILABLE = True
    except ImportError as e2:
        logging.error(f"No vector database available: {e2}")
        VECTOR_DB_AVAILABLE = False

# AI Agents with fallback handling
AGENTS_AVAILABLE = False
agents = {}
try:
    from agents import (
        ResearchAgent, 
        CEOAgent, 
        PerformanceAgent, 
        CoachingAgent,
        BusinessIntelligenceAgent,
        ContactCenterDirectorAgent
    )
    AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AI agents not available: {e}")

# Load environment variables safely
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not available, using system environment only")

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a') if os.access('.', os.W_OK) else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Apply security configuration if available
if SECURITY_AVAILABLE:
    try:
        security_config = SecurityConfig.get_flask_config()
        app.config.update(security_config)
        cors = SecureCORS.init_cors(app)
        jwt_auth = JWTAuth(app)
        logger.info("Security features enabled")
    except Exception as e:
        logger.error(f"Security configuration failed: {e}")
        SECURITY_AVAILABLE = False
else:
    # Minimal CORS for development
    try:
        from flask_cors import CORS
        CORS(app)
        logger.info("Basic CORS enabled")
    except ImportError:
        logger.warning("CORS not available")

# Global application state
app_state = {
    'vector_db': None,
    'agents': {},
    'initialization_time': datetime.now(),
    'error_count': 0,
    'last_error': None
}

def initialize_vector_database():
    """Initialize vector database with comprehensive error handling"""
    if not VECTOR_DB_AVAILABLE:
        logger.error("Vector database module not available")
        return None
    
    try:
        logger.info("Initializing vector database...")
        vector_db = VectorDatabase()
        
        # Test database connectivity
        status = vector_db.get_status()
        if status.get('status') == 'connected' or status.get('initialized'):
            logger.info("Vector database initialized successfully")
            return vector_db
        else:
            logger.error(f"Vector database initialization failed: {status}")
            return None
            
    except Exception as e:
        logger.error(f"Vector database initialization error: {e}")
        return None

def initialize_agents(vector_db):
    """Initialize AI agents with error handling"""
    if not AGENTS_AVAILABLE or not vector_db:
        logger.warning("AI agents or vector database not available")
        return {}
    
    agents_dict = {}
    
    try:
        # Handle encrypted API keys
        encrypted_api_key = os.getenv('OPENAI_API_KEY_ENCRYPTED')
        if encrypted_api_key and SECURITY_AVAILABLE:
            try:
                api_key = decrypt_api_key(encrypted_api_key)
                os.environ['OPENAI_API_KEY'] = api_key
                logger.info("API key decrypted successfully")
            except Exception as e:
                logger.error(f"API key decryption failed: {e}")
        
        # Initialize each agent with individual error handling
        agent_classes = {
            'research': ResearchAgent,
            'ceo': CEOAgent,
            'performance': PerformanceAgent,
            'coaching': CoachingAgent,
            'business_intelligence': BusinessIntelligenceAgent,
            'contact_center': ContactCenterDirectorAgent
        }
        
        for name, agent_class in agent_classes.items():
            try:
                agents_dict[name] = agent_class(vector_db)
                logger.info(f"Initialized {name} agent")
            except Exception as e:
                logger.error(f"Failed to initialize {name} agent: {e}")
                continue
        
        logger.info(f"Initialized {len(agents_dict)} AI agents")
        
    except Exception as e:
        logger.error(f"Agent initialization error: {e}")
    
    return agents_dict

# Initialize application components
@safe_operation("application_initialization")
def initialize_application():
    """Initialize all application components"""
    global app_state
    
    logger.info("Starting application initialization...")
    
    # Initialize vector database
    app_state['vector_db'] = initialize_vector_database()
    
    # Initialize agents
    app_state['agents'] = initialize_agents(app_state['vector_db'])
    
    # Check dependency status
    dependencies = {
        'chromadb': DependencyManager.check_dependency('chromadb'),
        'openai': DependencyManager.check_dependency('openai'),
        'magic': DependencyManager.check_dependency('magic', fallback_available=True),
        'bleach': DependencyManager.check_dependency('bleach', fallback_available=True),
        'cryptography': DependencyManager.check_dependency('cryptography', fallback_available=True),
    }
    
    logger.info(f"Dependency status: {dependencies}")
    
    # Set up cleanup handler
    atexit.register(cleanup_on_exit)
    
    logger.info("Application initialization completed")

def cleanup_on_exit():
    """Cleanup function called on application exit"""
    logger.info("Performing application cleanup...")
    FileCleanupManager.cleanup_all()
    logger.info("Cleanup completed")

# Security headers middleware
@app.after_request
def set_security_headers(response):
    """Add security headers to all responses"""
    try:
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        if os.getenv('FLASK_ENV') == 'production':
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    except Exception as e:
        logger.warning(f"Failed to set security headers: {e}")
    
    return response

# Error handler for uncaught exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    global app_state
    app_state['error_count'] += 1
    app_state['last_error'] = str(e)
    
    ErrorRecovery.log_error(e, "global_exception_handler")
    
    error_response, status = ErrorRecovery.create_error_response(
        ErrorCategories.SYSTEM,
        'unexpected_error',
        str(e),
        500,
        ['Try refreshing the page', 'Contact support if the problem persists']
    )
    
    return jsonify(error_response), status

@app.route('/favicon.ico')
def favicon():
    """Return a simple favicon to prevent 404 errors"""
    from flask import Response
    transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(transparent_png, mimetype='image/png')

@app.route('/')
def index():
    """Main dashboard route"""
    return render_template('index.html')

# Authentication endpoints (if security is available)
if SECURITY_AVAILABLE:
    @app.route('/api/auth/login', methods=['POST'])
    @validate_input(required_json_fields=['username', 'password'])
    def login():
        """User login endpoint"""
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # Simple demo authentication
            demo_users = {
                'admin': {'password': 'admin123', 'role': 'admin'},
                'user': {'password': 'user123', 'role': 'user'},
                'readonly': {'password': 'readonly123', 'role': 'readonly'}
            }
            
            if username in demo_users and demo_users[username]['password'] == password:
                role = demo_users[username]['role']
                token = jwt_auth.generate_token(username, role)
                
                logger.info(f"User {username} logged in successfully")
                return jsonify({
                    'token': token,
                    'user': username,
                    'role': role,
                    'expires_in': app.config['JWT_EXPIRATION_HOURS'] * 3600
                })
            else:
                logger.warning(f"Failed login attempt for user: {username}")
                return jsonify({'error': 'Invalid credentials'}), 401
        
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({'error': 'Login failed'}), 500

def require_auth_if_available(permission: str = None):
    """Conditional authentication decorator"""
    def decorator(f):
        if SECURITY_AVAILABLE:
            return require_auth(permission)(f)
        else:
            return f
    return decorator

@app.route('/api/agents', methods=['GET'])
@require_auth_if_available('read')
@handle_errors(ErrorCategories.SYSTEM)
def get_agents():
    """Get list of available agents"""
    if not app_state['agents']:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DEPENDENCY,
            'ai_service_down',
            'AI agents are not available',
            503,
            ['Check if the application is properly configured', 'Contact support for assistance']
        )
        return jsonify(error_response), status
    
    agent_info = {}
    for name, agent in app_state['agents'].items():
        try:
            agent_info[name] = {
                'name': getattr(agent, 'name', name),
                'role': getattr(agent, 'role', 'Unknown'),
                'description': getattr(agent, 'description', 'No description available'),
                'capabilities': getattr(agent, 'capabilities', [])
            }
        except Exception as e:
            logger.warning(f"Failed to get info for agent {name}: {e}")
            agent_info[name] = {
                'name': name,
                'role': 'Unknown',
                'description': 'Agent information unavailable',
                'capabilities': []
            }
    
    return jsonify(agent_info)

@app.route('/api/chat/<agent_name>', methods=['POST'])
@require_auth_if_available('read')
@handle_errors(ErrorCategories.SYSTEM)
def chat_with_agent(agent_name):
    """Chat with a specific agent"""
    if SECURITY_AVAILABLE:
        @validate_input(required_json_fields=['query'])
        def _chat_with_validation():
            return _perform_chat(agent_name)
        return _chat_with_validation()
    else:
        return _perform_chat(agent_name)

def _perform_chat(agent_name):
    """Core chat functionality"""
    if not app_state['agents']:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DEPENDENCY,
            'ai_service_down',
            'AI agents are not available',
            503,
            ['Check the application configuration', 'Try again later']
        )
        return jsonify(error_response), status
    
    if agent_name not in app_state['agents']:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.USER_INPUT,
            'invalid_agent',
            f'Agent "{agent_name}" not found',
            404,
            [f'Available agents: {", ".join(app_state["agents"].keys())}']
        )
        return jsonify(error_response), status
    
    data = request.get_json()
    query = data.get('query', '') if data else ''
    
    if not query:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.USER_INPUT,
            'empty_query',
            'Query is required',
            400,
            ['Please provide a question or message for the agent']
        )
        return jsonify(error_response), status
    
    try:
        agent = app_state['agents'][agent_name]
        response = agent.process_query(query)
        
        user_id = getattr(request, 'current_user', {}).get('user_id', 'anonymous')
        logger.info(f"User {user_id} chatted with {agent_name}")
        
        return jsonify({
            'agent': agent_name,
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'user': user_id
        })
    
    except Exception as e:
        logger.error(f"Chat error with {agent_name}: {e}")
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DEPENDENCY,
            'ai_processing_error',
            str(e),
            500,
            ['Try rephrasing your question', 'Try again in a few moments']
        )
        return jsonify(error_response), status

@app.route('/api/documents', methods=['POST'])
@require_auth_if_available('write')
@handle_errors(ErrorCategories.DATABASE)
def upload_document():
    """Upload and process a document"""
    if not app_state['vector_db']:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DATABASE,
            'database_unavailable',
            'Database is not available',
            503,
            ['Check database configuration', 'Contact support']
        )
        return jsonify(error_response), status
    
    if SECURITY_AVAILABLE:
        @validate_input(required_json_fields=['content'])
        def _upload_with_validation():
            return _perform_upload()
        return _upload_with_validation()
    else:
        return _perform_upload()

def _perform_upload():
    """Core document upload functionality"""
    data = request.get_json()
    content = data.get('content', '') if data else ''
    title = data.get('title', 'Untitled Document') if data else 'Untitled Document'
    source = data.get('source', 'user_upload') if data else 'user_upload'
    
    if not content:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.USER_INPUT,
            'empty_content',
            'Document content is required',
            400,
            ['Please provide document content']
        )
        return jsonify(error_response), status
    
    try:
        doc_id = app_state['vector_db'].add_document(content, title, source)
        
        user_id = getattr(request, 'current_user', {}).get('user_id', 'anonymous')
        logger.info(f"User {user_id} uploaded document: {title}")
        
        return jsonify({
            'document_id': doc_id,
            'title': title,
            'status': 'processed',
            'timestamp': datetime.now().isoformat(),
            'uploaded_by': user_id
        })
    
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DATABASE,
            'storage_failed',
            str(e),
            500,
            ['Try uploading again', 'Check if the document is valid']
        )
        return jsonify(error_response), status

@app.route('/api/documents/upload', methods=['POST'])
@require_auth_if_available('write')
def upload_document_file():
    """Upload a document file"""
    if not app_state['vector_db']:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DATABASE,
            'database_unavailable',
            'Database is not available',
            503,
            ['Check database configuration', 'Contact support']
        )
        return jsonify(error_response), status
    
    if SECURITY_AVAILABLE:
        @enhanced_validate_file_upload(max_size=10*1024*1024)
        def _upload_file_with_validation():
            return _perform_file_upload()
        return _upload_file_with_validation()
    else:
        return _perform_file_upload()

def _perform_file_upload():
    """Core file upload functionality"""
    try:
        if 'file' not in request.files:
            error_response, status = ErrorRecovery.create_error_response(
                ErrorCategories.USER_INPUT,
                'no_file',
                'No file provided',
                400,
                ['Please select a file to upload']
            )
            return jsonify(error_response), status
        
        file = request.files['file']
        if not file.filename:
            error_response, status = ErrorRecovery.create_error_response(
                ErrorCategories.USER_INPUT,
                'empty_filename',
                'No file selected',
                400,
                ['Please select a valid file']
            )
            return jsonify(error_response), status
        
        # Read file content
        file.seek(0)
        content = file.read().decode('utf-8', errors='ignore')
        
        # Get validation info if available
        validation_result = getattr(request, 'file_validation', {})
        title = validation_result.get('file_info', {}).get('secure_filename', file.filename)
        
        # Process document
        doc_id = app_state['vector_db'].add_document(content, title, 'file_upload')
        
        user_id = getattr(request, 'current_user', {}).get('user_id', 'anonymous')
        logger.info(f"User {user_id} uploaded file: {title}")
        
        response_data = {
            'document_id': doc_id,
            'title': title,
            'status': 'processed',
            'timestamp': datetime.now().isoformat(),
            'uploaded_by': user_id
        }
        
        # Add file info if validation was performed
        if validation_result:
            response_data['file_info'] = validation_result.get('file_info', {})
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"File upload error: {e}")
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.FILE_PROCESSING,
            'processing_failed',
            str(e),
            500,
            ['Try uploading the file again', 'Check if the file is valid and not corrupted']
        )
        return jsonify(error_response), status

@app.route('/api/documents', methods=['GET'])
@require_auth_if_available('read')
@handle_errors(ErrorCategories.DATABASE)
def get_documents():
    """Get list of stored documents"""
    if not app_state['vector_db']:
        return jsonify([])  # Return empty list if database not available
    
    try:
        documents = app_state['vector_db'].get_all_documents()
        return jsonify(documents)
    
    except Exception as e:
        logger.error(f"Get documents error: {e}")
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DATABASE,
            'retrieval_failed',
            str(e),
            500,
            ['Try refreshing the page', 'Check database connectivity']
        )
        return jsonify(error_response), status

@app.route('/api/search', methods=['POST'])
@require_auth_if_available('read')
@handle_errors(ErrorCategories.DATABASE)
def search_documents():
    """Search documents in vector database"""
    if not app_state['vector_db']:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DATABASE,
            'database_unavailable',
            'Search is not available',
            503,
            ['Check database configuration', 'Contact support']
        )
        return jsonify(error_response), status
    
    if SECURITY_AVAILABLE:
        @validate_input(required_json_fields=['query'])
        def _search_with_validation():
            return _perform_search()
        return _search_with_validation()
    else:
        return _perform_search()

def _perform_search():
    """Core search functionality"""
    data = request.get_json()
    query = data.get('query', '') if data else ''
    limit = data.get('limit', 5) if data else 5
    
    if not query:
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.USER_INPUT,
            'empty_query',
            'Search query is required',
            400,
            ['Please provide a search term']
        )
        return jsonify(error_response), status
    
    try:
        results = app_state['vector_db'].search(query, limit)
        
        user_id = getattr(request, 'current_user', {}).get('user_id', 'anonymous')
        logger.info(f"User {user_id} searched: {query}")
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat(),
            'searched_by': user_id
        })
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        error_response, status = ErrorRecovery.create_error_response(
            ErrorCategories.DATABASE,
            'search_failed',
            str(e),
            500,
            ['Try simplifying your search query', 'Try again in a few moments']
        )
        return jsonify(error_response), status

@app.route('/api/health', methods=['GET'])
@handle_errors(ErrorCategories.SYSTEM)
def health_check():
    """Comprehensive health check endpoint"""
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': str(datetime.now() - app_state['initialization_time']),
        'components': {}
    }
    
    # Check vector database
    if app_state['vector_db']:
        try:
            db_health = app_state['vector_db'].health_check()
            health_data['components']['database'] = db_health
        except Exception as e:
            health_data['components']['database'] = {
                'healthy': False,
                'error': str(e)
            }
    else:
        health_data['components']['database'] = {
            'healthy': False,
            'error': 'Database not initialized'
        }
    
    # Check agents
    health_data['components']['agents'] = {
        'healthy': len(app_state['agents']) > 0,
        'count': len(app_state['agents']),
        'available_agents': list(app_state['agents'].keys())
    }
    
    # Check security features
    health_data['components']['security'] = {
        'healthy': SECURITY_AVAILABLE,
        'features_enabled': [
            'Input validation',
            'JWT authentication',
            'CORS protection',
            'File upload validation',
            'API key encryption'
        ] if SECURITY_AVAILABLE else ['Basic CORS']
    }
    
    # Check dependencies
    health_data['components']['dependencies'] = DependencyManager.get_dependency_status()
    
    # Overall health
    component_health = [comp.get('healthy', False) for comp in health_data['components'].values()]
    health_data['status'] = 'healthy' if any(component_health) else 'unhealthy'
    health_data['error_count'] = app_state['error_count']
    
    return jsonify(health_data)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get detailed application status"""
    status_data = {
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'initialization_time': app_state['initialization_time'].isoformat(),
        'features': {
            'security': SECURITY_AVAILABLE,
            'vector_database': app_state['vector_db'] is not None,
            'ai_agents': len(app_state['agents']) > 0
        },
        'statistics': {
            'error_count': app_state['error_count'],
            'last_error': app_state['last_error'],
            'agents_available': len(app_state['agents'])
        }
    }
    
    if app_state['vector_db']:
        try:
            db_status = app_state['vector_db'].get_status()
            status_data['database_status'] = db_status
        except Exception as e:
            status_data['database_status'] = {'error': str(e)}
    
    return jsonify(status_data)

# Initialize application on startup
initialize_application()

if __name__ == '__main__':
    logger.info("Starting Robust Vector RAG Database Application")
    
    # Display feature status
    features = []
    if SECURITY_AVAILABLE:
        features.append("Security")
    if VECTOR_DB_AVAILABLE:
        features.append("Vector Database")
    if AGENTS_AVAILABLE:
        features.append("AI Agents")
    
    logger.info(f"Available features: {', '.join(features) if features else 'Basic mode'}")
    
    # Use secure defaults
    host = '127.0.0.1' if os.getenv('FLASK_ENV') == 'production' else '0.0.0.0'
    debug = os.getenv('FLASK_ENV') != 'production'
    
    try:
        app.run(host=host, port=8000, debug=debug)
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)
    finally:
        cleanup_on_exit()