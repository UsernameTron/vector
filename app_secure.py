"""
Secure Vector RAG Database Application
Main application entry point with comprehensive security features
"""

from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Import security modules
from security import (
    validate_input, 
    require_auth, 
    JWTAuth,
    encrypt_api_key,
    decrypt_api_key,
    validate_file_upload
)
from security.cors_config import SecureCORS
from config.security import SecurityConfig

# Import application modules
from vector_db import VectorDatabase
from agents import (
    ResearchAgent, 
    CEOAgent, 
    PerformanceAgent, 
    CoachingAgent,
    BusinessIntelligenceAgent,
    ContactCenterDirectorAgent
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app with security configuration
app = Flask(__name__)

# Apply security configuration
security_config = SecurityConfig.get_flask_config()
app.config.update(security_config)

# Initialize secure CORS
cors = SecureCORS.init_cors(app)

# Initialize JWT authentication
jwt_auth = JWTAuth(app)

# Initialize Vector Database
vector_db = VectorDatabase()

# Initialize AI Agents with encrypted API keys
def initialize_agents():
    """Initialize AI agents with proper error handling"""
    try:
        # Get encrypted OpenAI API key from environment
        encrypted_api_key = os.getenv('OPENAI_API_KEY_ENCRYPTED')
        if encrypted_api_key:
            # Decrypt API key for use
            api_key = decrypt_api_key(encrypted_api_key)
            os.environ['OPENAI_API_KEY'] = api_key
        
        agents = {
            'research': ResearchAgent(vector_db),
            'ceo': CEOAgent(vector_db),
            'performance': PerformanceAgent(vector_db),
            'coaching': CoachingAgent(vector_db),
            'business_intelligence': BusinessIntelligenceAgent(vector_db),
            'contact_center': ContactCenterDirectorAgent(vector_db)
        }
        
        logger.info(f"Initialized {len(agents)} AI agents")
        return agents
    
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        return {}

agents = initialize_agents()

# Security headers middleware
@app.after_request
def set_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    return response

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

@app.route('/api/auth/login', methods=['POST'])
@validate_input(required_json_fields=['username', 'password'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Simple demo authentication (replace with proper user management)
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

@app.route('/api/agents', methods=['GET'])
@require_auth('read')
def get_agents():
    """Get list of available agents (requires authentication)"""
    agent_info = {}
    for name, agent in agents.items():
        agent_info[name] = {
            'name': agent.name,
            'role': agent.role,
            'description': agent.description,
            'capabilities': agent.capabilities
        }
    return jsonify(agent_info)

@app.route('/api/chat/<agent_name>', methods=['POST'])
@require_auth('read')
@validate_input(required_json_fields=['query'])
def chat_with_agent(agent_name):
    """Chat with a specific agent (requires authentication)"""
    try:
        if agent_name not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        agent = agents[agent_name]
        response = agent.process_query(query)
        
        logger.info(f"User {request.current_user['user_id']} chatted with {agent_name}")
        
        return jsonify({
            'agent': agent_name,
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'user': request.current_user['user_id']
        })
    
    except Exception as e:
        logger.error(f"Error in chat with {agent_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['POST'])
@require_auth('write')
@validate_input(required_json_fields=['content'])
def upload_document():
    """Upload and process a document (requires write permission)"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        title = data.get('title', 'Untitled Document')
        source = data.get('source', 'user_upload')
        
        if not content:
            return jsonify({'error': 'Content is required'}), 400
        
        doc_id = vector_db.add_document(content, title, source)
        
        logger.info(f"User {request.current_user['user_id']} uploaded document: {title}")
        
        return jsonify({
            'document_id': doc_id,
            'title': title,
            'status': 'processed',
            'timestamp': datetime.now().isoformat(),
            'uploaded_by': request.current_user['user_id']
        })
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/upload', methods=['POST'])
@require_auth('write')
@validate_file_upload(max_size=10*1024*1024)  # 10MB limit
def upload_document_file():
    """Upload a document file (requires write permission)"""
    try:
        file = request.files['file']
        validation_result = request.file_validation
        
        # Read file content
        file.seek(0)
        content = file.read().decode('utf-8', errors='ignore')
        
        # Process document
        title = validation_result['file_info']['secure_filename']
        doc_id = vector_db.add_document(content, title, 'file_upload')
        
        logger.info(f"User {request.current_user['user_id']} uploaded file: {title}")
        
        return jsonify({
            'document_id': doc_id,
            'title': title,
            'status': 'processed',
            'file_info': validation_result['file_info'],
            'timestamp': datetime.now().isoformat(),
            'uploaded_by': request.current_user['user_id']
        })
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
@require_auth('read')
def get_documents():
    """Get list of stored documents (requires authentication)"""
    try:
        documents = vector_db.get_all_documents()
        return jsonify(documents)
    
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
@require_auth('read')
@validate_input(required_json_fields=['query'])
def search_documents():
    """Search documents in vector database (requires authentication)"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        limit = data.get('limit', 5)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = vector_db.search(query, limit)
        
        logger.info(f"User {request.current_user['user_id']} searched: {query}")
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat(),
            'searched_by': request.current_user['user_id']
        })
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users', methods=['GET'])
@require_auth(admin_only=True)
def get_users():
    """Get users (admin only)"""
    # Demo endpoint - replace with proper user management
    users = [
        {'id': 'admin', 'role': 'admin', 'active': True},
        {'id': 'user', 'role': 'user', 'active': True},
        {'id': 'readonly', 'role': 'readonly', 'active': True}
    ]
    return jsonify(users)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint (public)"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database_status': vector_db.get_status(),
        'agents_count': len(agents),
        'security_features': [
            'Input validation',
            'JWT authentication',
            'CORS protection',
            'File upload validation',
            'API key encryption'
        ]
    })

@app.route('/api/security/config', methods=['GET'])
@require_auth(admin_only=True)
def get_security_config():
    """Get security configuration status (admin only)"""
    config_status = SecurityConfig.validate_security_config()
    return jsonify({
        'security_checks': config_status,
        'recommendations': [
            'Ensure all secret keys are properly configured',
            'Use HTTPS in production',
            'Configure proper CORS origins',
            'Set up rate limiting',
            'Enable request logging'
        ]
    })

if __name__ == '__main__':
    # Validate security configuration on startup
    config_status = SecurityConfig.validate_security_config()
    if not all(config_status.values()):
        logger.warning("Security configuration incomplete!")
        logger.info("Run: python -c \"from config.security import SecurityConfig; SecurityConfig.save_development_env()\" to generate secure defaults")
    
    logger.info("Starting Secure Vector RAG Database Application")
    logger.info(f"Security features enabled: JWT auth, input validation, CORS, file validation")
    
    # Use secure defaults
    host = '127.0.0.1' if os.getenv('FLASK_ENV') == 'production' else '0.0.0.0'
    debug = os.getenv('FLASK_ENV') != 'production'
    
    app.run(host=host, port=8000, debug=debug)