"""
Vector RAG Database Application
Main application entry point with Flask backend
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Import our custom modules
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup middleware
try:
    from middleware.error_handlers import register_error_handlers
    from middleware.validation import setup_validation_middleware
    
    register_error_handlers(app)
    setup_validation_middleware(app)
    logger.info("✅ Middleware registered successfully")
except ImportError as e:
    logger.warning(f"⚠️ Middleware not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to setup middleware: {e}")

# Initialize Vector Database
vector_db = VectorDatabase()

# Initialize AI Agents
agents = {
    'research': ResearchAgent(vector_db),
    'ceo': CEOAgent(vector_db),
    'performance': PerformanceAgent(vector_db),
    'coaching': CoachingAgent(vector_db),
    'business_intelligence': BusinessIntelligenceAgent(vector_db),
    'contact_center': ContactCenterDirectorAgent(vector_db)
}

@app.route('/favicon.ico')
def favicon():
    """Return a simple favicon to prevent 404 errors"""
    from flask import Response
    # Return a minimal 1x1 transparent PNG
    transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    return Response(transparent_png, mimetype='image/png')

@app.route('/')
def index():
    """Main dashboard route"""
    return render_template('index.html')

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get list of available agents"""
    agent_info = {}
    for name, agent in agents.items():
        agent_info[name] = {
            'name': agent.name,
            'role': agent.role,
            'description': agent.description,
            'capabilities': agent.capabilities
        }
    return jsonify(agent_info)

@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    """Chat with a specific agent"""
    try:
        data = request.get_json()
        agent_name = data.get('agent', '')
        message = data.get('message', '')
        
        if not agent_name:
            return jsonify({'error': 'Agent name is required'}), 400
            
        if agent_name not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        agent = agents[agent_name]
        response = agent.process_query(message)
        
        return jsonify({
            'agent': agent_name,
            'query': message,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/<agent_name>', methods=['POST'])
def chat_with_agent_legacy(agent_name):
    """Legacy chat endpoint for backwards compatibility"""
    try:
        if agent_name not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        agent = agents[agent_name]
        response = agent.process_query(query)
        
        return jsonify({
            'agent': agent_name,
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in chat with {agent_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        title = data.get('title', 'Untitled Document')
        source = data.get('source', 'user_upload')
        
        if not content:
            return jsonify({'error': 'Content is required'}), 400
        
        doc_id = vector_db.add_document(content, title, source)
        
        return jsonify({
            'document_id': doc_id,
            'title': title,
            'status': 'processed',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of stored documents"""
    try:
        documents = vector_db.get_all_documents()
        return jsonify(documents)
    
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Search documents in vector database"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        limit = data.get('limit', 5)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = vector_db.search(query, limit)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database_status': vector_db.get_status(),
        'agents_count': len(agents)
    })

@app.route('/health', methods=['GET'])
def health():
    """Alternative health check endpoint"""
    try:
        db_status = vector_db.get_status()
        db_healthy = db_status.get('status') == 'connected' if isinstance(db_status, dict) else False
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'agents_available': len(agents),
            'vector_db_available': db_healthy,
            'database_status': db_status,
            'agents_count': len(agents)
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'agents_available': len(agents),
            'vector_db_available': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get application status"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'database_status': vector_db.get_status(),
        'agents_count': len(agents)
    })

if __name__ == '__main__':
    logger.info("Starting Vector RAG Database Application")
    app.run(host='0.0.0.0', port=5001, debug=True)
