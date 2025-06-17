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

@app.route('/api/chat/<agent_name>', methods=['POST'])
def chat_with_agent(agent_name):
    """Chat with a specific agent"""
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

if __name__ == '__main__':
    logger.info("Starting Vector RAG Database Application")
    app.run(host='0.0.0.0', port=5000, debug=True)
