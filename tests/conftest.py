"""Pytest configuration and fixtures"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    with patch('openai.OpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock typical responses
        mock_instance.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )
        mock_instance.models.list.return_value = Mock()
        
        yield mock_instance


@pytest.fixture
def mock_vector_db():
    """Mock vector database for testing"""
    mock_db = Mock()
    mock_db.add_document.return_value = "test-doc-id"
    mock_db.search.return_value = [
        {
            "content": "Test document content",
            "metadata": {"title": "Test Document", "source": "test"}
        }
    ]
    mock_db.get_document_count.return_value = 5
    return mock_db


@pytest.fixture
def test_app():
    """Create a test Flask application"""
    from flask import Flask
    from flask_cors import CORS
    import sys
    import os
    
    # Create a simple test app
    app = Flask(__name__)
    CORS(app)
    
    app.config.update({
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key',
        'OPENAI_API_KEY': 'sk-test-key',
        'CHROMA_PERSIST_DIRECTORY': ':memory:',
    })
    
    # Add basic routes for testing
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'agents_available': 6}
    
    @app.route('/api/agents')
    def agents():
        return {
            'agents': [
                {'name': 'Research Agent', 'role': 'research', 'description': 'Test agent', 'capabilities': ['research']},
                {'name': 'CEO Agent', 'role': 'ceo', 'description': 'Test agent', 'capabilities': ['strategy']},
            ]
        }
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        from flask import request, jsonify
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data.get('message'):
            return jsonify({'error': 'Message is required'}), 400
        
        valid_agents = ['research', 'ceo', 'performance', 'coaching', 'business_intelligence', 'contact_center']
        if data.get('agent_type') not in valid_agents:
            return jsonify({'error': 'Invalid agent type'}), 400
            
        return {'response': 'Test response'}
    
    @app.route('/api/search', methods=['POST'])
    def search():
        from flask import request, jsonify
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data.get('query'):
            return jsonify({'error': 'Query is required'}), 400
            
        return {'results': []}
    
    @app.route('/api/upload', methods=['POST']) 
    def upload():
        from flask import request, jsonify
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data.get('content'):
            return jsonify({'error': 'Content is required'}), 400
        if not data.get('content').strip():
            return jsonify({'error': 'Content cannot be empty'}), 400
            
        return {'document_id': 'test-doc-123'}
    
    @app.route('/api/status')
    def status():
        return {'timestamp': '2024-01-01T00:00:00Z', 'version': '1.0.0'}
    
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ.update({
        'FLASK_ENV': 'testing',
        'OPENAI_API_KEY': 'sk-test-key-for-testing',
        'ENCRYPTION_SECRET': 'test-encryption-secret',
        'ENCRYPTION_SALT': 'dGVzdC1zYWx0LTEyMzQ=',  # base64 encoded 'test-salt-1234'
        'CHROMA_PERSIST_DIRECTORY': ':memory:',
    })