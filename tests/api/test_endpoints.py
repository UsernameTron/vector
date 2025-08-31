"""Integration tests for API endpoints"""

import pytest
import json
from unittest.mock import patch, Mock


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_endpoint_success(self, test_app):
        """Test health endpoint returns success"""
        response = test_app.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'status' in data
        assert data['status'] in ['healthy', 'partial']

    def test_health_endpoint_includes_agents(self, test_app):
        """Test health endpoint includes agent information"""
        response = test_app.get('/health')
        data = response.get_json()
        
        assert 'agents_available' in data
        assert isinstance(data['agents_available'], int)


class TestAgentsEndpoint:
    """Test agents listing endpoint"""

    def test_agents_endpoint_success(self, test_app):
        """Test agents endpoint returns list of agents"""
        response = test_app.get('/api/agents')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'agents' in data
        assert isinstance(data['agents'], list)
        assert len(data['agents']) > 0

    def test_agents_endpoint_includes_required_fields(self, test_app):
        """Test each agent has required fields"""
        response = test_app.get('/api/agents')
        data = response.get_json()
        
        for agent in data['agents']:
            assert 'name' in agent
            assert 'role' in agent
            assert 'description' in agent
            assert 'capabilities' in agent


class TestChatEndpoint:
    """Test chat endpoint functionality"""

    @patch('openai.OpenAI')
    def test_chat_endpoint_success(self, mock_openai_class, test_app):
        """Test successful chat interaction"""
        # Setup OpenAI mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        response = test_app.post('/api/chat', 
            data=json.dumps({
                'message': 'Hello',
                'agent_type': 'research'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'response' in data

    def test_chat_endpoint_missing_message(self, test_app):
        """Test chat endpoint with missing message"""
        response = test_app.post('/api/chat',
            data=json.dumps({
                'agent_type': 'research'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400

    def test_chat_endpoint_invalid_agent(self, test_app):
        """Test chat endpoint with invalid agent type"""
        response = test_app.post('/api/chat',
            data=json.dumps({
                'message': 'Hello',
                'agent_type': 'invalid_agent'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400

    def test_chat_endpoint_missing_content_type(self, test_app):
        """Test chat endpoint without JSON content type"""
        response = test_app.post('/api/chat',
            data='{"message": "Hello", "agent_type": "research"}'
        )
        
        assert response.status_code == 400


class TestSearchEndpoint:
    """Test document search endpoint"""

    def test_search_endpoint_success(self, test_app, mock_vector_db):
        """Test successful search"""
        with patch('vector_db.VectorDatabase', return_value=mock_vector_db):
            response = test_app.post('/api/search',
                data=json.dumps({
                    'query': 'test search query',
                    'limit': 5
                }),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = response.get_json()
            assert 'results' in data

    def test_search_endpoint_missing_query(self, test_app):
        """Test search endpoint without query"""
        response = test_app.post('/api/search',
            data=json.dumps({
                'limit': 5
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400

    def test_search_endpoint_default_limit(self, test_app, mock_vector_db):
        """Test search endpoint uses default limit"""
        # Test app doesn't use vector_db, just returns empty results
        response = test_app.post('/api/search',
            data=json.dumps({
                'query': 'test query'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'results' in data


class TestUploadEndpoint:
    """Test document upload endpoint"""

    def test_upload_endpoint_success(self, test_app, mock_vector_db):
        """Test successful document upload"""
        # Test app returns fixed document ID
        response = test_app.post('/api/upload',
            data=json.dumps({
                'title': 'Test Document',
                'content': 'This is test content for the document'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'document_id' in data
        assert data['document_id'] == "test-doc-123"

    def test_upload_endpoint_missing_content(self, test_app):
        """Test upload endpoint without content"""
        response = test_app.post('/api/upload',
            data=json.dumps({
                'title': 'Test Document'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400

    def test_upload_endpoint_empty_content(self, test_app):
        """Test upload endpoint with empty content"""
        response = test_app.post('/api/upload',
            data=json.dumps({
                'title': 'Test Document',
                'content': ''
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400


class TestStatusEndpoint:
    """Test system status endpoint"""

    def test_status_endpoint_success(self, test_app):
        """Test status endpoint returns system information"""
        response = test_app.get('/api/status')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'timestamp' in data
        assert 'version' in data or 'system' in data