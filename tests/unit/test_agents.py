"""Unit tests for AI agents"""

import pytest
from unittest.mock import Mock, patch
import os

from agents import BaseAgent, ResearchAgent, CEOAgent


class TestBaseAgent:
    """Test the base agent functionality"""

    def test_base_agent_initialization(self, mock_vector_db, mock_openai_client):
        """Test base agent can be initialized properly"""
        agent = BaseAgent(
            vector_db=mock_vector_db,
            name="Test Agent",
            role="tester",
            description="A test agent",
            capabilities=["testing", "mocking"]
        )
        
        assert agent.name == "Test Agent"
        assert agent.role == "tester"
        assert agent.description == "A test agent"
        assert "testing" in agent.capabilities
        assert agent.vector_db == mock_vector_db

    def test_get_context_success(self, mock_vector_db):
        """Test context retrieval from vector database"""
        # Setup mock response
        mock_vector_db.search.return_value = [
            {
                "content": "Test content 1",
                "metadata": {"title": "Doc 1", "source": "test"}
            },
            {
                "content": "Test content 2", 
                "metadata": {"title": "Doc 2", "source": "test"}
            }
        ]
        
        agent = BaseAgent(
            vector_db=mock_vector_db,
            name="Test Agent",
            role="tester", 
            description="Test",
            capabilities=[]
        )
        
        context = agent.get_context("test query", limit=2)
        
        assert "Test content 1" in context
        assert "Test content 2" in context
        assert "Doc 1" in context
        assert "Doc 2" in context
        mock_vector_db.search.assert_called_once_with("test query", 2)

    def test_get_context_handles_exception(self, mock_vector_db):
        """Test context retrieval handles database exceptions"""
        mock_vector_db.search.side_effect = Exception("Database error")
        
        agent = BaseAgent(
            vector_db=mock_vector_db,
            name="Test Agent",
            role="tester",
            description="Test", 
            capabilities=[]
        )
        
        context = agent.get_context("test query")
        assert context == ""

    @patch('openai.OpenAI')
    def test_generate_response_success(self, mock_openai_class, mock_vector_db):
        """Test successful response generation"""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test AI response"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = BaseAgent(
            vector_db=mock_vector_db,
            name="Test Agent", 
            role="tester",
            description="Test",
            capabilities=[]
        )
        
        response = agent.generate_response(
            system_prompt="You are a test assistant",
            user_query="Test query",
            context="Test context"
        )
        
        assert response == "Test AI response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_generate_response_handles_exception(self, mock_openai_class, mock_vector_db):
        """Test response generation handles OpenAI exceptions"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        agent = BaseAgent(
            vector_db=mock_vector_db,
            name="Test Agent",
            role="tester", 
            description="Test",
            capabilities=[]
        )
        
        response = agent.generate_response("system", "user", "context")
        assert response == "I'm sorry, I'm having trouble connecting to my AI service right now. Please try again later."


class TestResearchAgent:
    """Test the Research Agent specifically"""

    @patch('openai.OpenAI')
    def test_research_agent_initialization(self, mock_openai_class, mock_vector_db):
        """Test Research Agent initializes with correct properties"""
        agent = ResearchAgent(mock_vector_db)
        
        assert agent.name == "Research Agent"
        assert agent.role == "research_specialist"
        assert "market research" in agent.description.lower()
        assert len(agent.capabilities) > 0

    @patch('openai.OpenAI')
    def test_research_agent_query_processing(self, mock_openai_class, mock_vector_db):
        """Test Research Agent can process queries"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Research analysis result"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = ResearchAgent(mock_vector_db)
        response = agent.query("What are the latest AI trends?")
        
        assert response == "Research analysis result"


class TestCEOAgent:
    """Test the CEO Agent specifically"""

    @patch('openai.OpenAI') 
    def test_ceo_agent_initialization(self, mock_openai_class, mock_vector_db):
        """Test CEO Agent initializes with correct properties"""
        agent = CEOAgent(mock_vector_db)
        
        assert agent.name == "CEO Agent"
        assert agent.role == "chief_executive"
        assert "strategic" in agent.description.lower()
        assert len(agent.capabilities) > 0

    @patch('openai.OpenAI')
    def test_ceo_agent_strategic_query(self, mock_openai_class, mock_vector_db):
        """Test CEO Agent can handle strategic queries"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Strategic recommendation"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = CEOAgent(mock_vector_db)
        response = agent.query("What is our strategic direction?")
        
        assert response == "Strategic recommendation"