"""Unit tests for Vector Database functionality"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid
import tempfile
import shutil
import os

from vector_db import VectorDatabase


class TestVectorDatabase:
    """Test Vector Database functionality"""

    @patch('chromadb.PersistentClient')
    def test_vector_db_initialization_new_collection(self, mock_client_class):
        """Test VectorDB initialization with new collection"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate collection doesn't exist
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_collection = Mock()
        mock_client.create_collection.return_value = mock_collection
        
        db = VectorDatabase()
        
        assert db.client == mock_client
        assert db.collection == mock_collection
        mock_client.create_collection.assert_called_once()

    @patch('chromadb.PersistentClient')
    def test_vector_db_initialization_existing_collection(self, mock_client_class):
        """Test VectorDB initialization with existing collection"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate collection exists
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        
        db = VectorDatabase()
        
        assert db.client == mock_client
        assert db.collection == mock_collection
        mock_client.get_collection.assert_called_once()

    @patch('chromadb.PersistentClient')
    def test_add_document_success(self, mock_client_class):
        """Test successful document addition"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        db = VectorDatabase()
        
        doc_id = db.add_document(
            content="Test document content",
            title="Test Document",
            source="test_source"
        )
        
        # Verify document was added
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
        assert call_args[1]['documents'] == ["Test document content"]
        assert call_args[1]['metadatas'][0]['title'] == "Test Document"
        assert call_args[1]['metadatas'][0]['source'] == "test_source"

    @patch('chromadb.PersistentClient')
    def test_add_document_handles_exception(self, mock_client_class):
        """Test document addition handles exceptions"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.add.side_effect = Exception("Database error")
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        db = VectorDatabase()
        
        # The method raises exception, doesn't return None
        with pytest.raises(Exception, match="Database error"):
            db.add_document("Test content", "Test Title")

    @patch('chromadb.PersistentClient')
    def test_search_documents_success(self, mock_client_class):
        """Test successful document search"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [["Document 1", "Document 2"]],
            'metadatas': [[
                {'title': 'Doc 1', 'source': 'test'},
                {'title': 'Doc 2', 'source': 'test'}
            ]],
            'distances': [[0.1, 0.3]],
            'ids': [['id1', 'id2']]
        }
        
        db = VectorDatabase()
        results = db.search("test query", limit=2)
        
        assert len(results) == 2
        assert results[0]['content'] == "Document 1"
        assert results[0]['metadata']['title'] == 'Doc 1'
        assert results[0]['distance'] == 0.1
        assert results[1]['content'] == "Document 2"
        
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=2
        )

    @patch('chromadb.PersistentClient')
    def test_search_documents_empty_results(self, mock_client_class):
        """Test search with no results"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock empty results
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]],
            'ids': [[]]
        }
        
        db = VectorDatabase()
        results = db.search("nonexistent query")
        
        assert len(results) == 0

    @patch('chromadb.PersistentClient') 
    def test_search_handles_exception(self, mock_client_class):
        """Test search handles database exceptions"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Search error")
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        db = VectorDatabase()
        # The search method raises exceptions, doesn't return empty list
        with pytest.raises(Exception, match="Search error"):
            db.search("test query")

    @patch('chromadb.PersistentClient')
    def test_get_document_count_success(self, mock_client_class):
        """Test getting document count"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        db = VectorDatabase()
        count = db.get_document_count()
        
        assert count == 42
        mock_collection.count.assert_called_once()

    @patch('chromadb.PersistentClient')
    def test_get_document_count_handles_exception(self, mock_client_class):
        """Test document count handles exceptions"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.side_effect = Exception("Count error")
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        db = VectorDatabase()
        count = db.get_document_count()
        
        assert count == 0

    @patch.dict(os.environ, {
        'CHROMA_PERSIST_DIRECTORY': '/custom/path',
        'DEFAULT_COLLECTION_NAME': 'custom_collection'
    })
    @patch('chromadb.PersistentClient')
    def test_uses_environment_config(self, mock_client_class):
        """Test that VectorDB uses environment configuration"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        db = VectorDatabase()
        
        assert db.persist_directory == '/custom/path'
        assert db.collection_name == 'custom_collection'
        mock_client_class.assert_called_with(path='/custom/path')
        mock_client.get_collection.assert_called_with(name='custom_collection')