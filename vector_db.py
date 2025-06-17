"""
Vector Database Module
ChromaDB integration for document storage and retrieval
"""

import chromadb
from chromadb.config import Settings
import os
import uuid
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        """Initialize ChromaDB client and collection"""
        self.persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        self.collection_name = os.getenv('DEFAULT_COLLECTION_NAME', 'vector_rag_collection')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_document(self, content: str, title: str = "Untitled", source: str = "unknown") -> str:
        """Add a document to the vector database"""
        try:
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Prepare metadata
            metadata = {
                "title": title,
                "source": source,
                "timestamp": timestamp,
                "content_length": len(content)
            }
            
            # Add to collection
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added document {doc_id}: {title}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            logger.info(f"Search query: '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get a specific document by ID"""
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result['documents']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {str(e)}")
            raise
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents (metadata only)"""
        try:
            results = self.collection.get()
            
            documents = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    documents.append({
                        'id': results['ids'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {},
                        'content_preview': results['documents'][i][:200] + "..." if results['documents'] and len(results['documents'][i]) > 200 else results['documents'][i] if results['documents'] else ""
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get database status information"""
        try:
            count = self.collection.count()
            return {
                'status': 'connected',
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
