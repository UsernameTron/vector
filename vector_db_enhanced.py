"""
Enhanced Vector Database Module
ChromaDB integration with explicit embedding configuration and optimizations
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedVectorDatabase:
    """Enhanced vector database with explicit embedding configuration and optimizations"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB client with explicit embedding configuration
        
        Args:
            embedding_model: Choice of embedding model
                - "all-MiniLM-L6-v2": Fast, good performance (384 dim)
                - "all-mpnet-base-v2": Better quality, slower (768 dim)
                - "paraphrase-MiniLM-L6-v2": Optimized for paraphrase detection
        """
        self.persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db_enhanced')
        self.collection_name = os.getenv('DEFAULT_COLLECTION_NAME', 'vector_rag_enhanced')
        self.embedding_model_name = embedding_model
        
        # Initialize embedding function with explicit model
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize {embedding_model}, falling back to default: {e}")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Enhanced HNSW configuration for better performance
        self.collection_metadata = {
            "hnsw:space": "cosine",
            "hnsw:M": 16,  # Number of connections for every new element
            "hnsw:efConstruction": 100,  # Size of dynamic candidate list
            "hnsw:efSearch": 100,  # Search parameter
            "embedding_model": self.embedding_model_name
        }
        
        # Get or create collection with explicit embedding function
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=self.collection_metadata
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_document(self, content: str, title: str = "Untitled", source: str = "unknown", 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the vector database"""
        try:
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Prepare metadata
            doc_metadata = {
                "title": title,
                "source": source,
                "timestamp": timestamp,
                "content_length": len(content),
                "embedding_model": self.embedding_model_name
            }
            
            # Add custom metadata if provided
            if metadata:
                doc_metadata.update(metadata)
            
            # Add to collection
            self.collection.add(
                documents=[content],
                metadatas=[doc_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added document {doc_id}: {title} (length: {len(content)})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def search(self, query: str, limit: int = 5, 
               where: Optional[Dict[str, Any]] = None,
               min_relevance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar documents with enhanced filtering
        
        Args:
            query: Search query
            limit: Maximum number of results
            where: Metadata filters
            min_relevance: Minimum relevance score (1 - distance)
        """
        try:
            # Prepare query parameters
            query_params = {
                "query_texts": [query],
                "n_results": limit
            }
            
            if where:
                query_params["where"] = where
            
            results = self.collection.query(**query_params)
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    relevance_score = 1 - distance
                    
                    # Filter by minimum relevance
                    if relevance_score >= min_relevance:
                        formatted_results.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'distance': distance,
                            'relevance_score': relevance_score
                        })
            
            # Sort by relevance score
            formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"Search query: '{query}' returned {len(formatted_results)} results "
                       f"(min_relevance: {min_relevance})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
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
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all documents with pagination support"""
        try:
            results = self.collection.get(
                limit=limit,
                offset=offset
            )
            
            documents = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    doc_content = results['documents'][i] if results['documents'] else ""
                    preview_length = 200
                    content_preview = (doc_content[:preview_length] + "..." 
                                     if len(doc_content) > preview_length 
                                     else doc_content)
                    
                    documents.append({
                        'id': results['ids'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {},
                        'content_preview': content_preview,
                        'full_content_length': len(doc_content)
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
    
    def delete_documents_by_filter(self, where: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter"""
        try:
            # Get documents matching filter first
            results = self.collection.get(where=where)
            doc_ids = results['ids']
            
            if doc_ids:
                self.collection.delete(ids=doc_ids)
                logger.info(f"Deleted {len(doc_ids)} documents matching filter: {where}")
                return len(doc_ids)
            else:
                logger.info(f"No documents found matching filter: {where}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents with filter {where}: {str(e)}")
            return 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced database status information"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_docs = self.collection.get(limit=min(10, count))
            embedding_models = set()
            sources = set()
            
            if sample_docs['metadatas']:
                for metadata in sample_docs['metadatas']:
                    if 'embedding_model' in metadata:
                        embedding_models.add(metadata['embedding_model'])
                    if 'source' in metadata:
                        sources.add(metadata['source'])
            
            return {
                'status': 'connected',
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory,
                'embedding_model': self.embedding_model_name,
                'collection_metadata': self.collection_metadata,
                'embedding_models_in_use': list(embedding_models),
                'document_sources': list(sources),
                'hnsw_configuration': {
                    'M': self.collection_metadata.get('hnsw:M'),
                    'efConstruction': self.collection_metadata.get('hnsw:efConstruction'),
                    'space': self.collection_metadata.get('hnsw:space')
                }
            }
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def update_hnsw_params(self, ef_search: int = None):
        """Update HNSW search parameters for query optimization"""
        try:
            if ef_search:
                # This would require ChromaDB API support for runtime parameter updates
                logger.info(f"HNSW efSearch parameter update requested: {ef_search}")
                # Implementation depends on ChromaDB version and API availability
        except Exception as e:
            logger.error(f"Error updating HNSW parameters: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed collection statistics"""
        try:
            count = self.collection.count()
            if count == 0:
                return {'total_documents': 0}
            
            # Get all documents for analysis
            all_docs = self.collection.get()
            
            stats = {
                'total_documents': count,
                'embedding_model': self.embedding_model_name
            }
            
            if all_docs['metadatas']:
                # Analyze content lengths
                content_lengths = []
                sources = {}
                timestamps = []
                
                for i, metadata in enumerate(all_docs['metadatas']):
                    if 'content_length' in metadata:
                        content_lengths.append(metadata['content_length'])
                    
                    if 'source' in metadata:
                        source = metadata['source']
                        sources[source] = sources.get(source, 0) + 1
                    
                    if 'timestamp' in metadata:
                        timestamps.append(metadata['timestamp'])
                
                if content_lengths:
                    stats['content_stats'] = {
                        'avg_length': sum(content_lengths) / len(content_lengths),
                        'min_length': min(content_lengths),
                        'max_length': max(content_lengths),
                        'total_characters': sum(content_lengths)
                    }
                
                stats['sources'] = sources
                stats['date_range'] = {
                    'oldest': min(timestamps) if timestamps else None,
                    'newest': max(timestamps) if timestamps else None
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}