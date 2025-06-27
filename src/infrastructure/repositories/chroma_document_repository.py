"""
ChromaDB implementation of document repository
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
import uuid

from src.domain.interfaces import IDocumentRepository, IVectorStore
from src.domain.entities import Document, SearchResult, SearchQuery, HealthStatus, DocumentStatus
from src.presentation.responses import NotFoundException, ExternalServiceException
from src.infrastructure.container import singleton

logger = logging.getLogger(__name__)


@singleton(IDocumentRepository)
class ChromaDocumentRepository(IDocumentRepository):
    """ChromaDB implementation of document repository"""
    
    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store
        self._documents: Dict[str, Document] = {}  # In-memory cache for metadata
        self._initialized = False
        self._client = None
        self._collection = None
        self._init_database()
    
    def _init_database(self):
        """Initialize ChromaDB connection"""
        try:
            persist_directory = "./chroma_db"
            collection_name = "vector_rag_collection"
            
            self._client = chromadb.PersistentClient(path=persist_directory)
            
            # Get or create collection
            try:
                self._collection = self._client.get_collection(name=collection_name)
                logger.info(f"Connected to existing collection: {collection_name}")
            except Exception:
                self._collection = self._client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise ExternalServiceException("ChromaDB", f"Database initialization failed: {e}")
    
    async def create(self, document: Document) -> Document:
        """Create a new document"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Database not initialized")
        
        try:
            # Add to vector store
            await self.vector_store.add_document(document)
            
            # Cache metadata
            self._documents[document.id] = document
            
            # Mark as completed
            document.mark_as_processed()
            
            logger.info(f"Created document: {document.id}")
            return document
            
        except Exception as e:
            document.mark_as_failed(str(e))
            logger.error(f"Failed to create document {document.id}: {e}")
            raise ExternalServiceException("ChromaDB", f"Failed to create document: {e}")
    
    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Database not initialized")
        
        # Check cache first
        if document_id in self._documents:
            return self._documents[document_id]
        
        try:
            # Fetch from ChromaDB
            result = self._collection.get(ids=[document_id])
            
            if not result['ids'] or len(result['ids']) == 0:
                return None
            
            # Reconstruct document from ChromaDB data
            metadata = result['metadatas'][0] if result['metadatas'] else {}
            content = result['documents'][0] if result['documents'] else ""
            
            document = Document(
                id=document_id,
                title=metadata.get('title', 'Untitled'),
                content=content,
                source=metadata.get('source', 'unknown'),
                status=DocumentStatus(metadata.get('status', 'completed')),
                metadata=metadata,
                created_at=datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat())),
                created_by=metadata.get('created_by')
            )
            
            # Cache the document
            self._documents[document_id] = document
            return document
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise ExternalServiceException("ChromaDB", f"Failed to retrieve document: {e}")
    
    async def get_all(self, offset: int = 0, limit: int = 100) -> List[Document]:
        """Get all documents with pagination"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Database not initialized")
        
        try:
            # Get batch from ChromaDB
            result = self._collection.get(
                limit=limit,
                offset=offset
            )
            
            documents = []
            
            if result['ids']:
                for i, doc_id in enumerate(result['ids']):
                    try:
                        metadata = result['metadatas'][i] if i < len(result['metadatas']) else {}
                        content = result['documents'][i] if i < len(result['documents']) else ""
                        
                        document = Document(
                            id=doc_id,
                            title=metadata.get('title', 'Untitled'),
                            content=content,
                            source=metadata.get('source', 'unknown'),
                            status=DocumentStatus(metadata.get('status', 'completed')),
                            metadata=metadata,
                            created_at=datetime.fromisoformat(metadata.get('timestamp', datetime.now().isoformat())),
                            created_by=metadata.get('created_by')
                        )
                        
                        documents.append(document)
                        
                        # Cache the document
                        self._documents[doc_id] = document
                        
                    except Exception as e:
                        logger.warning(f"Failed to reconstruct document {doc_id}: {e}")
                        continue
            
            logger.info(f"Retrieved {len(documents)} documents (offset: {offset}, limit: {limit})")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            raise ExternalServiceException("ChromaDB", f"Failed to retrieve documents: {e}")
    
    async def update(self, document: Document) -> Document:
        """Update existing document"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Database not initialized")
        
        try:
            # Check if document exists
            existing = await self.get_by_id(document.id)
            if not existing:
                raise NotFoundException("Document", document.id)
            
            # Update in vector store
            await self.vector_store.update_document(document)
            
            # Update timestamp
            document.updated_at = datetime.now()
            
            # Update cache
            self._documents[document.id] = document
            
            logger.info(f"Updated document: {document.id}")
            return document
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            raise ExternalServiceException("ChromaDB", f"Failed to update document: {e}")
    
    async def delete(self, document_id: str) -> bool:
        """Delete document by ID"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Database not initialized")
        
        try:
            # Check if document exists
            existing = await self.get_by_id(document_id)
            if not existing:
                raise NotFoundException("Document", document_id)
            
            # Delete from vector store
            await self.vector_store.delete_document(document_id)
            
            # Remove from cache
            self._documents.pop(document_id, None)
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise ExternalServiceException("ChromaDB", f"Failed to delete document: {e}")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search documents"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Database not initialized")
        
        try:
            # Use vector store for similarity search
            results = await self.vector_store.search_similar(query.query, query.limit)
            
            logger.info(f"Search query '{query.query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query.query}': {e}")
            raise ExternalServiceException("ChromaDB", f"Search failed: {e}")
    
    async def get_count(self) -> int:
        """Get total document count"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Database not initialized")
        
        try:
            count = self._collection.count()
            return count
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            raise ExternalServiceException("ChromaDB", f"Failed to get count: {e}")


@singleton(IVectorStore)
class ChromaVectorStore(IVectorStore):
    """ChromaDB implementation of vector store interface"""
    
    def __init__(self):
        self._client = None
        self._collection = None
        self._initialized = False
        self._init_connection()
    
    def _init_connection(self):
        """Initialize ChromaDB connection"""
        try:
            persist_directory = "./chroma_db"
            collection_name = "vector_rag_collection"
            
            self._client = chromadb.PersistentClient(path=persist_directory)
            
            # Get or create collection
            try:
                self._collection = self._client.get_collection(name=collection_name)
                logger.info(f"Connected to existing collection: {collection_name}")
            except Exception:
                self._collection = self._client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector store: {e}")
            self._initialized = False
    
    async def add_document(self, document: Document) -> bool:
        """Add document to vector store"""
        if not self._initialized:
            return False
        
        try:
            # Prepare metadata for ChromaDB
            metadata = {
                'title': document.title,
                'source': document.source,
                'timestamp': document.created_at.isoformat(),
                'status': document.status.value,
                'content_length': len(document.content),
                'created_by': document.created_by
            }
            
            # Add custom metadata
            metadata.update(document.metadata)
            
            # Add to collection
            self._collection.add(
                documents=[document.content],
                metadatas=[metadata],
                ids=[document.id]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to vector store: {e}")
            return False
    
    async def search_similar(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        if not self._initialized:
            raise ExternalServiceException("ChromaDB", "Vector store not initialized")
        
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                ids = results['ids'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [None] * len(documents)
                
                for i in range(len(documents)):
                    try:
                        relevance_score = 1 - distances[i] if distances[i] is not None else 0.5
                        
                        search_result = SearchResult(
                            document_id=ids[i],
                            content=documents[i],
                            metadata=metadatas[i] if i < len(metadatas) else {},
                            relevance_score=max(0, min(1, relevance_score)),
                            distance=distances[i] if i < len(distances) else None
                        )
                        
                        search_results.append(search_result)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process search result {i}: {e}")
                        continue
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise ExternalServiceException("ChromaDB", f"Vector search failed: {e}")
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store"""
        if not self._initialized:
            return False
        
        try:
            self._collection.delete(ids=[document_id])
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document from vector store: {e}")
            return False
    
    async def update_document(self, document: Document) -> bool:
        """Update document in vector store"""
        if not self._initialized:
            return False
        
        try:
            # ChromaDB doesn't have update, so we delete and re-add
            await self.delete_document(document.id)
            return await self.add_document(document)
            
        except Exception as e:
            logger.error(f"Failed to update document in vector store: {e}")
            return False
    
    async def get_health_status(self) -> HealthStatus:
        """Get vector store health status"""
        try:
            if not self._initialized:
                return HealthStatus(
                    component="ChromaDB",
                    healthy=False,
                    message="Not initialized"
                )
            
            # Test basic operation
            count = self._collection.count()
            
            return HealthStatus(
                component="ChromaDB",
                healthy=True,
                message=f"Operational with {count} documents",
                details={"document_count": count}
            )
            
        except Exception as e:
            return HealthStatus(
                component="ChromaDB",
                healthy=False,
                message=f"Health check failed: {e}"
            )