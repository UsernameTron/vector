"""
Robust Vector Database Module
ChromaDB integration with comprehensive error handling and recovery mechanisms
"""

import chromadb
from chromadb.config import Settings
import os
import uuid
import logging
import time
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from utils.error_handler import (
    ErrorCategories, 
    ErrorRecovery, 
    DependencyManager, 
    FileCleanupManager,
    handle_errors,
    safe_operation
)

logger = logging.getLogger(__name__)


class VectorDatabaseError(Exception):
    """Custom exception for vector database operations"""
    pass


class VectorDatabase:
    """Robust vector database with comprehensive error handling and recovery"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize ChromaDB client with error handling and recovery"""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        self.collection_name = os.getenv('DEFAULT_COLLECTION_NAME', 'vector_rag_collection')
        self.client = None
        self.collection = None
        self.is_initialized = False
        self.initialization_errors = []
        
        # Ensure persist directory exists
        self._ensure_persist_directory()
        
        # Initialize with recovery
        self._initialize_with_recovery()
    
    def _ensure_persist_directory(self):
        """Ensure the persist directory exists and is writable"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Test write permissions
            test_file = os.path.join(self.persist_directory, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            logger.info(f"Persist directory ready: {self.persist_directory}")
            
        except PermissionError as e:
            logger.error(f"Permission denied for persist directory {self.persist_directory}: {e}")
            # Try fallback directory
            fallback_dir = os.path.join(os.path.expanduser('~'), '.vector_rag_db')
            logger.info(f"Trying fallback directory: {fallback_dir}")
            self.persist_directory = fallback_dir
            self._ensure_persist_directory()
            
        except Exception as e:
            logger.error(f"Failed to create persist directory {self.persist_directory}: {e}")
            raise VectorDatabaseError(f"Cannot create database directory: {e}")
    
    def _initialize_with_recovery(self):
        """Initialize database with recovery mechanisms"""
        for attempt in range(self.max_retries):
            try:
                self._initialize_database()
                self.is_initialized = True
                logger.info(f"Vector database initialized successfully on attempt {attempt + 1}")
                return
                
            except Exception as e:
                self.initialization_errors.append(str(e))
                logger.warning(f"Initialization attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Try recovery strategies
                    if self._attempt_recovery(e, attempt):
                        continue
                    
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} initialization attempts failed")
                    self._handle_initialization_failure()
    
    def _initialize_database(self):
        """Core database initialization"""
        # Check if ChromaDB is available
        if not DependencyManager.check_dependency('chromadb'):
            raise VectorDatabaseError("ChromaDB dependency not available")
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB client initialized with path: {self.persist_directory}")
        except Exception as e:
            raise VectorDatabaseError(f"Failed to initialize ChromaDB client: {e}")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as e:
                raise VectorDatabaseError(f"Failed to create collection: {e}")
    
    def _attempt_recovery(self, error: Exception, attempt: int) -> bool:
        """Attempt to recover from initialization errors"""
        error_str = str(error).lower()
        
        # Recovery strategy 1: Corrupted database
        if 'corrupt' in error_str or 'database' in error_str:
            logger.info("Attempting recovery from corrupted database...")
            return self._recover_from_corruption()
        
        # Recovery strategy 2: Permission issues
        if 'permission' in error_str or 'access denied' in error_str:
            logger.info("Attempting recovery from permission issues...")
            return self._recover_from_permissions()
        
        # Recovery strategy 3: Lock issues
        if 'lock' in error_str or 'already in use' in error_str:
            logger.info("Attempting recovery from lock issues...")
            return self._recover_from_locks()
        
        # Recovery strategy 4: Collection name conflicts
        if 'collection' in error_str and 'exist' in error_str:
            logger.info("Attempting recovery from collection conflicts...")
            return self._recover_from_collection_conflict()
        
        return False
    
    def _recover_from_corruption(self) -> bool:
        """Recover from database corruption"""
        try:
            backup_dir = f"{self.persist_directory}_backup_{int(time.time())}"
            
            if os.path.exists(self.persist_directory):
                logger.info(f"Backing up corrupted database to: {backup_dir}")
                shutil.move(self.persist_directory, backup_dir)
            
            # Recreate directory
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info("Database directory recreated after corruption recovery")
            return True
            
        except Exception as e:
            logger.error(f"Corruption recovery failed: {e}")
            return False
    
    def _recover_from_permissions(self) -> bool:
        """Recover from permission issues"""
        try:
            # Try alternative directory
            alt_dir = os.path.join(os.path.expanduser('~'), '.vector_rag_fallback')
            logger.info(f"Switching to alternative directory: {alt_dir}")
            
            os.makedirs(alt_dir, exist_ok=True)
            self.persist_directory = alt_dir
            return True
            
        except Exception as e:
            logger.error(f"Permission recovery failed: {e}")
            return False
    
    def _recover_from_locks(self) -> bool:
        """Recover from lock issues"""
        try:
            # Wait longer and try to clear locks
            logger.info("Waiting for locks to clear...")
            time.sleep(5)
            
            # Remove potential lock files
            lock_patterns = ['*.lock', '*.lck', '*.pid']
            for pattern in lock_patterns:
                import glob
                lock_files = glob.glob(os.path.join(self.persist_directory, pattern))
                for lock_file in lock_files:
                    try:
                        os.remove(lock_file)
                        logger.info(f"Removed lock file: {lock_file}")
                    except Exception:
                        pass
            
            return True
            
        except Exception as e:
            logger.error(f"Lock recovery failed: {e}")
            return False
    
    def _recover_from_collection_conflict(self) -> bool:
        """Recover from collection name conflicts"""
        try:
            # Generate new collection name
            timestamp = int(time.time())
            self.collection_name = f"{self.collection_name}_{timestamp}"
            logger.info(f"Using new collection name: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Collection conflict recovery failed: {e}")
            return False
    
    def _handle_initialization_failure(self):
        """Handle complete initialization failure"""
        logger.error("Vector database initialization failed completely")
        logger.error(f"Errors encountered: {self.initialization_errors}")
        
        # Set up minimal fallback mode
        self.is_initialized = False
        self.client = None
        self.collection = None
    
    def _ensure_initialized(self):
        """Ensure database is initialized before operations"""
        if not self.is_initialized:
            raise VectorDatabaseError(
                "Vector database is not initialized. Please check logs for initialization errors."
            )
    
    @handle_errors(ErrorCategories.DATABASE)
    @safe_operation("add_document")
    def add_document(self, content: str, title: str = "Untitled", source: str = "unknown") -> str:
        """Add a document to the vector database with comprehensive error handling"""
        self._ensure_initialized()
        
        # Validate input
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        
        if len(content.strip()) == 0:
            raise ValueError("Content cannot be empty or whitespace only")
        
        try:
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Prepare metadata
            metadata = {
                "title": title[:500],  # Limit title length
                "source": source[:100],  # Limit source length
                "timestamp": timestamp,
                "content_length": len(content),
                "content_hash": hash(content)  # For duplicate detection
            }
            
            # Check for potential duplicates
            if self._is_duplicate_content(content):
                logger.warning(f"Potential duplicate content detected for: {title}")
            
            # Add to collection with retry logic
            self._add_with_retry(doc_id, content, metadata)
            
            logger.info(f"Added document {doc_id}: {title} ({len(content)} chars)")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document '{title}': {e}")
            raise VectorDatabaseError(f"Failed to add document: {e}")
    
    def _is_duplicate_content(self, content: str) -> bool:
        """Check if content might be a duplicate"""
        try:
            content_hash = hash(content)
            # Simple duplicate check - in production, use more sophisticated methods
            results = self.collection.get(
                where={"content_hash": content_hash},
                limit=1
            )
            return len(results.get('ids', [])) > 0
        except Exception:
            return False  # If check fails, proceed anyway
    
    def _add_with_retry(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Add document with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Add attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    @handle_errors(ErrorCategories.DATABASE)
    @safe_operation("search_documents")
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents with comprehensive error handling"""
        self._ensure_initialized()
        
        # Validate input
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if len(query.strip()) == 0:
            raise ValueError("Query cannot be empty or whitespace only")
        
        if limit <= 0 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        try:
            # Perform search with timeout and retry
            results = self._search_with_retry(query, limit)
            
            # Format results safely
            formatted_results = self._format_search_results(results)
            
            logger.info(f"Search query: '{query[:100]}...' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:100]}...': {e}")
            
            # Provide fallback empty results with helpful message
            return []
    
    def _search_with_retry(self, query: str, limit: int) -> Dict[str, Any]:
        """Perform search with retry logic"""
        for attempt in range(self.max_retries):
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                return results
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Search attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(self.retry_delay)
                else:
                    raise VectorDatabaseError(f"Search failed after {self.max_retries} attempts: {e}")
    
    def _format_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Safely format search results"""
        formatted_results = []
        
        try:
            if not results or not results.get('documents') or not results['documents'][0]:
                return formatted_results
            
            documents = results['documents'][0]
            ids = results.get('ids', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            for i in range(len(documents)):
                try:
                    result = {
                        'id': ids[i] if i < len(ids) else f"unknown_{i}",
                        'content': documents[i] if documents[i] else "",
                        'metadata': metadatas[i] if i < len(metadatas) and metadatas[i] else {},
                        'distance': distances[i] if i < len(distances) else None,
                        'relevance_score': 1 - distances[i] if i < len(distances) and distances[i] is not None else None
                    }
                    formatted_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to format search result {i}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to format search results: {e}")
        
        return formatted_results
    
    @handle_errors(ErrorCategories.DATABASE)
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID with error handling"""
        self._ensure_initialized()
        
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("Document ID must be a non-empty string")
        
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result and result.get('documents') and result['documents']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result.get('metadatas', [{}])[0]
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            raise VectorDatabaseError(f"Failed to retrieve document: {e}")
    
    @handle_errors(ErrorCategories.DATABASE)
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with error handling and pagination"""
        self._ensure_initialized()
        
        try:
            # Get documents in batches to avoid memory issues
            all_documents = []
            batch_size = 100
            offset = 0
            
            while True:
                try:
                    results = self.collection.get(
                        limit=batch_size,
                        offset=offset
                    )
                    
                    if not results or not results.get('ids') or len(results['ids']) == 0:
                        break
                    
                    batch_docs = self._format_document_list(results)
                    all_documents.extend(batch_docs)
                    
                    if len(results['ids']) < batch_size:
                        break  # Last batch
                    
                    offset += batch_size
                    
                except Exception as e:
                    logger.error(f"Failed to get document batch at offset {offset}: {e}")
                    break
            
            logger.info(f"Retrieved {len(all_documents)} documents")
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []  # Return empty list on error rather than failing
    
    def _format_document_list(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format document list results safely"""
        documents = []
        
        try:
            ids = results.get('ids', [])
            metadatas = results.get('metadatas', [])
            document_contents = results.get('documents', [])
            
            for i in range(len(ids)):
                try:
                    content = document_contents[i] if i < len(document_contents) else ""
                    preview = content[:200] + "..." if len(content) > 200 else content
                    
                    doc = {
                        'id': ids[i],
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'content_preview': preview
                    }
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"Failed to format document {i}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to format document list: {e}")
        
        return documents
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive database status information"""
        status = {
            'status': 'unknown',
            'initialized': self.is_initialized,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'document_count': 0,
            'initialization_errors': self.initialization_errors,
            'last_check': datetime.now().isoformat()
        }
        
        if not self.is_initialized:
            status['status'] = 'not_initialized'
            status['error'] = 'Database failed to initialize'
            return status
        
        try:
            count = self.collection.count()
            status.update({
                'status': 'connected',
                'document_count': count,
                'client_type': type(self.client).__name__,
                'collection_metadata': getattr(self.collection, 'metadata', {})
            })
            
        except Exception as e:
            logger.error(f"Failed to get database status: {e}")
            status.update({
                'status': 'error',
                'error': str(e)
            })
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            'healthy': False,
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check initialization
        health['checks']['initialized'] = {
            'status': self.is_initialized,
            'message': 'Database initialized' if self.is_initialized else 'Database not initialized'
        }
        
        if not self.is_initialized:
            health['checks']['initialization_errors'] = {
                'status': False,
                'message': f"Initialization failed: {'; '.join(self.initialization_errors)}"
            }
            return health
        
        # Check directory access
        try:
            os.access(self.persist_directory, os.R_OK | os.W_OK)
            health['checks']['directory_access'] = {
                'status': True,
                'message': f"Directory accessible: {self.persist_directory}"
            }
        except Exception as e:
            health['checks']['directory_access'] = {
                'status': False,
                'message': f"Directory access failed: {e}"
            }
        
        # Check database operations
        try:
            count = self.collection.count()
            health['checks']['database_operations'] = {
                'status': True,
                'message': f"Database operational, {count} documents"
            }
        except Exception as e:
            health['checks']['database_operations'] = {
                'status': False,
                'message': f"Database operations failed: {e}"
            }
        
        # Overall health
        health['healthy'] = all(check['status'] for check in health['checks'].values())
        
        return health