"""
Enhanced Vector Database
Drop-in replacement for existing VectorDatabase with hybrid retrieval, 
reranking, caching, and observability
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import original vector database
from vector_db import VectorDatabase

# Import new components
try:
    from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalConfig
    from src.retrieval.reranker import Reranker, RerankingConfig
    from src.cache.cache_manager import get_cache_manager, cache_get, cache_set
    from src.monitoring.telemetry import get_logger, trace_retrieval, record_metric
    from src.utils.resilience import resilient_database_call, resilient_api_call
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced components not fully available: {e}")
    ENHANCED_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnhancedConfig:
    """Configuration for enhanced vector database"""
    enable_hybrid_retrieval: bool = True
    enable_reranking: bool = True
    enable_caching: bool = True
    enable_observability: bool = True
    
    # Retrieval weights
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    
    # Reranking configuration
    rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    mmr_diversity_factor: float = 0.3
    
    # Cache configuration
    cache_ttl_seconds: int = 3600
    enable_search_cache: bool = True
    enable_embedding_cache: bool = True
    
    # Performance tuning
    top_k_factor: int = 2  # Retrieve 2x more for reranking
    min_cache_query_length: int = 10  # Only cache longer queries

class EnhancedVectorDatabase:
    """
    Enhanced Vector Database with hybrid retrieval, reranking, and caching
    Drop-in replacement for the original VectorDatabase class
    """
    
    def __init__(self, config: Optional[EnhancedConfig] = None):
        """Initialize enhanced vector database"""
        self.config = config or EnhancedConfig()
        
        # Initialize base vector database
        self.base_db = VectorDatabase()
        
        # Initialize enhanced components if available
        self.hybrid_retriever = None
        self.reranker = None
        self.cache_manager = None
        self.logger = None
        
        if ENHANCED_COMPONENTS_AVAILABLE:
            self._initialize_enhanced_components()
        else:
            logger.warning("Enhanced components not available, using basic functionality")
        
        # Track initialization status
        self.enhanced_mode = ENHANCED_COMPONENTS_AVAILABLE and (
            self.config.enable_hybrid_retrieval or 
            self.config.enable_reranking or 
            self.config.enable_caching
        )
        
        if self.enhanced_mode:
            logger.info("EnhancedVectorDatabase initialized with advanced features")
        else:
            logger.info("EnhancedVectorDatabase initialized in compatibility mode")
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced components"""
        try:
            # Initialize logger
            if self.config.enable_observability:
                self.logger = get_logger()
            
            # Initialize cache manager
            if self.config.enable_caching:
                self.cache_manager = get_cache_manager()
            
            # Initialize hybrid retriever
            if self.config.enable_hybrid_retrieval:
                retrieval_config = RetrievalConfig(
                    dense_weight=self.config.dense_weight,
                    lexical_weight=self.config.lexical_weight,
                    enable_reranking=self.config.enable_reranking,
                    top_k_factor=self.config.top_k_factor
                )
                self.hybrid_retriever = HybridRetriever(
                    vector_db=self.base_db,
                    config=retrieval_config
                )
                
                # Build BM25 index
                self.hybrid_retriever.build_bm25_index()
            
            # Initialize reranker
            if self.config.enable_reranking:
                rerank_config = RerankingConfig(
                    model_name=self.config.rerank_model,
                    mmr_diversity_factor=self.config.mmr_diversity_factor
                )
                self.reranker = Reranker(rerank_config)
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced components: {e}")
            # Fall back to basic mode
            self.hybrid_retriever = None
            self.reranker = None
            self.cache_manager = None
    
    def _should_cache_query(self, query: str) -> bool:
        """Determine if query should be cached"""
        return (
            self.config.enable_caching and
            self.config.enable_search_cache and
            len(query.strip()) >= self.config.min_cache_query_length
        )
    
    def _get_cache_key(self, query: str, limit: int) -> str:
        """Generate cache key for search query"""
        return f"{query.strip().lower()}:limit:{limit}"
    
    @resilient_database_call
    def add_document(self, content: str, title: str = "Untitled", source: str = "unknown") -> str:
        """Add a document to the vector database"""
        try:
            # Add to base database
            doc_id = self.base_db.add_document(content, title, source)
            
            # Invalidate BM25 index cache if using hybrid retrieval
            if self.hybrid_retriever:
                # Mark for index rebuild on next search
                self.hybrid_retriever.corpus_indexed = False
            
            # Record metrics
            if self.logger:
                record_metric("document_count", 1.0, {"operation": "add"})
            
            if self.logger:
                self.logger.info(
                    "Document added",
                    doc_id=doc_id,
                    title=title,
                    content_length=len(content),
                    enhanced_mode=self.enhanced_mode
                )
            
            return doc_id
            
        except Exception as e:
            if self.logger:
                self.logger.error("Failed to add document", error=str(e), title=title)
            raise
    
    @trace_retrieval
    @resilient_database_call
    def search(self, query: str, limit: int = 5, use_enhanced: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents with enhanced retrieval
        
        Args:
            query: Search query
            limit: Number of results to return
            use_enhanced: Force enable/disable enhanced features (overrides config)
            
        Returns:
            List of search results
        """
        use_enhanced_features = (
            use_enhanced if use_enhanced is not None 
            else self.enhanced_mode
        )
        
        # Try cache first
        cache_key = None
        if use_enhanced_features and self._should_cache_query(query):
            cache_key = self._get_cache_key(query, limit)
            cached_results = cache_get("search", cache_key)
            if cached_results:
                if self.logger:
                    record_metric("cache_hits", 1.0, {"type": "search"})
                    self.logger.debug("Search cache hit", query=query, limit=limit)
                return cached_results
        
        try:
            # Perform search
            if use_enhanced_features and self.hybrid_retriever:
                # Use hybrid retrieval
                search_results = self.hybrid_retriever.search(
                    query=query,
                    top_k=limit,
                    use_reranking=self.config.enable_reranking
                )
                
                # Convert SearchResult objects to dict format
                results = []
                for result in search_results:
                    result_dict = {
                        'id': result.id,
                        'content': result.content,
                        'metadata': result.metadata,
                        'distance': 1.0 - result.combined_score,  # Convert score to distance
                        'combined_score': result.combined_score,
                        'dense_score': result.dense_score,
                        'lexical_score': result.lexical_score,
                        'rank': result.rank,
                        'source': result.source
                    }
                    
                    # Add rerank score if available
                    if hasattr(result, 'rerank_score'):
                        result_dict['rerank_score'] = result.rerank_score
                    
                    results.append(result_dict)
                
            else:
                # Use basic vector search
                results = self.base_db.search(query, limit)
                
                # Add enhanced metadata
                for i, result in enumerate(results):
                    result.update({
                        'combined_score': 1.0 - result.get('distance', 0.5),
                        'rank': i,
                        'source': 'dense'
                    })
            
            # Cache results if applicable
            if use_enhanced_features and cache_key and results:
                cache_set("search", cache_key, results, self.config.cache_ttl_seconds)
                if self.logger:
                    record_metric("cache_misses", 1.0, {"type": "search"})
            
            # Log search
            if self.logger:
                self.logger.info(
                    "Search completed",
                    query=query,
                    limit=limit,
                    result_count=len(results),
                    enhanced_mode=use_enhanced_features,
                    cached=cache_key is not None
                )
            
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Search failed",
                    query=query,
                    limit=limit,
                    error=str(e),
                    enhanced_mode=use_enhanced_features
                )
            
            # Graceful degradation: try basic search if enhanced fails
            if use_enhanced_features and self.base_db:
                try:
                    logger.warning("Enhanced search failed, falling back to basic search")
                    return self.base_db.search(query, limit)
                except:
                    pass
            
            raise
    
    @resilient_database_call
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get a specific document by ID"""
        return self.base_db.get_document(doc_id)
    
    @resilient_database_call
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents (metadata only)"""
        return self.base_db.get_all_documents()
    
    @resilient_database_call
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        success = self.base_db.delete_document(doc_id)
        
        if success and self.hybrid_retriever:
            # Mark for index rebuild
            self.hybrid_retriever.corpus_indexed = False
        
        return success
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced database status information"""
        base_status = self.base_db.get_status()
        
        enhanced_status = {
            **base_status,
            'enhanced_mode': self.enhanced_mode,
            'features': {
                'hybrid_retrieval': self.config.enable_hybrid_retrieval and self.hybrid_retriever is not None,
                'reranking': self.config.enable_reranking and self.reranker is not None,
                'caching': self.config.enable_caching and self.cache_manager is not None,
                'observability': self.config.enable_observability and self.logger is not None
            }
        }
        
        # Add component-specific status
        if self.hybrid_retriever:
            enhanced_status['hybrid_retrieval'] = self.hybrid_retriever.get_stats()
        
        if self.reranker:
            enhanced_status['reranking'] = self.reranker.get_stats()
        
        if self.cache_manager:
            enhanced_status['cache'] = self.cache_manager.get_stats()
        
        return enhanced_status
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection"""
        return self.base_db.get_document_count()
    
    def rebuild_search_index(self) -> bool:
        """Rebuild search indices for enhanced performance"""
        if not self.hybrid_retriever:
            logger.info("No hybrid retriever available, skipping index rebuild")
            return True
        
        try:
            success = self.hybrid_retriever.rebuild_index()
            if self.logger:
                self.logger.info("Search index rebuild completed", success=success)
            return success
        except Exception as e:
            if self.logger:
                self.logger.error("Search index rebuild failed", error=str(e))
            return False
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """Clear search and embedding caches"""
        if not self.cache_manager:
            return True
        
        try:
            if cache_type:
                success = self.cache_manager.clear(cache_type)
            else:
                success = self.cache_manager.clear()
            
            if self.logger:
                self.logger.info("Cache cleared", cache_type=cache_type or "all", success=success)
            
            return success
        except Exception as e:
            if self.logger:
                self.logger.error("Cache clear failed", error=str(e))
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics"""
        stats = {
            'document_count': self.get_document_count(),
            'enhanced_mode': self.enhanced_mode,
            'config': {
                'dense_weight': self.config.dense_weight,
                'lexical_weight': self.config.lexical_weight,
                'enable_reranking': self.config.enable_reranking,
                'enable_caching': self.config.enable_caching
            }
        }
        
        # Add component stats
        if self.hybrid_retriever:
            stats['retrieval'] = self.hybrid_retriever.get_stats()
        
        if self.cache_manager:
            stats['cache'] = self.cache_manager.get_stats()
        
        return stats


# Backwards compatibility - create instance that can be imported as drop-in replacement
# This maintains compatibility with existing code that imports VectorDatabase

def create_enhanced_vector_db(
    enable_hybrid: bool = True,
    enable_reranking: bool = True,
    enable_caching: bool = True
) -> EnhancedVectorDatabase:
    """
    Create an enhanced vector database instance
    
    Args:
        enable_hybrid: Enable hybrid retrieval (dense + lexical)
        enable_reranking: Enable cross-encoder reranking
        enable_caching: Enable search result caching
        
    Returns:
        Enhanced vector database instance
    """
    config = EnhancedConfig(
        enable_hybrid_retrieval=enable_hybrid,
        enable_reranking=enable_reranking,
        enable_caching=enable_caching
    )
    
    return EnhancedVectorDatabase(config)


# For backwards compatibility, you can replace imports like this:
# from vector_db import VectorDatabase
# with:
# from enhanced_vector_db import EnhancedVectorDatabase as VectorDatabase