"""
Hybrid Retrieval System
Combines dense (vector) and sparse (BM25) search for optimal recall and precision
"""

import os
import json
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import xxhash
from datetime import datetime

# Import existing components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from vector_db import VectorDatabase

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval"""
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    top_k_factor: int = 2  # Retrieve 2x more results for reranking
    min_score_threshold: float = 0.0
    enable_reranking: bool = True
    cache_bm25_index: bool = True
    bm25_cache_dir: str = "./data/bm25_cache"

@dataclass
class SearchResult:
    """Unified search result structure"""
    id: str
    content: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    lexical_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0
    source: str = "unknown"  # 'dense', 'lexical', 'hybrid'

class BM25Index:
    """BM25 index wrapper with caching and persistence"""
    
    def __init__(self, cache_dir: str = "./data/bm25_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.doc_ids = []
        self.tokenized_docs = []
        self.index_hash = None
        
    def _compute_corpus_hash(self, documents: List[Dict[str, Any]]) -> str:
        """Compute hash of document corpus for cache validation"""
        content = "".join([doc.get('content', '') for doc in documents])
        return xxhash.xxh64(content.encode()).hexdigest()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # In production, use more sophisticated tokenization
        return text.lower().split()
    
    def _get_cache_path(self, corpus_hash: str) -> Path:
        """Get cache file path for given corpus hash"""
        return self.cache_dir / f"bm25_index_{corpus_hash}.pkl"
    
    def build_index(self, documents: List[Dict[str, Any]], force_rebuild: bool = False) -> bool:
        """Build BM25 index from documents"""
        try:
            if not documents:
                logger.warning("No documents provided for BM25 index")
                return False
            
            # Compute corpus hash
            corpus_hash = self._compute_corpus_hash(documents)
            cache_path = self._get_cache_path(corpus_hash)
            
            # Try to load from cache
            if not force_rebuild and cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.index = cached_data['index']
                        self.doc_ids = cached_data['doc_ids']
                        self.tokenized_docs = cached_data['tokenized_docs']
                        self.index_hash = corpus_hash
                    logger.info(f"Loaded BM25 index from cache: {cache_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load cached index: {e}")
            
            # Build new index
            logger.info(f"Building BM25 index for {len(documents)} documents")
            
            self.tokenized_docs = []
            self.doc_ids = []
            
            for doc in documents:
                content = doc.get('content', '')
                if content.strip():
                    tokens = self._tokenize(content)
                    self.tokenized_docs.append(tokens)
                    self.doc_ids.append(doc.get('id', str(len(self.doc_ids))))
            
            if not self.tokenized_docs:
                logger.error("No valid documents found for indexing")
                return False
            
            # Create BM25 index
            self.index = BM25Okapi(self.tokenized_docs)
            self.index_hash = corpus_hash
            
            # Cache the index
            try:
                cache_data = {
                    'index': self.index,
                    'doc_ids': self.doc_ids,
                    'tokenized_docs': self.tokenized_docs,
                    'created_at': datetime.now().isoformat(),
                    'doc_count': len(documents)
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cached BM25 index to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache index: {e}")
            
            logger.info(f"Built BM25 index with {len(self.doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search BM25 index"""
        if not self.index or not query.strip():
            return []
        
        try:
            query_tokens = self._tokenize(query)
            scores = self.index.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            
            for idx in top_indices:
                if idx < len(self.doc_ids) and scores[idx] > 0:
                    results.append((self.doc_ids[idx], float(scores[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

class HybridRetriever:
    """Hybrid retrieval combining dense and sparse search"""
    
    def __init__(self, 
                 vector_db: Optional[VectorDatabase] = None,
                 config: Optional[RetrievalConfig] = None):
        """
        Initialize hybrid retriever
        
        Args:
            vector_db: Vector database instance
            config: Retrieval configuration
        """
        self.vector_db = vector_db or VectorDatabase()
        self.config = config or RetrievalConfig()
        self.bm25_index = BM25Index(self.config.bm25_cache_dir)
        self.corpus_indexed = False
        
        # Validate configuration
        if abs(self.config.dense_weight + self.config.lexical_weight - 1.0) > 0.001:
            logger.warning("Dense and lexical weights don't sum to 1.0, normalizing...")
            total = self.config.dense_weight + self.config.lexical_weight
            self.config.dense_weight /= total
            self.config.lexical_weight /= total
        
        logger.info(f"Initialized HybridRetriever with weights: dense={self.config.dense_weight:.2f}, lexical={self.config.lexical_weight:.2f}")
    
    def build_bm25_index(self, force_rebuild: bool = False) -> bool:
        """Build BM25 index from vector database documents"""
        try:
            # Get all documents from vector database
            documents = self.vector_db.get_all_documents()
            
            if not documents:
                logger.warning("No documents found in vector database")
                return False
            
            # Convert to format needed by BM25Index
            bm25_docs = []
            for doc in documents:
                bm25_docs.append({
                    'id': doc['id'],
                    'content': doc.get('content', doc.get('content_preview', ''))
                })
            
            success = self.bm25_index.build_index(bm25_docs, force_rebuild)
            if success:
                self.corpus_indexed = True
                logger.info("BM25 index built successfully")
            else:
                logger.error("Failed to build BM25 index")
                
            return success
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            return False
    
    def _dense_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform dense vector search"""
        try:
            results = self.vector_db.search(query, limit=top_k)
            
            search_results = []
            for i, result in enumerate(results):
                # Convert distance to similarity score (assuming cosine distance)
                distance = result.get('distance', 1.0)
                similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity
                
                search_results.append(SearchResult(
                    id=result['id'],
                    content=result['content'],
                    metadata=result['metadata'],
                    dense_score=similarity,
                    lexical_score=0.0,
                    combined_score=similarity,
                    rank=i,
                    source='dense'
                ))
            
            logger.debug(f"Dense search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def _lexical_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform BM25 lexical search"""
        try:
            if not self.corpus_indexed:
                logger.warning("BM25 index not built, building now...")
                if not self.build_bm25_index():
                    logger.error("Failed to build BM25 index for lexical search")
                    return []
            
            bm25_results = self.bm25_index.search(query, top_k)
            
            search_results = []
            for i, (doc_id, score) in enumerate(bm25_results):
                # Get full document from vector DB
                doc = self.vector_db.get_document(doc_id)
                if doc:
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=doc['content'],
                        metadata=doc['metadata'],
                        dense_score=0.0,
                        lexical_score=score,
                        combined_score=score,
                        rank=i,
                        source='lexical'
                    ))
            
            logger.debug(f"Lexical search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
            return []
    
    def _normalize_scores(self, results: List[SearchResult], score_type: str) -> List[SearchResult]:
        """Normalize scores to [0, 1] range"""
        if not results:
            return results
        
        if score_type == 'dense':
            scores = [r.dense_score for r in results]
        elif score_type == 'lexical':
            scores = [r.lexical_score for r in results]
        else:
            return results
        
        if not scores or max(scores) == min(scores):
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        for i, result in enumerate(results):
            normalized_score = (scores[i] - min_score) / score_range
            if score_type == 'dense':
                result.dense_score = normalized_score
            elif score_type == 'lexical':
                result.lexical_score = normalized_score
        
        return results
    
    def _fuse_scores(self, 
                    dense_results: List[SearchResult], 
                    lexical_results: List[SearchResult]) -> List[SearchResult]:
        """Fuse dense and lexical search results using weighted combination"""
        
        # Normalize scores
        dense_results = self._normalize_scores(dense_results, 'dense')
        lexical_results = self._normalize_scores(lexical_results, 'lexical')
        
        # Create combined result set
        all_docs = {}
        
        # Add dense results
        for result in dense_results:
            all_docs[result.id] = SearchResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                dense_score=result.dense_score,
                lexical_score=0.0,
                combined_score=0.0,
                rank=0,
                source='dense'
            )
        
        # Merge lexical results
        for result in lexical_results:
            if result.id in all_docs:
                # Document found in both searches
                all_docs[result.id].lexical_score = result.lexical_score
                all_docs[result.id].source = 'hybrid'
            else:
                # Document only in lexical search
                all_docs[result.id] = SearchResult(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    dense_score=0.0,
                    lexical_score=result.lexical_score,
                    combined_score=0.0,
                    rank=0,
                    source='lexical'
                )
        
        # Calculate combined scores
        fused_results = []
        for doc in all_docs.values():
            combined_score = (
                self.config.dense_weight * doc.dense_score +
                self.config.lexical_weight * doc.lexical_score
            )
            doc.combined_score = combined_score
            
            # Apply minimum score threshold
            if combined_score >= self.config.min_score_threshold:
                fused_results.append(doc)
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(fused_results):
            result.rank = i
        
        logger.debug(f"Fused {len(dense_results)} dense + {len(lexical_results)} lexical = {len(fused_results)} results")
        return fused_results
    
    def search(self, 
               query: str, 
               top_k: int = 10, 
               use_reranking: Optional[bool] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining dense and lexical retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use reranking (overrides config)
            
        Returns:
            List of search results sorted by relevance
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        use_reranking = use_reranking if use_reranking is not None else self.config.enable_reranking
        search_k = top_k * self.config.top_k_factor if use_reranking else top_k
        
        logger.info(f"Hybrid search: '{query}' (top_k={top_k}, search_k={search_k}, reranking={use_reranking})")
        
        try:
            # Perform both searches in parallel conceptually
            dense_results = self._dense_search(query, search_k)
            lexical_results = self._lexical_search(query, search_k)
            
            # Fuse results
            fused_results = self._fuse_scores(dense_results, lexical_results)
            
            # Apply reranking if enabled
            if use_reranking and len(fused_results) > top_k:
                # Import reranker here to avoid circular imports
                try:
                    from .reranker import Reranker
                    reranker = Reranker()
                    fused_results = reranker.rerank(query, fused_results, top_k)
                    logger.debug(f"Applied reranking, reduced to {len(fused_results)} results")
                except ImportError:
                    logger.warning("Reranker not available, skipping reranking")
                except Exception as e:
                    logger.error(f"Reranking failed: {e}")
            
            # Return top-k results
            final_results = fused_results[:top_k]
            
            logger.info(f"Hybrid search completed: {len(final_results)} results returned")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense search only
            try:
                logger.info("Falling back to dense search only")
                return self._dense_search(query, top_k)
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        stats = {
            'config': {
                'dense_weight': self.config.dense_weight,
                'lexical_weight': self.config.lexical_weight,
                'enable_reranking': self.config.enable_reranking,
                'top_k_factor': self.config.top_k_factor
            },
            'corpus_indexed': self.corpus_indexed,
            'vector_db_status': self.vector_db.get_status()
        }
        
        if self.corpus_indexed:
            stats['bm25_stats'] = {
                'doc_count': len(self.bm25_index.doc_ids),
                'index_hash': self.bm25_index.index_hash
            }
        
        return stats
    
    def rebuild_index(self) -> bool:
        """Force rebuild of BM25 index"""
        logger.info("Force rebuilding BM25 index...")
        return self.build_bm25_index(force_rebuild=True)