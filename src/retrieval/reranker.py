"""
Reranking Module
Cross-encoder based reranking with MMR for diversity
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RerankingConfig:
    """Configuration for reranking"""
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    max_length: int = 512
    batch_size: int = 32
    enable_mmr: bool = True
    mmr_diversity_factor: float = 0.3  # 0.0 = only relevance, 1.0 = only diversity
    cache_enabled: bool = True
    cache_dir: str = "./data/rerank_cache"

class Reranker:
    """Cross-encoder based reranker with MMR support"""
    
    def __init__(self, config: Optional[RerankingConfig] = None):
        """
        Initialize reranker
        
        Args:
            config: Reranking configuration
        """
        self.config = config or RerankingConfig()
        self.model = None
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load cross-encoder model
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading reranking model: {self.config.model_name}")
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length
            )
            logger.info("Reranking model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not available, reranking disabled")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load reranking model: {e}")
            self.model = None
    
    def _compute_cache_key(self, query: str, doc_ids: List[str]) -> str:
        """Compute cache key for reranking results"""
        import xxhash
        content = query + "".join(sorted(doc_ids))
        return xxhash.xxh64(content.encode()).hexdigest()
    
    def _get_cached_scores(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Get cached reranking scores"""
        if not self.config.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"rerank_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    return cached_data.get('scores', {})
            except Exception as e:
                logger.warning(f"Failed to load cached reranking scores: {e}")
        
        return None
    
    def _cache_scores(self, cache_key: str, scores: Dict[str, float]):
        """Cache reranking scores"""
        if not self.config.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"rerank_{cache_key}.pkl"
        try:
            cache_data = {
                'scores': scores,
                'model_name': self.config.model_name,
                'timestamp': np.datetime64('now').item().isoformat()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to cache reranking scores: {e}")
    
    def _cross_encoder_rerank(self, query: str, results: List[Any]) -> List[Any]:
        """Rerank using cross-encoder model"""
        if not self.model or not results:
            return results
        
        try:
            # Check cache first
            doc_ids = [r.id for r in results]
            cache_key = self._compute_cache_key(query, doc_ids)
            cached_scores = self._get_cached_scores(cache_key)
            
            if cached_scores and all(doc_id in cached_scores for doc_id in doc_ids):
                logger.debug("Using cached reranking scores")
                for result in results:
                    result.rerank_score = cached_scores[result.id]
            else:
                # Compute new scores
                logger.debug(f"Computing reranking scores for {len(results)} documents")
                
                # Prepare query-document pairs
                pairs = []
                for result in results:
                    # Truncate content if too long
                    content = result.content
                    if len(content) > 2000:  # Rough character limit
                        content = content[:2000] + "..."
                    pairs.append([query, content])
                
                # Compute scores in batches
                all_scores = []
                for i in range(0, len(pairs), self.config.batch_size):
                    batch_pairs = pairs[i:i+self.config.batch_size]
                    batch_scores = self.model.predict(batch_pairs)
                    all_scores.extend(batch_scores)
                
                # Assign scores to results
                score_dict = {}
                for i, result in enumerate(results):
                    score = float(all_scores[i])
                    result.rerank_score = score
                    score_dict[result.id] = score
                
                # Cache scores
                self._cache_scores(cache_key, score_dict)
            
            # Sort by rerank scores
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result.rank = i
                result.combined_score = result.rerank_score  # Update combined score
            
            logger.info(f"Cross-encoder reranking completed for {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results
    
    def _compute_diversity_matrix(self, results: List[Any]) -> np.ndarray:
        """Compute pairwise diversity matrix using content similarity"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight model for diversity computation
            diversity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get content embeddings
            contents = [result.content[:500] for result in results]  # Truncate for efficiency
            embeddings = diversity_model.encode(contents)
            
            # Compute cosine similarity matrix
            similarity_matrix = np.dot(embeddings, embeddings.T)
            norms = np.linalg.norm(embeddings, axis=1)
            similarity_matrix = similarity_matrix / np.outer(norms, norms)
            
            # Convert similarity to diversity (1 - similarity)
            diversity_matrix = 1.0 - similarity_matrix
            return diversity_matrix
            
        except Exception as e:
            logger.warning(f"Failed to compute diversity matrix: {e}")
            # Fallback: assume all documents are equally diverse
            n = len(results)
            return np.ones((n, n)) - np.eye(n)
    
    def _mmr_rerank(self, results: List[Any], lambda_param: float = 0.7) -> List[Any]:
        """Maximal Marginal Relevance reranking for diversity"""
        if len(results) <= 1:
            return results
        
        try:
            logger.debug(f"Applying MMR reranking with λ={lambda_param}")
            
            # Compute diversity matrix
            diversity_matrix = self._compute_diversity_matrix(results)
            
            # Extract relevance scores (assume rerank_score exists)
            relevance_scores = np.array([
                getattr(result, 'rerank_score', result.combined_score) 
                for result in results
            ])
            
            # Normalize relevance scores to [0, 1]
            if relevance_scores.max() > relevance_scores.min():
                relevance_scores = (relevance_scores - relevance_scores.min()) / (
                    relevance_scores.max() - relevance_scores.min()
                )
            
            # MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(results)))
            
            # Select first document (highest relevance)
            first_idx = np.argmax(relevance_scores)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Iteratively select remaining documents
            while remaining_indices and len(selected_indices) < len(results):
                mmr_scores = []
                
                for idx in remaining_indices:
                    # Relevance component
                    relevance = relevance_scores[idx]
                    
                    # Diversity component (minimum similarity to selected documents)
                    if selected_indices:
                        similarities = [1.0 - diversity_matrix[idx][sel_idx] for sel_idx in selected_indices]
                        max_similarity = max(similarities)
                    else:
                        max_similarity = 0.0
                    
                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    mmr_scores.append((idx, mmr_score))
                
                # Select document with highest MMR score
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Reorder results
            reordered_results = [results[i] for i in selected_indices]
            
            # Update ranks and scores
            for i, result in enumerate(reordered_results):
                result.rank = i
                # Keep original scores, just reorder
            
            logger.debug(f"MMR reranking completed, reordered {len(reordered_results)} documents")
            return reordered_results
            
        except Exception as e:
            logger.error(f"MMR reranking failed: {e}")
            return results
    
    def rerank(self, 
              query: str, 
              results: List[Any], 
              top_k: Optional[int] = None,
              use_mmr: Optional[bool] = None) -> List[Any]:
        """
        Rerank search results using cross-encoder and optionally MMR
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of results to return (default: all)
            use_mmr: Whether to apply MMR for diversity (overrides config)
            
        Returns:
            Reranked list of results
        """
        if not results:
            return results
        
        use_mmr = use_mmr if use_mmr is not None else self.config.enable_mmr
        top_k = top_k or len(results)
        
        logger.info(f"Reranking {len(results)} results for query: '{query}' (MMR={use_mmr})")
        
        try:
            # Step 1: Cross-encoder reranking for relevance
            reranked_results = self._cross_encoder_rerank(query, results.copy())
            
            # Step 2: Apply MMR for diversity if enabled
            if use_mmr and len(reranked_results) > 1:
                lambda_param = 1.0 - self.config.mmr_diversity_factor
                reranked_results = self._mmr_rerank(reranked_results, lambda_param)
            
            # Step 3: Return top-k results
            final_results = reranked_results[:top_k]
            
            logger.info(f"Reranking completed: returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranking statistics"""
        return {
            'config': {
                'model_name': self.config.model_name,
                'enable_mmr': self.config.enable_mmr,
                'mmr_diversity_factor': self.config.mmr_diversity_factor,
                'cache_enabled': self.config.cache_enabled
            },
            'model_loaded': self.model is not None,
            'cache_files': len(list(self.cache_dir.glob('rerank_*.pkl'))) if self.cache_dir.exists() else 0
        }
    
    def clear_cache(self):
        """Clear reranking cache"""
        try:
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob('rerank_*.pkl'):
                    cache_file.unlink()
                logger.info("Reranking cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")