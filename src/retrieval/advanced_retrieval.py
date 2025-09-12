"""
Advanced Retrieval Patterns Implementation

This module implements state-of-the-art retrieval patterns including:
- ColBERT: Contextualized Late Interaction over BERT
- SPLADE: SParse Lexical AnD Expansion model
- Dense Passage Retrieval (DPR)
- Hybrid retrieval combining multiple approaches

Author: RAG Enhancement System
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    """Result from advanced retrieval pattern"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str
    latency_ms: float


@dataclass
class ColBERTConfig:
    """Configuration for ColBERT retrieval"""
    model_name: str = "colbert-ir/colbertv2.0"
    max_doc_length: int = 512
    max_query_length: int = 64
    similarity_metric: str = "maxsim"  # maxsim, avg, sum
    dim: int = 128
    mask_punctuation: bool = True
    attend_to_mask_tokens: bool = False


@dataclass
class SPLADEConfig:
    """Configuration for SPLADE retrieval"""
    model_name: str = "naver/splade-cocondenser-ensembledistil"
    max_length: int = 512
    alpha: float = 1e-4  # Sparsity regularization
    beta: float = 1e-2   # FLOPS regularization
    agg: str = "max"     # Aggregation method: max, sum, mean
    
    
@dataclass
class DenseRetrievalConfig:
    """Configuration for dense passage retrieval"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    normalize_embeddings: bool = True
    similarity_metric: str = "cosine"


class BaseRetriever(ABC):
    """Abstract base class for retrieval patterns"""
    
    @abstractmethod
    def encode_query(self, query: str) -> Union[np.ndarray, torch.Tensor]:
        """Encode query for retrieval"""
        pass
    
    @abstractmethod
    def encode_documents(self, documents: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """Encode documents for indexing"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[RetrievalResult]:
        """Perform retrieval"""
        pass


class ColBERTRetriever(BaseRetriever):
    """
    ColBERT (Contextualized Late Interaction over BERT) implementation
    
    ColBERT produces fine-grained representations and uses late interaction
    for efficient and effective retrieval.
    """
    
    def __init__(self, config: ColBERTConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.eval()
            
            # Linear projection for dimensionality reduction
            self.linear = nn.Linear(self.model.config.hidden_size, config.dim)
            
        except Exception as e:
            self.logger.warning(f"Failed to load ColBERT model {config.model_name}: {e}")
            # Fallback to sentence transformer
            config.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.eval()
            self.linear = nn.Linear(self.model.config.hidden_size, config.dim)
    
    def _tokenize_and_encode(self, texts: List[str], max_length: int) -> torch.Tensor:
        """Tokenize and encode texts"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            
            # Apply linear projection
            embeddings = self.linear(last_hidden_state)
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings, encoded['attention_mask']
    
    def _mask_punctuation(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mask punctuation tokens if configured"""
        if not self.config.mask_punctuation:
            return attention_mask
            
        # Simple punctuation masking (can be enhanced with proper token classification)
        punctuation_ids = [self.tokenizer.convert_tokens_to_ids(token) 
                          for token in ['.', ',', '!', '?', ';', ':']]
        
        for punct_id in punctuation_ids:
            if punct_id is not None:
                punct_mask = (tokens == punct_id)
                attention_mask = attention_mask & ~punct_mask
                
        return attention_mask
    
    def _maxsim_similarity(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor,
                          query_mask: torch.Tensor, doc_mask: torch.Tensor) -> torch.Tensor:
        """Compute MaxSim similarity between query and document"""
        # query_embeddings: [batch_size, query_len, dim]
        # doc_embeddings: [batch_size, doc_len, dim]
        
        batch_size = query_embeddings.size(0)
        similarities = torch.zeros(batch_size)
        
        for i in range(batch_size):
            q_emb = query_embeddings[i][query_mask[i].bool()]  # [valid_query_len, dim]
            d_emb = doc_embeddings[i][doc_mask[i].bool()]      # [valid_doc_len, dim]
            
            if q_emb.size(0) == 0 or d_emb.size(0) == 0:
                similarities[i] = 0.0
                continue
            
            # Compute cosine similarity matrix
            sim_matrix = torch.mm(q_emb, d_emb.t())  # [valid_query_len, valid_doc_len]
            
            # MaxSim: for each query token, find max similarity with document tokens
            max_sims = torch.max(sim_matrix, dim=1)[0]  # [valid_query_len]
            similarities[i] = torch.sum(max_sims)
            
        return similarities
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query for ColBERT retrieval"""
        embeddings, attention_mask = self._tokenize_and_encode([query], self.config.max_query_length)
        tokens = self.tokenizer(query, return_tensors="pt")['input_ids']
        attention_mask = self._mask_punctuation(tokens, attention_mask)
        
        return embeddings, attention_mask
    
    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """Encode documents for ColBERT indexing"""
        embeddings_list = []
        masks_list = []
        
        # Process in batches to manage memory
        batch_size = 8
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            embeddings, attention_mask = self._tokenize_and_encode(batch_docs, self.config.max_doc_length)
            
            # Mask punctuation for each document
            for j, doc in enumerate(batch_docs):
                tokens = self.tokenizer(doc, return_tensors="pt")['input_ids']
                attention_mask[j] = self._mask_punctuation(tokens, attention_mask[j])
            
            embeddings_list.append(embeddings)
            masks_list.append(attention_mask)
        
        return torch.cat(embeddings_list, dim=0), torch.cat(masks_list, dim=0)
    
    def retrieve(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[RetrievalResult]:
        """Perform ColBERT retrieval"""
        start_time = time.time()
        
        try:
            # Extract document texts
            doc_texts = [doc.get('content', '') for doc in documents]
            
            # Encode query and documents
            query_emb, query_mask = self.encode_query(query)
            doc_embs, doc_masks = self.encode_documents(doc_texts)
            
            # Compute similarities
            similarities = []
            for i in range(len(doc_texts)):
                sim = self._maxsim_similarity(
                    query_emb.unsqueeze(0),
                    doc_embs[i:i+1],
                    query_mask,
                    doc_masks[i:i+1]
                )
                similarities.append(sim.item())
            
            # Rank and select top-k
            similarities = np.array(similarities)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            latency_ms = (time.time() - start_time) * 1000
            
            results = []
            for idx in top_indices:
                results.append(RetrievalResult(
                    document_id=documents[idx].get('id', f'doc_{idx}'),
                    content=doc_texts[idx],
                    score=similarities[idx],
                    metadata=documents[idx].get('metadata', {}),
                    retrieval_method="ColBERT",
                    latency_ms=latency_ms / len(top_indices)
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"ColBERT retrieval error: {e}")
            return []


class SPLADERetriever(BaseRetriever):
    """
    SPLADE (SParse Lexical AnD Expansion model) implementation
    
    SPLADE learns sparse representations with term expansion for effective retrieval.
    """
    
    def __init__(self, config: SPLADEConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.eval()
            
            # Additional layers for SPLADE
            self.output_layer = nn.Linear(self.model.config.hidden_size, self.tokenizer.vocab_size)
            
        except Exception as e:
            self.logger.warning(f"Failed to load SPLADE model {config.model_name}: {e}")
            # Fallback to sentence transformer
            config.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.eval()
            self.output_layer = nn.Linear(self.model.config.hidden_size, self.tokenizer.vocab_size)
    
    def _encode_with_splade(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using SPLADE approach"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            
            # Project to vocabulary space
            vocab_logits = self.output_layer(last_hidden_state)
            
            # Apply ReLU and log(1 + x) for sparsity
            sparse_repr = torch.log(1 + F.relu(vocab_logits))
            
            # Aggregate over sequence dimension
            if self.config.agg == "max":
                sparse_repr = torch.max(sparse_repr, dim=1)[0]
            elif self.config.agg == "sum":
                sparse_repr = torch.sum(sparse_repr, dim=1)
            else:  # mean
                sparse_repr = torch.mean(sparse_repr, dim=1)
            
            # Apply sparsity regularization
            sparse_repr = sparse_repr * (sparse_repr > self.config.alpha).float()
            
        return sparse_repr
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query for SPLADE retrieval"""
        return self._encode_with_splade([query])
    
    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """Encode documents for SPLADE indexing"""
        doc_representations = []
        
        # Process in batches
        batch_size = 16
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_repr = self._encode_with_splade(batch_docs)
            doc_representations.append(batch_repr)
        
        return torch.cat(doc_representations, dim=0)
    
    def retrieve(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[RetrievalResult]:
        """Perform SPLADE retrieval"""
        start_time = time.time()
        
        try:
            # Extract document texts
            doc_texts = [doc.get('content', '') for doc in documents]
            
            # Encode query and documents
            query_repr = self.encode_query(query)
            doc_reprs = self.encode_documents(doc_texts)
            
            # Compute sparse similarities (dot product for sparse vectors)
            similarities = torch.mm(query_repr, doc_reprs.t()).squeeze().numpy()
            
            # Rank and select top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            latency_ms = (time.time() - start_time) * 1000
            
            results = []
            for idx in top_indices:
                results.append(RetrievalResult(
                    document_id=documents[idx].get('id', f'doc_{idx}'),
                    content=doc_texts[idx],
                    score=similarities[idx],
                    metadata=documents[idx].get('metadata', {}),
                    retrieval_method="SPLADE",
                    latency_ms=latency_ms / len(top_indices)
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"SPLADE retrieval error: {e}")
            return []


class DensePassageRetriever(BaseRetriever):
    """
    Dense Passage Retrieval (DPR) implementation
    
    Uses dense vector representations for semantic similarity matching.
    """
    
    def __init__(self, config: DenseRetrievalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load dense retrieval model: {e}")
            raise
    
    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to dense vectors"""
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            if self.config.normalize_embeddings:
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.numpy()
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query for dense retrieval"""
        return self._encode_texts([query])
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents for dense indexing"""
        return self._encode_texts(documents)
    
    def retrieve(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[RetrievalResult]:
        """Perform dense passage retrieval"""
        start_time = time.time()
        
        try:
            # Extract document texts
            doc_texts = [doc.get('content', '') for doc in documents]
            
            # Encode query and documents
            query_embedding = self.encode_query(query)
            doc_embeddings = self.encode_documents(doc_texts)
            
            # Compute similarities
            if self.config.similarity_metric == "cosine":
                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            else:  # dot product
                similarities = np.dot(doc_embeddings, query_embedding.T).squeeze()
            
            # Rank and select top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            latency_ms = (time.time() - start_time) * 1000
            
            results = []
            for idx in top_indices:
                results.append(RetrievalResult(
                    document_id=documents[idx].get('id', f'doc_{idx}'),
                    content=doc_texts[idx],
                    score=similarities[idx],
                    metadata=documents[idx].get('metadata', {}),
                    retrieval_method="DPR",
                    latency_ms=latency_ms / len(top_indices)
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dense retrieval error: {e}")
            return []


@dataclass 
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval combining multiple methods"""
    colbert_config: Optional[ColBERTConfig] = None
    splade_config: Optional[SPLADEConfig] = None
    dense_config: Optional[DenseRetrievalConfig] = None
    weights: Dict[str, float] = None  # Weights for combining scores
    fusion_method: str = "weighted_sum"  # weighted_sum, rrf, comb_sum
    rrf_k: int = 60  # Parameter for Reciprocal Rank Fusion


class HybridAdvancedRetriever:
    """
    Hybrid retrieval system combining ColBERT, SPLADE, and Dense retrieval
    
    Provides state-of-the-art retrieval performance by leveraging the strengths
    of different retrieval paradigms.
    """
    
    def __init__(self, config: HybridRetrievalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize retrievers
        self.retrievers = {}
        
        if config.colbert_config:
            try:
                self.retrievers['colbert'] = ColBERTRetriever(config.colbert_config)
                self.logger.info("ColBERT retriever initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ColBERT: {e}")
        
        if config.splade_config:
            try:
                self.retrievers['splade'] = SPLADERetriever(config.splade_config)
                self.logger.info("SPLADE retriever initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SPLADE: {e}")
        
        if config.dense_config:
            try:
                self.retrievers['dense'] = DensePassageRetriever(config.dense_config)
                self.logger.info("Dense retriever initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Dense retriever: {e}")
        
        # Default weights if not provided
        if config.weights is None:
            num_retrievers = len(self.retrievers)
            self.config.weights = {name: 1.0/num_retrievers for name in self.retrievers.keys()}
        
        self.logger.info(f"Hybrid retriever initialized with {len(self.retrievers)} methods: {list(self.retrievers.keys())}")
    
    def _reciprocal_rank_fusion(self, result_lists: List[List[RetrievalResult]], k: int = 60) -> List[RetrievalResult]:
        """Apply Reciprocal Rank Fusion to combine multiple ranking lists"""
        doc_scores = {}
        doc_results = {}
        
        for results in result_lists:
            for rank, result in enumerate(results):
                doc_id = result.document_id
                score = 1.0 / (k + rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_results[doc_id] = result
                
                doc_scores[doc_id] += score
        
        # Sort by combined scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final results
        final_results = []
        for doc_id, score in sorted_docs:
            result = doc_results[doc_id]
            result.score = score
            result.retrieval_method = "RRF_Hybrid"
            final_results.append(result)
        
        return final_results
    
    def _weighted_sum_fusion(self, result_lists: List[List[RetrievalResult]], 
                           method_names: List[str]) -> List[RetrievalResult]:
        """Apply weighted sum fusion to combine results"""
        doc_scores = {}
        doc_results = {}
        
        for results, method_name in zip(result_lists, method_names):
            weight = self.config.weights.get(method_name, 1.0)
            
            for result in results:
                doc_id = result.document_id
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_results[doc_id] = result
                
                doc_scores[doc_id] += weight * result.score
        
        # Normalize scores
        max_score = max(doc_scores.values()) if doc_scores else 1.0
        for doc_id in doc_scores:
            doc_scores[doc_id] /= max_score
        
        # Sort by combined scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final results
        final_results = []
        for doc_id, score in sorted_docs:
            result = doc_results[doc_id]
            result.score = score
            result.retrieval_method = "WeightedSum_Hybrid"
            final_results.append(result)
        
        return final_results
    
    async def retrieve_async(self, query: str, documents: List[Dict[str, Any]], 
                           top_k: int = 10) -> List[RetrievalResult]:
        """Perform asynchronous hybrid retrieval"""
        start_time = time.time()
        
        # Run retrievers in parallel
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as executor:
            futures = {}
            for name, retriever in self.retrievers.items():
                future = executor.submit(retriever.retrieve, query, documents, top_k * 2)
                futures[name] = future
            
            # Collect results
            result_lists = []
            method_names = []
            for name, future in futures.items():
                try:
                    results = future.result(timeout=30)  # 30 second timeout
                    result_lists.append(results)
                    method_names.append(name)
                except Exception as e:
                    self.logger.warning(f"Retriever {name} failed: {e}")
        
        if not result_lists:
            self.logger.error("All retrievers failed")
            return []
        
        # Combine results using specified fusion method
        if self.config.fusion_method == "rrf":
            combined_results = self._reciprocal_rank_fusion(result_lists, self.config.rrf_k)
        else:  # weighted_sum or comb_sum
            combined_results = self._weighted_sum_fusion(result_lists, method_names)
        
        # Update latency
        total_latency = (time.time() - start_time) * 1000
        for result in combined_results[:top_k]:
            result.latency_ms = total_latency / min(top_k, len(combined_results))
        
        return combined_results[:top_k]
    
    def retrieve(self, query: str, documents: List[Dict[str, Any]], 
                top_k: int = 10) -> List[RetrievalResult]:
        """Perform hybrid retrieval (synchronous wrapper)"""
        return asyncio.run(self.retrieve_async(query, documents, top_k))
    
    def get_retriever_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for individual retrievers"""
        performance = {}
        
        for name, retriever in self.retrievers.items():
            performance[name] = {
                "type": type(retriever).__name__,
                "config": retriever.config.__dict__ if hasattr(retriever, 'config') else {},
                "weight": self.config.weights.get(name, 1.0),
                "status": "active"
            }
        
        return performance


class AdvancedRetrievalManager:
    """
    Manager class for advanced retrieval patterns
    
    Provides a unified interface for accessing different retrieval methods
    and handles configuration, initialization, and result processing.
    """
    
    def __init__(self, enable_colbert: bool = True, enable_splade: bool = True, 
                 enable_dense: bool = True, custom_configs: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.custom_configs = custom_configs or {}
        
        # Initialize hybrid configuration
        hybrid_config = HybridRetrievalConfig()
        
        if enable_colbert:
            colbert_config = ColBERTConfig()
            if 'colbert' in self.custom_configs:
                colbert_config.__dict__.update(self.custom_configs['colbert'])
            hybrid_config.colbert_config = colbert_config
        
        if enable_splade:
            splade_config = SPLADEConfig()
            if 'splade' in self.custom_configs:
                splade_config.__dict__.update(self.custom_configs['splade'])
            hybrid_config.splade_config = splade_config
        
        if enable_dense:
            dense_config = DenseRetrievalConfig()
            if 'dense' in self.custom_configs:
                dense_config.__dict__.update(self.custom_configs['dense'])
            hybrid_config.dense_config = dense_config
        
        # Apply hybrid configuration
        if 'hybrid' in self.custom_configs:
            for key, value in self.custom_configs['hybrid'].items():
                if hasattr(hybrid_config, key):
                    setattr(hybrid_config, key, value)
        
        # Initialize hybrid retriever
        try:
            self.hybrid_retriever = HybridAdvancedRetriever(hybrid_config)
            self.logger.info("Advanced retrieval manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced retrieval manager: {e}")
            raise
    
    def retrieve(self, query: str, documents: List[Dict[str, Any]], 
                top_k: int = 10, method: str = "hybrid") -> List[RetrievalResult]:
        """
        Perform retrieval using specified method
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results to return
            method: Retrieval method ('hybrid', 'colbert', 'splade', 'dense')
        
        Returns:
            List of retrieval results
        """
        if method == "hybrid":
            return self.hybrid_retriever.retrieve(query, documents, top_k)
        elif method in self.hybrid_retriever.retrievers:
            retriever = self.hybrid_retriever.retrievers[method]
            return retriever.retrieve(query, documents, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    async def retrieve_async(self, query: str, documents: List[Dict[str, Any]], 
                           top_k: int = 10) -> List[RetrievalResult]:
        """Perform asynchronous hybrid retrieval"""
        return await self.hybrid_retriever.retrieve_async(query, documents, top_k)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all retrievers"""
        return {
            "retrievers": self.hybrid_retriever.get_retriever_performance(),
            "fusion_method": self.hybrid_retriever.config.fusion_method,
            "weights": self.hybrid_retriever.config.weights
        }
    
    def benchmark_retrievers(self, test_queries: List[str], 
                           documents: List[Dict[str, Any]], 
                           top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Benchmark individual retrievers on test queries
        
        Returns:
            Performance metrics for each retriever
        """
        results = {}
        
        for method_name in self.hybrid_retriever.retrievers.keys():
            method_results = []
            total_time = 0
            
            for query in test_queries:
                start_time = time.time()
                retrieval_results = self.retrieve(query, documents, top_k, method_name)
                end_time = time.time()
                
                method_results.append({
                    'num_results': len(retrieval_results),
                    'avg_score': np.mean([r.score for r in retrieval_results]) if retrieval_results else 0,
                    'latency_ms': (end_time - start_time) * 1000
                })
                total_time += (end_time - start_time)
            
            # Aggregate metrics
            results[method_name] = {
                'avg_num_results': np.mean([r['num_results'] for r in method_results]),
                'avg_score': np.mean([r['avg_score'] for r in method_results]),
                'avg_latency_ms': np.mean([r['latency_ms'] for r in method_results]),
                'total_time_ms': total_time * 1000,
                'queries_per_second': len(test_queries) / total_time if total_time > 0 else 0
            }
        
        return results


# Example usage and testing functions
def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for testing"""
    return [
        {
            'id': 'doc_1',
            'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.',
            'metadata': {'topic': 'AI', 'length': 'short'}
        },
        {
            'id': 'doc_2', 
            'content': 'Deep learning neural networks have revolutionized computer vision and natural language processing tasks.',
            'metadata': {'topic': 'Deep Learning', 'length': 'medium'}
        },
        {
            'id': 'doc_3',
            'content': 'Transformers architecture introduced attention mechanisms that significantly improved language model performance.',
            'metadata': {'topic': 'NLP', 'length': 'medium'}
        },
        {
            'id': 'doc_4',
            'content': 'Retrieval-augmented generation combines information retrieval with language generation for better factual accuracy.',
            'metadata': {'topic': 'RAG', 'length': 'long'}
        },
        {
            'id': 'doc_5',
            'content': 'Vector databases store high-dimensional embeddings for efficient similarity search and retrieval.',
            'metadata': {'topic': 'Vector DB', 'length': 'short'}
        }
    ]


async def demo_advanced_retrieval():
    """Demonstration of advanced retrieval patterns"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create sample data
    documents = create_sample_documents()
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?", 
        "What are transformers in NLP?",
        "Explain retrieval augmented generation",
        "What is a vector database?"
    ]
    
    try:
        # Initialize advanced retrieval manager
        manager = AdvancedRetrievalManager(
            enable_colbert=True,
            enable_splade=True, 
            enable_dense=True
        )
        
        logger.info("Testing individual retrieval methods...")
        
        # Test individual methods
        for method in ['dense', 'hybrid']:  # Test available methods
            logger.info(f"\n--- Testing {method.upper()} Retrieval ---")
            
            for query in test_queries[:2]:  # Test first 2 queries
                results = manager.retrieve(query, documents, top_k=3, method=method)
                
                print(f"Query: {query}")
                for i, result in enumerate(results):
                    print(f"  {i+1}. [{result.retrieval_method}] Score: {result.score:.4f}")
                    print(f"     Content: {result.content[:100]}...")
                    print(f"     Latency: {result.latency_ms:.2f}ms")
                print()
        
        # Benchmark retrievers
        logger.info("\n--- Benchmarking Retrievers ---")
        benchmark_results = manager.benchmark_retrievers(test_queries, documents, top_k=5)
        
        for method, metrics in benchmark_results.items():
            print(f"{method.upper()}:")
            print(f"  Average Latency: {metrics['avg_latency_ms']:.2f}ms")
            print(f"  Average Score: {metrics['avg_score']:.4f}")
            print(f"  Queries/Second: {metrics['queries_per_second']:.2f}")
            print()
        
        # Performance metrics
        logger.info("\n--- Performance Metrics ---")
        perf_metrics = manager.get_performance_metrics()
        print(f"Active Retrievers: {list(perf_metrics['retrievers'].keys())}")
        print(f"Fusion Method: {perf_metrics['fusion_method']}")
        print(f"Weights: {perf_metrics['weights']}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demo_advanced_retrieval())