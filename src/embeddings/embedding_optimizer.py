"""
Embedding Optimization and Quantization System

This module provides advanced embedding optimization capabilities:
- Quantized embeddings for memory efficiency
- Domain-specific fine-tuning support  
- Multi-vector representations
- Embedding compression and indexing
- Performance monitoring and optimization
"""

import logging
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch and sentence-transformers not available. Some features will be disabled.")
    TORCH_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Using fallback similarity search.")
    FAISS_AVAILABLE = False


class QuantizationType(Enum):
    """Types of quantization methods"""
    NONE = "none"                    # No quantization (float32)
    INT8 = "int8"                   # 8-bit quantization
    INT4 = "int4"                   # 4-bit quantization
    BINARY = "binary"               # Binary quantization
    PRODUCT_QUANTIZATION = "pq"     # Product quantization


class EmbeddingType(Enum):
    """Types of embeddings for different purposes"""
    DENSE_GENERAL = "dense_general"     # General-purpose dense embeddings
    DENSE_DOMAIN = "dense_domain"       # Domain-specific dense embeddings
    SPARSE_KEYWORD = "sparse_keyword"   # Sparse keyword-based embeddings
    HYBRID = "hybrid"                   # Combination of dense and sparse
    MULTI_VECTOR = "multi_vector"       # Multiple vectors per document


@dataclass
class QuantizationConfig:
    """Configuration for embedding quantization"""
    quantization_type: QuantizationType
    bits_per_dimension: int = 8
    preserve_norm: bool = True
    centroids: int = 256  # For product quantization
    subvectors: int = 8   # For product quantization
    calibration_samples: int = 10000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding performance"""
    memory_usage_mb: float = 0.0
    compression_ratio: float = 1.0
    quantization_error: float = 0.0
    retrieval_accuracy: float = 1.0
    encoding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    throughput_docs_per_second: float = 0.0


class EmbeddingModel(Protocol):
    """Protocol for embedding models"""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class QuantizedEmbedding:
    """Quantized embedding representation"""
    
    def __init__(self, 
                 quantized_data: np.ndarray,
                 quantization_config: QuantizationConfig,
                 codebook: Optional[np.ndarray] = None,
                 scale: Optional[float] = None,
                 original_norm: Optional[float] = None):
        self.quantized_data = quantized_data
        self.config = quantization_config
        self.codebook = codebook
        self.scale = scale
        self.original_norm = original_norm
        self.creation_time = time.time()
    
    def dequantize(self) -> np.ndarray:
        """Convert quantized embedding back to float"""
        if self.config.quantization_type == QuantizationType.NONE:
            return self.quantized_data
        
        elif self.config.quantization_type == QuantizationType.INT8:
            # Simple linear dequantization
            dequantized = self.quantized_data.astype(np.float32)
            if self.scale:
                dequantized *= self.scale
            return dequantized
        
        elif self.config.quantization_type == QuantizationType.BINARY:
            # Binary to float conversion
            return np.where(self.quantized_data > 0, 1.0, -1.0).astype(np.float32)
        
        elif self.config.quantization_type == QuantizationType.PRODUCT_QUANTIZATION:
            # Product quantization dequantization
            if self.codebook is not None:
                return self._dequantize_product_quantization()
        
        return self.quantized_data.astype(np.float32)
    
    def _dequantize_product_quantization(self) -> np.ndarray:
        """Dequantize product quantized embeddings"""
        subvector_size = self.codebook.shape[1]
        reconstructed = np.zeros(len(self.quantized_data) * subvector_size, dtype=np.float32)
        
        for i, code in enumerate(self.quantized_data):
            start_idx = i * subvector_size
            end_idx = start_idx + subvector_size
            reconstructed[start_idx:end_idx] = self.codebook[code]
        
        return reconstructed
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        size_bytes = self.quantized_data.nbytes
        if self.codebook is not None:
            size_bytes += self.codebook.nbytes
        return size_bytes / (1024 * 1024)


class EmbeddingQuantizer:
    """Advanced embedding quantization with multiple strategies"""
    
    def __init__(self):
        self.calibration_data = {}
        self.quantization_stats = {}
    
    def quantize_embeddings(self, 
                          embeddings: np.ndarray,
                          config: QuantizationConfig,
                          calibration_embeddings: Optional[np.ndarray] = None) -> List[QuantizedEmbedding]:
        """Quantize embeddings using specified method"""
        
        start_time = time.time()
        quantized_embeddings = []
        
        if config.quantization_type == QuantizationType.NONE:
            # No quantization
            for embedding in embeddings:
                quantized_embeddings.append(QuantizedEmbedding(embedding, config))
        
        elif config.quantization_type == QuantizationType.INT8:
            quantized_embeddings = self._quantize_int8(embeddings, config)
        
        elif config.quantization_type == QuantizationType.BINARY:
            quantized_embeddings = self._quantize_binary(embeddings, config)
        
        elif config.quantization_type == QuantizationType.PRODUCT_QUANTIZATION:
            quantized_embeddings = self._quantize_product_quantization(
                embeddings, config, calibration_embeddings
            )
        
        # Calculate metrics
        quantization_time = (time.time() - start_time) * 1000
        self._update_quantization_stats(config.quantization_type, embeddings, quantized_embeddings, quantization_time)
        
        return quantized_embeddings
    
    def _quantize_int8(self, embeddings: np.ndarray, config: QuantizationConfig) -> List[QuantizedEmbedding]:
        """8-bit quantization with scale preservation"""
        quantized_embeddings = []
        
        for embedding in embeddings:
            # Calculate scale
            abs_max = np.max(np.abs(embedding))
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            
            # Quantize
            quantized = np.round(embedding / scale).astype(np.int8)
            
            # Preserve norm if requested
            original_norm = None
            if config.preserve_norm:
                original_norm = np.linalg.norm(embedding)
            
            quantized_embeddings.append(QuantizedEmbedding(
                quantized_data=quantized,
                quantization_config=config,
                scale=scale,
                original_norm=original_norm
            ))
        
        return quantized_embeddings
    
    def _quantize_binary(self, embeddings: np.ndarray, config: QuantizationConfig) -> List[QuantizedEmbedding]:
        """Binary quantization (1-bit per dimension)"""
        quantized_embeddings = []
        
        for embedding in embeddings:
            # Simple sign-based quantization
            quantized = np.sign(embedding).astype(np.int8)
            
            original_norm = None
            if config.preserve_norm:
                original_norm = np.linalg.norm(embedding)
            
            quantized_embeddings.append(QuantizedEmbedding(
                quantized_data=quantized,
                quantization_config=config,
                original_norm=original_norm
            ))
        
        return quantized_embeddings
    
    def _quantize_product_quantization(self, 
                                     embeddings: np.ndarray,
                                     config: QuantizationConfig,
                                     calibration_embeddings: Optional[np.ndarray] = None) -> List[QuantizedEmbedding]:
        """Product quantization for high compression"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available for product quantization. Using INT8 fallback.")
            config_fallback = QuantizationConfig(QuantizationType.INT8, preserve_norm=config.preserve_norm)
            return self._quantize_int8(embeddings, config_fallback)
        
        dimension = embeddings.shape[1]
        
        # Use calibration data or sample from input
        training_data = calibration_embeddings if calibration_embeddings is not None else embeddings
        
        # Create and train product quantizer
        pq = faiss.ProductQuantizer(dimension, config.subvectors, config.bits_per_dimension)
        pq.train(training_data.astype(np.float32))
        
        # Quantize embeddings
        quantized_embeddings = []
        codes = pq.compute_codes(embeddings.astype(np.float32))
        
        # Create codebook for dequantization
        codebook = faiss.vector_to_array(pq.centroids).reshape(
            config.subvectors * config.centroids, -1
        )
        
        for i, code in enumerate(codes):
            original_norm = None
            if config.preserve_norm:
                original_norm = np.linalg.norm(embeddings[i])
            
            quantized_embeddings.append(QuantizedEmbedding(
                quantized_data=code,
                quantization_config=config,
                codebook=codebook,
                original_norm=original_norm
            ))
        
        return quantized_embeddings
    
    def _update_quantization_stats(self, 
                                 quantization_type: QuantizationType,
                                 original_embeddings: np.ndarray,
                                 quantized_embeddings: List[QuantizedEmbedding],
                                 quantization_time_ms: float):
        """Update quantization statistics"""
        
        # Calculate compression ratio
        original_size = original_embeddings.nbytes
        quantized_size = sum(qe.get_memory_usage() for qe in quantized_embeddings) * 1024 * 1024
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        
        # Calculate quantization error (for a sample)
        sample_size = min(100, len(quantized_embeddings))
        sample_indices = np.random.choice(len(quantized_embeddings), sample_size, replace=False)
        
        total_error = 0.0
        for idx in sample_indices:
            original = original_embeddings[idx]
            dequantized = quantized_embeddings[idx].dequantize()
            
            # Cosine similarity error
            cos_sim = np.dot(original, dequantized) / (np.linalg.norm(original) * np.linalg.norm(dequantized))
            error = 1.0 - cos_sim
            total_error += error
        
        avg_error = total_error / sample_size
        
        # Store statistics
        self.quantization_stats[quantization_type.value] = {
            'compression_ratio': compression_ratio,
            'quantization_error': avg_error,
            'quantization_time_ms': quantization_time_ms,
            'memory_saved_mb': (original_size - quantized_size) / (1024 * 1024),
            'samples_processed': len(quantized_embeddings)
        }
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get quantization performance statistics"""
        return self.quantization_stats.copy()


class DomainSpecificEmbedding:
    """Domain-specific embedding fine-tuning support"""
    
    def __init__(self, base_model_name: str = "all-MiniLM-L6-v2"):
        self.base_model_name = base_model_name
        self.base_model = None
        self.fine_tuned_models = {}
        self.domain_vocabularies = {}
        
        if TORCH_AVAILABLE:
            self._initialize_base_model()
    
    def _initialize_base_model(self):
        """Initialize base sentence transformer model"""
        try:
            self.base_model = SentenceTransformer(self.base_model_name)
            logger.info(f"Initialized base model: {self.base_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize base model: {e}")
    
    def analyze_domain_vocabulary(self, domain_texts: List[str], domain_name: str) -> Dict[str, Any]:
        """Analyze domain-specific vocabulary and concepts"""
        
        # Simple vocabulary analysis
        word_freq = {}
        total_words = 0
        
        for text in domain_texts:
            words = text.lower().split()
            total_words += len(words)
            
            for word in words:
                # Simple word cleaning
                clean_word = ''.join(c for c in word if c.isalnum())
                if len(clean_word) > 2:
                    word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        # Find domain-specific terms (high frequency in domain, low in general)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        domain_terms = [word for word, freq in sorted_words[:100] if freq >= 3]
        
        vocab_analysis = {
            'domain_name': domain_name,
            'total_documents': len(domain_texts),
            'total_words': total_words,
            'unique_words': len(word_freq),
            'domain_specific_terms': domain_terms[:50],
            'avg_document_length': total_words / len(domain_texts) if domain_texts else 0,
            'vocabulary_density': len(word_freq) / total_words if total_words > 0 else 0
        }
        
        self.domain_vocabularies[domain_name] = vocab_analysis
        return vocab_analysis
    
    def suggest_fine_tuning_strategy(self, domain_name: str) -> Dict[str, Any]:
        """Suggest fine-tuning strategy based on domain analysis"""
        
        if domain_name not in self.domain_vocabularies:
            return {'error': 'Domain not analyzed'}
        
        vocab_analysis = self.domain_vocabularies[domain_name]
        
        # Determine strategy based on domain characteristics
        strategy = {
            'recommended_approach': 'continued_pretraining',
            'reasoning': [],
            'hyperparameters': {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'epochs': 3,
                'warmup_steps': 100
            },
            'data_requirements': {
                'minimum_samples': 1000,
                'recommended_samples': 5000,
                'current_samples': vocab_analysis['total_documents']
            }
        }
        
        # Adjust strategy based on domain characteristics
        vocab_density = vocab_analysis['vocabulary_density']
        doc_count = vocab_analysis['total_documents']
        
        if vocab_density > 0.1:  # High vocabulary diversity
            strategy['recommended_approach'] = 'domain_adaptive_pretraining'
            strategy['reasoning'].append("High vocabulary diversity suggests domain adaptation needed")
        
        if doc_count < 1000:
            strategy['recommended_approach'] = 'few_shot_learning'
            strategy['reasoning'].append("Limited data suggests few-shot approach")
            strategy['hyperparameters']['learning_rate'] = 1e-5  # Lower learning rate
        
        return strategy
    
    def create_domain_embeddings(self, texts: List[str], domain_name: str) -> np.ndarray:
        """Create embeddings optimized for specific domain"""
        
        if not self.base_model:
            logger.error("Base model not available")
            return np.array([])
        
        # Use base model for now (fine-tuning would require training data)
        embeddings = self.base_model.encode(texts, show_progress_bar=True)
        
        # Apply domain-specific post-processing if available
        if domain_name in self.domain_vocabularies:
            embeddings = self._apply_domain_weighting(embeddings, texts, domain_name)
        
        return embeddings
    
    def _apply_domain_weighting(self, embeddings: np.ndarray, texts: List[str], domain_name: str) -> np.ndarray:
        """Apply domain-specific weighting to embeddings"""
        
        vocab_analysis = self.domain_vocabularies[domain_name]
        domain_terms = set(vocab_analysis['domain_specific_terms'])
        
        # Simple domain weighting based on term presence
        weighted_embeddings = embeddings.copy()
        
        for i, text in enumerate(texts):
            words = set(text.lower().split())
            domain_overlap = len(words & domain_terms)
            
            if domain_overlap > 0:
                # Boost embeddings for domain-rich texts
                boost_factor = 1.0 + (domain_overlap / len(words)) * 0.1
                weighted_embeddings[i] *= boost_factor
        
        return weighted_embeddings


class MultiVectorEmbedding:
    """Multi-vector representation system for comprehensive document understanding"""
    
    def __init__(self):
        self.vector_types = {
            'semantic': None,      # Semantic understanding
            'keyword': None,       # Keyword matching
            'structural': None,    # Document structure
            'temporal': None       # Temporal aspects
        }
        self.combination_weights = {
            'semantic': 0.6,
            'keyword': 0.2,
            'structural': 0.1,
            'temporal': 0.1
        }
    
    def create_multi_vector_representation(self, 
                                         text: str, 
                                         metadata: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """Create multiple vector representations of text"""
        
        vectors = {}
        
        # Semantic vector (using sentence transformer)
        if TORCH_AVAILABLE:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                vectors['semantic'] = model.encode([text])[0]
            except Exception as e:
                logger.warning(f"Failed to create semantic vector: {e}")
                vectors['semantic'] = np.random.random(384).astype(np.float32)  # Placeholder
        else:
            vectors['semantic'] = np.random.random(384).astype(np.float32)
        
        # Keyword vector (TF-IDF style)
        vectors['keyword'] = self._create_keyword_vector(text)
        
        # Structural vector (document structure features)
        vectors['structural'] = self._create_structural_vector(text, metadata)
        
        # Temporal vector (time-based features if available)
        vectors['temporal'] = self._create_temporal_vector(text, metadata)
        
        return vectors
    
    def _create_keyword_vector(self, text: str, vocab_size: int = 1000) -> np.ndarray:
        """Create keyword-based sparse vector representation"""
        
        # Simple keyword extraction and hashing
        words = text.lower().split()
        word_counts = {}
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2:
                word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        # Create sparse vector using hashing
        vector = np.zeros(vocab_size, dtype=np.float32)
        
        for word, count in word_counts.items():
            hash_idx = hash(word) % vocab_size
            vector[hash_idx] = count / len(words)  # Normalized frequency
        
        return vector
    
    def _create_structural_vector(self, text: str, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Create vector representing document structure"""
        
        features = []
        
        # Text-based structural features
        features.extend([
            len(text),                    # Document length
            len(text.split()),           # Word count
            len([s for s in text.split('.') if s.strip()]),  # Sentence count
            text.count('\n'),            # Paragraph breaks
            text.count('?'),             # Questions
            text.count('!'),             # Exclamations
            len([w for w in text.split() if w.isupper()]),  # All caps words
            len([w for w in text.split() if w.isdigit()]),  # Numbers
        ])
        
        # Metadata-based features
        if metadata:
            features.extend([
                len(metadata),                           # Metadata richness
                1.0 if 'title' in metadata else 0.0,   # Has title
                1.0 if 'author' in metadata else 0.0,  # Has author
                1.0 if 'date' in metadata else 0.0,    # Has date
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Normalize features
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to fixed size
        target_size = 32
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
    
    def _create_temporal_vector(self, text: str, metadata: Dict[str, Any] = None) -> np.ndarray:
        """Create vector representing temporal aspects"""
        
        features = []
        
        # Temporal indicators in text
        temporal_words = ['yesterday', 'today', 'tomorrow', 'recent', 'current', 'future', 'past']
        temporal_count = sum(1 for word in temporal_words if word in text.lower())
        features.append(temporal_count / len(text.split()) if text.split() else 0)
        
        # Date references
        date_pattern_count = len(__import__('re').findall(r'\d{4}|\d{1,2}/\d{1,2}|\d{1,2}-\d{1,2}', text))
        features.append(date_pattern_count / len(text.split()) if text.split() else 0)
        
        # Metadata temporal features
        if metadata and 'timestamp' in metadata:
            try:
                from datetime import datetime
                doc_time = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                now = datetime.now()
                age_days = (now - doc_time.replace(tzinfo=None)).days
                features.append(1.0 / (1.0 + age_days / 365))  # Recency score
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Time-based language patterns
        time_phrases = ['in the future', 'in the past', 'currently', 'previously', 'recently']
        time_phrase_count = sum(1 for phrase in time_phrases if phrase in text.lower())
        features.append(time_phrase_count / len(text.split()) if text.split() else 0)
        
        # Pad to fixed size
        target_size = 16
        features = np.array(features, dtype=np.float32)
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
    
    def combine_vectors(self, multi_vectors: Dict[str, np.ndarray], 
                       custom_weights: Dict[str, float] = None) -> np.ndarray:
        """Combine multiple vectors into single representation"""
        
        weights = custom_weights if custom_weights else self.combination_weights
        combined_parts = []
        
        for vector_type, weight in weights.items():
            if vector_type in multi_vectors:
                vector = multi_vectors[vector_type]
                weighted_vector = vector * weight
                combined_parts.append(weighted_vector)
        
        if combined_parts:
            # Concatenate all weighted vectors
            return np.concatenate(combined_parts)
        else:
            return np.array([])
    
    def search_with_multi_vectors(self, 
                                query_vectors: Dict[str, np.ndarray],
                                document_vectors: List[Dict[str, np.ndarray]],
                                vector_weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """Search using multi-vector representations"""
        
        weights = vector_weights if vector_weights else self.combination_weights
        results = []
        
        for doc_idx, doc_vectors in enumerate(document_vectors):
            total_similarity = 0.0
            total_weight = 0.0
            
            for vector_type, weight in weights.items():
                if vector_type in query_vectors and vector_type in doc_vectors:
                    query_vec = query_vectors[vector_type]
                    doc_vec = doc_vectors[vector_type]
                    
                    # Cosine similarity
                    similarity = np.dot(query_vec, doc_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                    )
                    
                    total_similarity += similarity * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_similarity = total_similarity / total_weight
                results.append((doc_idx, avg_similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class EmbeddingOptimizer:
    """Main embedding optimization manager"""
    
    def __init__(self, cache_manager=None):
        self.quantizer = EmbeddingQuantizer()
        self.domain_embedding = DomainSpecificEmbedding()
        self.multi_vector = MultiVectorEmbedding()
        self.cache_manager = cache_manager
        self.optimization_stats = {}
        
        # Performance tracking
        self.encoding_times = []
        self.memory_usage_history = []
    
    def optimize_embedding_pipeline(self, 
                                  texts: List[str],
                                  optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize entire embedding pipeline"""
        
        start_time = time.time()
        results = {
            'original_texts_count': len(texts),
            'optimizations_applied': [],
            'performance_metrics': {},
            'embeddings': {}
        }
        
        # Domain analysis if requested
        if optimization_config.get('analyze_domain'):
            domain_name = optimization_config.get('domain_name', 'default')
            domain_analysis = self.domain_embedding.analyze_domain_vocabulary(texts, domain_name)
            results['domain_analysis'] = domain_analysis
            results['optimizations_applied'].append('domain_analysis')
        
        # Create base embeddings
        if optimization_config.get('embedding_type', 'dense_general') == 'multi_vector':
            # Multi-vector embeddings
            multi_embeddings = []
            for text in texts:
                multi_vec = self.multi_vector.create_multi_vector_representation(text)
                multi_embeddings.append(multi_vec)
            results['embeddings']['multi_vector'] = multi_embeddings
            results['optimizations_applied'].append('multi_vector')
        
        else:
            # Standard dense embeddings
            domain_name = optimization_config.get('domain_name')
            if domain_name:
                embeddings = self.domain_embedding.create_domain_embeddings(texts, domain_name)
            else:
                # Use base sentence transformer
                if TORCH_AVAILABLE:
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    embeddings = model.encode(texts, show_progress_bar=True)
                else:
                    # Fallback to random embeddings
                    embeddings = np.random.random((len(texts), 384)).astype(np.float32)
            
            results['embeddings']['dense'] = embeddings
        
        # Apply quantization if requested
        if optimization_config.get('quantization'):
            quantization_config = QuantizationConfig(**optimization_config['quantization'])
            
            if 'dense' in results['embeddings']:
                quantized_embeddings = self.quantizer.quantize_embeddings(
                    results['embeddings']['dense'], quantization_config
                )
                results['embeddings']['quantized'] = quantized_embeddings
                results['optimizations_applied'].append('quantization')
                
                # Add quantization metrics
                results['quantization_stats'] = self.quantizer.get_quantization_stats()
        
        # Cache embeddings if cache manager available
        if self.cache_manager and optimization_config.get('enable_caching', True):
            for i, text in enumerate(texts):
                if 'dense' in results['embeddings']:
                    self.cache_manager.cache_embedding(
                        text, results['embeddings']['dense'][i]
                    )
            results['optimizations_applied'].append('caching')
        
        # Calculate performance metrics
        total_time = (time.time() - start_time) * 1000
        results['performance_metrics'] = {
            'total_optimization_time_ms': total_time,
            'texts_per_second': len(texts) / (total_time / 1000) if total_time > 0 else 0,
            'optimizations_count': len(results['optimizations_applied'])
        }
        
        if 'dense' in results['embeddings']:
            embeddings_array = results['embeddings']['dense']
            results['performance_metrics'].update({
                'embedding_dimension': embeddings_array.shape[1],
                'memory_usage_mb': embeddings_array.nbytes / (1024 * 1024),
                'average_embedding_norm': np.mean([np.linalg.norm(emb) for emb in embeddings_array])
            })
        
        # Update optimization statistics
        self._update_optimization_stats(results)
        
        return results
    
    def benchmark_quantization_methods(self, 
                                     sample_embeddings: np.ndarray,
                                     methods: List[QuantizationType] = None) -> Dict[str, Any]:
        """Benchmark different quantization methods"""
        
        if methods is None:
            methods = [QuantizationType.NONE, QuantizationType.INT8, QuantizationType.BINARY]
        
        benchmark_results = {}
        
        for method in methods:
            config = QuantizationConfig(quantization_type=method)
            
            start_time = time.time()
            quantized = self.quantizer.quantize_embeddings(sample_embeddings, config)
            quantization_time = (time.time() - start_time) * 1000
            
            # Calculate metrics
            original_size = sample_embeddings.nbytes
            quantized_size = sum(qe.get_memory_usage() for qe in quantized) * 1024 * 1024
            
            # Quality assessment (sample)
            sample_idx = 0
            original_emb = sample_embeddings[sample_idx]
            dequantized_emb = quantized[sample_idx].dequantize()
            
            cos_similarity = np.dot(original_emb, dequantized_emb) / (
                np.linalg.norm(original_emb) * np.linalg.norm(dequantized_emb)
            )
            
            benchmark_results[method.value] = {
                'quantization_time_ms': quantization_time,
                'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1.0,
                'memory_saved_mb': (original_size - quantized_size) / (1024 * 1024),
                'quality_score': cos_similarity,
                'throughput_embeddings_per_second': len(sample_embeddings) / (quantization_time / 1000) if quantization_time > 0 else 0
            }
        
        return benchmark_results
    
    def _update_optimization_stats(self, results: Dict[str, Any]):
        """Update optimization statistics"""
        
        optimization_key = '_'.join(sorted(results['optimizations_applied']))
        
        if optimization_key not in self.optimization_stats:
            self.optimization_stats[optimization_key] = {
                'usage_count': 0,
                'total_time_ms': 0,
                'avg_time_ms': 0,
                'texts_processed': 0
            }
        
        stats = self.optimization_stats[optimization_key]
        stats['usage_count'] += 1
        stats['total_time_ms'] += results['performance_metrics']['total_optimization_time_ms']
        stats['texts_processed'] += results['original_texts_count']
        stats['avg_time_ms'] = stats['total_time_ms'] / stats['usage_count']
    
    def get_optimization_recommendations(self, 
                                       workload_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization recommendations based on workload"""
        
        recommendations = {
            'quantization': None,
            'embedding_type': 'dense_general',
            'caching_strategy': 'aggressive',
            'reasoning': []
        }
        
        # Analyze workload
        doc_count = workload_characteristics.get('document_count', 0)
        query_frequency = workload_characteristics.get('queries_per_second', 0)
        memory_constraint_mb = workload_characteristics.get('memory_limit_mb', 0)
        accuracy_requirement = workload_characteristics.get('accuracy_threshold', 0.9)
        
        # Memory-constrained environments
        if memory_constraint_mb > 0 and memory_constraint_mb < 1024:  # Less than 1GB
            if accuracy_requirement < 0.85:
                recommendations['quantization'] = {
                    'quantization_type': 'binary',
                    'preserve_norm': True
                }
                recommendations['reasoning'].append("Memory-constrained with moderate accuracy requirements suggests binary quantization")
            else:
                recommendations['quantization'] = {
                    'quantization_type': 'int8',
                    'preserve_norm': True
                }
                recommendations['reasoning'].append("Memory-constrained with high accuracy requirements suggests INT8 quantization")
        
        # High-throughput scenarios
        if query_frequency > 100:  # High query volume
            recommendations['caching_strategy'] = 'aggressive'
            recommendations['embedding_type'] = 'dense_general'  # Faster than multi-vector
            recommendations['reasoning'].append("High query frequency benefits from aggressive caching and simple embeddings")
        
        # Large document collections
        if doc_count > 100000:
            if not recommendations['quantization']:
                recommendations['quantization'] = {
                    'quantization_type': 'int8',
                    'preserve_norm': True
                }
            recommendations['reasoning'].append("Large document collection benefits from quantization")
        
        # Domain-specific workloads
        if workload_characteristics.get('domain_specific', False):
            recommendations['embedding_type'] = 'dense_domain'
            recommendations['reasoning'].append("Domain-specific content benefits from specialized embeddings")
        
        return recommendations
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        
        return {
            'optimization_stats': self.optimization_stats,
            'quantization_stats': self.quantizer.get_quantization_stats(),
            'domain_vocabularies': self.domain_embedding.domain_vocabularies,
            'performance_tracking': {
                'total_optimizations': sum(stats['usage_count'] for stats in self.optimization_stats.values()),
                'total_texts_processed': sum(stats['texts_processed'] for stats in self.optimization_stats.values()),
                'average_optimization_time_ms': np.mean([stats['avg_time_ms'] for stats in self.optimization_stats.values()]) if self.optimization_stats else 0
            }
        }