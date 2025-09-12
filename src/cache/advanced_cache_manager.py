"""
Advanced Caching Strategy with Redis

This module provides comprehensive caching capabilities for RAG systems:
- Embedding cache with persistence
- Query result cache with TTL
- Intelligent cache invalidation
- Multi-level cache hierarchy
- Performance monitoring and statistics
"""

import logging
import json
import pickle
import hashlib
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import threading
from collections import defaultdict, OrderedDict
import asyncio

logger = logging.getLogger(__name__)

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis not available. Using in-memory cache fallback.")
    REDIS_AVAILABLE = False


class CacheLevel(Enum):
    """Cache levels for hierarchical caching"""
    MEMORY = "memory"           # L1: Fast in-memory cache
    REDIS = "redis"             # L2: Redis persistent cache
    DISK = "disk"              # L3: Disk-based cache (future)


class CacheType(Enum):
    """Types of cached content"""
    EMBEDDING = "embedding"
    QUERY_RESULT = "query_result"
    DOCUMENT_CHUNK = "document_chunk"
    RERANKING_SCORE = "reranking_score"
    ENTITY_EXTRACTION = "entity_extraction"
    GRAPH_RELATIONSHIP = "graph_relationship"
    CONTEXT_WINDOW = "context_window"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_size_bytes: int = 0
    entries_by_type: Dict[str, int] = field(default_factory=dict)
    average_access_time_ms: float = 0.0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    redis_usage_mb: float = 0.0
    
    def update_hit_rate(self):
        """Update cache hit rate"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests
        else:
            self.hit_rate = 0.0


class MemoryCache:
    """High-performance in-memory LRU cache"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_size_bytes = 0
        self.statistics = CacheStatistics()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from memory cache"""
        with self.lock:
            self.statistics.total_requests += 1
            start_time = time.time()
            
            entry = self.cache.get(key)
            if entry and not entry.is_expired():
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.update_access()
                self.statistics.cache_hits += 1
                
                access_time = (time.time() - start_time) * 1000
                self._update_average_access_time(access_time)
                
                return entry
            
            elif entry and entry.is_expired():
                # Remove expired entry
                self._remove_entry(key)
            
            self.statistics.cache_misses += 1
            return None
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put entry into memory cache"""
        with self.lock:
            # Estimate size
            entry.size_bytes = self._estimate_size(entry.value)
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Check if we need to evict entries
            self._ensure_space(entry.size_bytes)
            
            # Add new entry
            self.cache[key] = entry
            self.current_size_bytes += entry.size_bytes
            
            # Update statistics
            cache_type_str = entry.cache_type.value
            self.statistics.entries_by_type[cache_type_str] = self.statistics.entries_by_type.get(cache_type_str, 0) + 1
            self.statistics.total_size_bytes = self.current_size_bytes
            self.statistics.memory_usage_mb = self.current_size_bytes / (1024 * 1024)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from memory cache"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all entries"""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.statistics = CacheStatistics()
    
    def _remove_entry(self, key: str):
        """Remove entry and update statistics"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size_bytes -= entry.size_bytes
            
            cache_type_str = entry.cache_type.value
            self.statistics.entries_by_type[cache_type_str] = max(0, self.statistics.entries_by_type.get(cache_type_str, 1) - 1)
    
    def _ensure_space(self, needed_bytes: int):
        """Ensure sufficient space by evicting old entries"""
        # Evict by size constraint
        while (self.current_size_bytes + needed_bytes > self.max_memory_bytes and 
               len(self.cache) > 0):
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)
        
        # Evict by count constraint
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                # Rough estimate using pickle
                return len(pickle.dumps(value))
        except Exception:
            # Fallback estimate
            return 1024
    
    def _update_average_access_time(self, access_time_ms: float):
        """Update running average access time"""
        if self.statistics.average_access_time_ms == 0:
            self.statistics.average_access_time_ms = access_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.statistics.average_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.statistics.average_access_time_ms
            )
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics"""
        with self.lock:
            stats = self.statistics
            stats.update_hit_rate()
            return stats


class RedisCache:
    """Redis-based persistent cache with async support"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 db: int = 0,
                 key_prefix: str = "rag_cache:",
                 max_connections: int = 20):
        self.redis_url = redis_url
        self.db = db
        self.key_prefix = key_prefix
        self.max_connections = max_connections
        
        self.redis_client = None
        self.async_redis_client = None
        self.statistics = CacheStatistics()
        self.connection_pool = None
        
        if REDIS_AVAILABLE:
            self._initialize_redis()
        else:
            logger.warning("Redis not available, using memory-only fallback")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                db=self.db,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_timeout=5.0
            )
            
            # Synchronous client
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None
    
    async def _get_async_client(self):
        """Get async Redis client"""
        if not self.async_redis_client:
            try:
                self.async_redis_client = aioredis.from_url(
                    self.redis_url,
                    db=self.db,
                    max_connections=self.max_connections
                )
                await self.async_redis_client.ping()
            except Exception as e:
                logger.error(f"Failed to initialize async Redis client: {e}")
                return None
        
        return self.async_redis_client
    
    def _make_key(self, key: str, cache_type: CacheType) -> str:
        """Create Redis key with prefix and type"""
        return f"{self.key_prefix}{cache_type.value}:{key}"
    
    def get(self, key: str, cache_type: CacheType) -> Optional[CacheEntry]:
        """Get entry from Redis cache"""
        if not self.redis_client:
            return None
        
        self.statistics.total_requests += 1
        start_time = time.time()
        
        try:
            redis_key = self._make_key(key, cache_type)
            
            # Get serialized entry
            serialized_data = self.redis_client.get(redis_key)
            if not serialized_data:
                self.statistics.cache_misses += 1
                return None
            
            # Deserialize entry
            entry_data = pickle.loads(serialized_data)
            entry = CacheEntry(**entry_data)
            
            # Check if expired
            if entry.is_expired():
                self.delete(key, cache_type)
                self.statistics.cache_misses += 1
                return None
            
            # Update access info
            entry.update_access()
            
            # Update in Redis (for access tracking)
            self._update_entry_async(redis_key, entry)
            
            self.statistics.cache_hits += 1
            
            access_time = (time.time() - start_time) * 1000
            self._update_average_access_time(access_time)
            
            return entry
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self.statistics.cache_misses += 1
            return None
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put entry into Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._make_key(key, entry.cache_type)
            
            # Serialize entry
            entry_data = asdict(entry)
            # Handle datetime serialization
            entry_data['created_at'] = entry_data['created_at'].isoformat()
            entry_data['last_accessed'] = entry_data['last_accessed'].isoformat()
            
            serialized_data = pickle.dumps(entry_data)
            
            # Set in Redis with optional TTL
            if entry.ttl_seconds:
                self.redis_client.setex(redis_key, entry.ttl_seconds, serialized_data)
            else:
                self.redis_client.set(redis_key, serialized_data)
            
            # Update statistics
            cache_type_str = entry.cache_type.value
            self.statistics.entries_by_type[cache_type_str] = self.statistics.entries_by_type.get(cache_type_str, 0) + 1
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache put error: {e}")
            return False
    
    def delete(self, key: str, cache_type: CacheType) -> bool:
        """Delete entry from Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._make_key(key, cache_type)
            result = self.redis_client.delete(redis_key)
            
            if result > 0:
                cache_type_str = cache_type.value
                self.statistics.entries_by_type[cache_type_str] = max(0, self.statistics.entries_by_type.get(cache_type_str, 1) - 1)
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    def clear_type(self, cache_type: CacheType) -> int:
        """Clear all entries of a specific type"""
        if not self.redis_client:
            return 0
        
        try:
            pattern = f"{self.key_prefix}{cache_type.value}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                self.statistics.entries_by_type[cache_type.value] = 0
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return 0
    
    def _update_entry_async(self, redis_key: str, entry: CacheEntry):
        """Update entry access info asynchronously"""
        try:
            # Run in background thread to avoid blocking
            def update_background():
                try:
                    entry_data = asdict(entry)
                    entry_data['created_at'] = entry_data['created_at'].isoformat()
                    entry_data['last_accessed'] = entry_data['last_accessed'].isoformat()
                    serialized_data = pickle.dumps(entry_data)
                    
                    if entry.ttl_seconds:
                        self.redis_client.setex(redis_key, entry.ttl_seconds, serialized_data)
                    else:
                        self.redis_client.set(redis_key, serialized_data)
                except Exception as e:
                    logger.debug(f"Background entry update failed: {e}")
            
            # Start background thread
            thread = threading.Thread(target=update_background)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.debug(f"Failed to start background update: {e}")
    
    def _update_average_access_time(self, access_time_ms: float):
        """Update running average access time"""
        if self.statistics.average_access_time_ms == 0:
            self.statistics.average_access_time_ms = access_time_ms
        else:
            alpha = 0.1
            self.statistics.average_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.statistics.average_access_time_ms
            )
    
    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        if not self.redis_client:
            return {}
        
        try:
            info = self.redis_client.info()
            return {
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics"""
        stats = self.statistics
        stats.update_hit_rate()
        
        # Add Redis-specific stats
        redis_info = self.get_redis_info()
        stats.redis_usage_mb = redis_info.get('used_memory_mb', 0)
        
        return stats


class CacheInvalidationManager:
    """Manages intelligent cache invalidation based on content changes"""
    
    def __init__(self):
        self.content_fingerprints = {}  # content_id -> hash
        self.dependency_graph = defaultdict(set)  # dependency relationships
        self.invalidation_rules = {}
        self.lock = threading.RLock()
    
    def register_content_fingerprint(self, content_id: str, content_hash: str):
        """Register content fingerprint for change detection"""
        with self.lock:
            self.content_fingerprints[content_id] = content_hash
    
    def add_dependency(self, dependent_key: str, dependency_content_id: str):
        """Add cache dependency relationship"""
        with self.lock:
            self.dependency_graph[dependency_content_id].add(dependent_key)
    
    def check_and_invalidate(self, content_id: str, new_content_hash: str, 
                           cache_manager: 'AdvancedCacheManager') -> List[str]:
        """Check for content changes and invalidate dependent caches"""
        with self.lock:
            old_hash = self.content_fingerprints.get(content_id)
            
            if old_hash is None:
                # New content
                self.content_fingerprints[content_id] = new_content_hash
                return []
            
            if old_hash == new_content_hash:
                # No change
                return []
            
            # Content changed - invalidate dependents
            invalidated_keys = []
            dependent_keys = self.dependency_graph.get(content_id, set())
            
            for cache_key in dependent_keys:
                # Determine cache type from key pattern
                cache_type = self._infer_cache_type(cache_key)
                if cache_manager.delete(cache_key, cache_type):
                    invalidated_keys.append(cache_key)
            
            # Update fingerprint
            self.content_fingerprints[content_id] = new_content_hash
            
            return invalidated_keys
    
    def _infer_cache_type(self, cache_key: str) -> CacheType:
        """Infer cache type from key pattern"""
        if 'embedding' in cache_key:
            return CacheType.EMBEDDING
        elif 'query' in cache_key:
            return CacheType.QUERY_RESULT
        elif 'rerank' in cache_key:
            return CacheType.RERANKING_SCORE
        elif 'entity' in cache_key:
            return CacheType.ENTITY_EXTRACTION
        elif 'context' in cache_key:
            return CacheType.CONTEXT_WINDOW
        else:
            return CacheType.DOCUMENT_CHUNK
    
    def clear_dependencies(self, content_id: str):
        """Clear all dependencies for a content ID"""
        with self.lock:
            if content_id in self.dependency_graph:
                del self.dependency_graph[content_id]
            if content_id in self.content_fingerprints:
                del self.content_fingerprints[content_id]


class AdvancedCacheManager:
    """Advanced multi-level cache manager with intelligent strategies"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 memory_cache_size: int = 1000,
                 memory_cache_mb: int = 512,
                 enable_redis: bool = True):
        
        # Initialize cache levels
        self.memory_cache = MemoryCache(memory_cache_size, memory_cache_mb)
        self.redis_cache = RedisCache(redis_url) if enable_redis and REDIS_AVAILABLE else None
        self.invalidation_manager = CacheInvalidationManager()
        
        # Cache configuration by type
        self.cache_configs = {
            CacheType.EMBEDDING: {
                'memory_enabled': True,
                'redis_enabled': True,
                'ttl_seconds': 86400 * 7,  # 7 days
                'priority': 'high'
            },
            CacheType.QUERY_RESULT: {
                'memory_enabled': True,
                'redis_enabled': True,
                'ttl_seconds': 3600,  # 1 hour
                'priority': 'medium'
            },
            CacheType.DOCUMENT_CHUNK: {
                'memory_enabled': False,
                'redis_enabled': True,
                'ttl_seconds': 86400,  # 1 day
                'priority': 'low'
            },
            CacheType.RERANKING_SCORE: {
                'memory_enabled': True,
                'redis_enabled': True,
                'ttl_seconds': 3600 * 6,  # 6 hours
                'priority': 'medium'
            },
            CacheType.ENTITY_EXTRACTION: {
                'memory_enabled': True,
                'redis_enabled': True,
                'ttl_seconds': 86400 * 3,  # 3 days
                'priority': 'medium'
            },
            CacheType.CONTEXT_WINDOW: {
                'memory_enabled': True,
                'redis_enabled': False,  # Too large for Redis
                'ttl_seconds': 1800,  # 30 minutes
                'priority': 'high'
            }
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_operations': 0,
            'cache_effectiveness': 0.0,
            'average_retrieval_time_ms': 0.0
        }
    
    def get(self, key: str, cache_type: CacheType, 
           compute_function: Optional[callable] = None) -> Tuple[Any, bool]:
        """
        Get value from cache with multi-level fallback
        
        Returns: (value, cache_hit)
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        config = self.cache_configs.get(cache_type, {})
        cache_key = self._create_cache_key(key, cache_type)
        
        # L1: Memory cache
        if config.get('memory_enabled', True):
            entry = self.memory_cache.get(cache_key)
            if entry:
                self._update_performance_stats(time.time() - start_time, True)
                return entry.value, True
        
        # L2: Redis cache
        if config.get('redis_enabled', True) and self.redis_cache:
            entry = self.redis_cache.get(cache_key, cache_type)
            if entry:
                # Promote to memory cache if enabled
                if config.get('memory_enabled', True):
                    self.memory_cache.put(cache_key, entry)
                
                self._update_performance_stats(time.time() - start_time, True)
                return entry.value, True
        
        # Cache miss - compute value if function provided
        if compute_function:
            try:
                computed_value = compute_function()
                self.put(key, computed_value, cache_type)
                self._update_performance_stats(time.time() - start_time, False)
                return computed_value, False
            except Exception as e:
                logger.error(f"Error computing cached value: {e}")
        
        self._update_performance_stats(time.time() - start_time, False)
        return None, False
    
    def put(self, key: str, value: Any, cache_type: CacheType, 
           content_id: Optional[str] = None) -> bool:
        """Put value into appropriate cache levels"""
        config = self.cache_configs.get(cache_type, {})
        cache_key = self._create_cache_key(key, cache_type)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            cache_type=cache_type,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=config.get('ttl_seconds'),
            metadata={
                'content_id': content_id,
                'priority': config.get('priority', 'medium')
            }
        )
        
        success = False
        
        # Store in memory cache
        if config.get('memory_enabled', True):
            success |= self.memory_cache.put(cache_key, entry)
        
        # Store in Redis cache
        if config.get('redis_enabled', True) and self.redis_cache:
            success |= self.redis_cache.put(cache_key, entry)
        
        # Register dependencies for invalidation
        if content_id:
            self.invalidation_manager.add_dependency(cache_key, content_id)
        
        return success
    
    def delete(self, key: str, cache_type: CacheType) -> bool:
        """Delete from all cache levels"""
        cache_key = self._create_cache_key(key, cache_type)
        
        success_memory = self.memory_cache.delete(cache_key)
        success_redis = True
        
        if self.redis_cache:
            success_redis = self.redis_cache.delete(cache_key, cache_type)
        
        return success_memory or success_redis
    
    def invalidate_by_content(self, content_id: str, new_content_hash: str) -> List[str]:
        """Invalidate caches based on content changes"""
        return self.invalidation_manager.check_and_invalidate(
            content_id, new_content_hash, self
        )
    
    def clear_type(self, cache_type: CacheType):
        """Clear all caches of a specific type"""
        # Clear memory cache (partial - need to filter by type)
        # This is a simplified implementation
        
        # Clear Redis cache
        if self.redis_cache:
            self.redis_cache.clear_type(cache_type)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'performance': self.performance_stats.copy(),
            'memory_cache': asdict(self.memory_cache.get_statistics()),
            'cache_levels_available': ['memory']
        }
        
        if self.redis_cache:
            stats['redis_cache'] = asdict(self.redis_cache.get_statistics())
            stats['redis_info'] = self.redis_cache.get_redis_info()
            stats['cache_levels_available'].append('redis')
        
        # Calculate overall effectiveness
        total_requests = stats['memory_cache']['total_requests']
        if self.redis_cache:
            total_requests += stats['redis_cache']['total_requests']
        
        total_hits = stats['memory_cache']['cache_hits']
        if self.redis_cache:
            total_hits += stats['redis_cache']['cache_hits']
        
        if total_requests > 0:
            stats['overall_hit_rate'] = total_hits / total_requests
        else:
            stats['overall_hit_rate'] = 0.0
        
        return stats
    
    def _create_cache_key(self, key: str, cache_type: CacheType) -> str:
        """Create standardized cache key"""
        # Create hash for very long keys
        if len(key) > 200:
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            return f"{cache_type.value}:{key_hash}"
        else:
            # Clean key for cache storage
            clean_key = re.sub(r'[^\w\-_.]', '_', key)
            return f"{cache_type.value}:{clean_key}"
    
    def _update_performance_stats(self, retrieval_time: float, cache_hit: bool):
        """Update performance statistics"""
        retrieval_time_ms = retrieval_time * 1000
        
        # Update average retrieval time
        if self.performance_stats['average_retrieval_time_ms'] == 0:
            self.performance_stats['average_retrieval_time_ms'] = retrieval_time_ms
        else:
            alpha = 0.1
            self.performance_stats['average_retrieval_time_ms'] = (
                alpha * retrieval_time_ms + 
                (1 - alpha) * self.performance_stats['average_retrieval_time_ms']
            )
        
        # Update cache effectiveness
        total_ops = self.performance_stats['total_operations']
        if total_ops > 0:
            # Simple running average of cache effectiveness
            if cache_hit:
                self.performance_stats['cache_effectiveness'] = (
                    self.performance_stats['cache_effectiveness'] * (total_ops - 1) + 1.0
                ) / total_ops
            else:
                self.performance_stats['cache_effectiveness'] = (
                    self.performance_stats['cache_effectiveness'] * (total_ops - 1)
                ) / total_ops
    
    # Convenience methods for specific cache types
    
    def cache_embedding(self, text: str, embedding: np.ndarray, model_name: str = "default") -> bool:
        """Cache embedding with content-based key"""
        key = f"{model_name}:{hashlib.sha256(text.encode()).hexdigest()}"
        return self.put(key, embedding, CacheType.EMBEDDING, content_id=text)
    
    def get_embedding(self, text: str, model_name: str = "default", 
                     compute_function: Optional[callable] = None) -> Tuple[Optional[np.ndarray], bool]:
        """Get cached embedding"""
        key = f"{model_name}:{hashlib.sha256(text.encode()).hexdigest()}"
        return self.get(key, CacheType.EMBEDDING, compute_function)
    
    def cache_query_result(self, query: str, results: List[Dict[str, Any]], 
                         search_params: Dict[str, Any] = None) -> bool:
        """Cache query results"""
        params_str = json.dumps(search_params or {}, sort_keys=True)
        key = f"{query}:{hashlib.sha256(params_str.encode()).hexdigest()}"
        return self.put(key, results, CacheType.QUERY_RESULT)
    
    def get_query_result(self, query: str, search_params: Dict[str, Any] = None) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
        """Get cached query results"""
        params_str = json.dumps(search_params or {}, sort_keys=True)
        key = f"{query}:{hashlib.sha256(params_str.encode()).hexdigest()}"
        return self.get(key, CacheType.QUERY_RESULT)
    
    def cache_reranking_scores(self, query: str, doc_ids: List[str], 
                             scores: List[float], model_name: str = "default") -> bool:
        """Cache reranking scores"""
        doc_ids_str = "|".join(sorted(doc_ids))
        key = f"{model_name}:{query}:{hashlib.sha256(doc_ids_str.encode()).hexdigest()}"
        score_data = dict(zip(doc_ids, scores))
        return self.put(key, score_data, CacheType.RERANKING_SCORE)
    
    def get_reranking_scores(self, query: str, doc_ids: List[str], 
                           model_name: str = "default") -> Tuple[Optional[Dict[str, float]], bool]:
        """Get cached reranking scores"""
        doc_ids_str = "|".join(sorted(doc_ids))
        key = f"{model_name}:{query}:{hashlib.sha256(doc_ids_str.encode()).hexdigest()}"
        return self.get(key, CacheType.RERANKING_SCORE)