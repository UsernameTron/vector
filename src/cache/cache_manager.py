"""
Cache Manager for Vector RAG Database
Provides caching for embeddings, search results, and other expensive operations
"""

import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for caching"""
    enable_disk_cache: bool = True
    enable_redis_cache: bool = False
    disk_cache_dir: str = "./data/cache"
    redis_url: str = "redis://localhost:6379/0"
    default_ttl_seconds: int = 3600 * 24  # 24 hours
    max_disk_cache_size_mb: int = 1000  # 1GB
    cleanup_interval_seconds: int = 3600  # 1 hour

class DiskCache:
    """File-based disk cache"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 1000):
        """Initialize disk cache"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
    
    def _compute_key_path(self, key: str) -> Path:
        """Compute file path for cache key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        # Create nested directories to avoid too many files in one directory
        return self.cache_dir / key_hash[:2] / key_hash[2:4] / f"{key_hash}.cache"
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        try:
            for cache_file in self.cache_dir.rglob("*.cache"):
                if cache_file.is_file():
                    total_size += cache_file.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to compute cache size: {e}")
        return total_size
    
    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.now()
            for cache_file in self.cache_dir.rglob("*.cache"):
                if cache_file.is_file():
                    try:
                        # Read metadata to check expiration
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                            if 'expires_at' in data and data['expires_at']:
                                expires_at = datetime.fromisoformat(data['expires_at'])
                                if current_time > expires_at:
                                    cache_file.unlink()
                                    logger.debug(f"Removed expired cache file: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Error checking cache file {cache_file}: {e}")
                        # Remove corrupted files
                        try:
                            cache_file.unlink()
                        except:
                            pass
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _ensure_space(self):
        """Ensure cache doesn't exceed size limits"""
        try:
            current_size = self._get_cache_size()
            if current_size > self.max_size_bytes:
                logger.info(f"Cache size {current_size / 1024 / 1024:.1f}MB exceeds limit, cleaning up")
                
                # Get all cache files sorted by modification time (oldest first)
                cache_files = []
                for cache_file in self.cache_dir.rglob("*.cache"):
                    if cache_file.is_file():
                        cache_files.append((cache_file.stat().st_mtime, cache_file))
                
                cache_files.sort()
                
                # Remove oldest files until we're under the limit
                removed_size = 0
                for mtime, cache_file in cache_files:
                    if current_size - removed_size <= self.max_size_bytes * 0.8:  # 80% of limit
                        break
                    try:
                        file_size = cache_file.stat().st_size
                        cache_file.unlink()
                        removed_size += file_size
                        logger.debug(f"Removed cache file for space: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {cache_file}: {e}")
                
                logger.info(f"Freed {removed_size / 1024 / 1024:.1f}MB of cache space")
                
        except Exception as e:
            logger.error(f"Cache space management failed: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            try:
                cache_path = self._compute_key_path(key)
                if not cache_path.exists():
                    return None
                
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check expiration
                if 'expires_at' in data and data['expires_at']:
                    expires_at = datetime.fromisoformat(data['expires_at'])
                    if datetime.now() > expires_at:
                        cache_path.unlink()
                        return None
                
                return data['value']
                
            except Exception as e:
                logger.debug(f"Cache get failed for key {key}: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self._lock:
            try:
                cache_path = self._compute_key_path(key)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Prepare cache data
                cache_data = {
                    'value': value,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': None
                }
                
                if ttl_seconds:
                    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                    cache_data['expires_at'] = expires_at.isoformat()
                
                # Write to cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                # Manage cache size
                self._ensure_space()
                
                return True
                
            except Exception as e:
                logger.warning(f"Cache set failed for key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self._lock:
            try:
                cache_path = self._compute_key_path(key)
                if cache_path.exists():
                    cache_path.unlink()
                    return True
                return False
                
            except Exception as e:
                logger.warning(f"Cache delete failed for key {key}: {e}")
                return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            try:
                for cache_file in self.cache_dir.rglob("*.cache"):
                    if cache_file.is_file():
                        cache_file.unlink()
                logger.info("Disk cache cleared")
                return True
                
            except Exception as e:
                logger.error(f"Cache clear failed: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.rglob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
            
            return {
                'type': 'disk',
                'directory': str(self.cache_dir),
                'file_count': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'usage_percent': (total_size / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'type': 'disk', 'error': str(e)}

class RedisCache:
    """Redis-based cache"""
    
    def __init__(self, redis_url: str):
        """Initialize Redis cache"""
        self.redis_url = redis_url
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.error("Redis not available - install with: pip install redis>=4.5.0")
            return
        
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=False)
            # Test connection
            self.client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.client:
            return None
        
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.debug(f"Redis get failed for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.client:
            return False
        
        try:
            serialized_data = pickle.dumps(value)
            if ttl_seconds:
                self.client.setex(key, ttl_seconds, serialized_data)
            else:
                self.client.set(key, serialized_data)
            return True
            
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.client:
            return False
        
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete failed for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        if not self.client:
            return False
        
        try:
            self.client.flushdb()
            logger.info("Redis cache cleared")
            return True
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.client:
            return {'type': 'redis', 'connected': False}
        
        try:
            info = self.client.info()
            return {
                'type': 'redis',
                'connected': True,
                'used_memory_bytes': info.get('used_memory', 0),
                'used_memory_mb': info.get('used_memory', 0) / 1024 / 1024,
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'connected_clients': info.get('connected_clients', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {'type': 'redis', 'connected': False, 'error': str(e)}

class CacheManager:
    """Multi-layer cache manager"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager"""
        self.config = config or CacheConfig()
        self.disk_cache = None
        self.redis_cache = None
        
        # Initialize caches
        if self.config.enable_disk_cache:
            self.disk_cache = DiskCache(
                self.config.disk_cache_dir,
                self.config.max_disk_cache_size_mb
            )
        
        if self.config.enable_redis_cache:
            self.redis_cache = RedisCache(self.config.redis_url)
    
    def _compute_cache_key(self, namespace: str, key: str) -> str:
        """Compute cache key with namespace"""
        return f"{namespace}:{hashlib.sha256(key.encode()).hexdigest()}"
    
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache (tries Redis first, then disk)"""
        cache_key = self._compute_cache_key(namespace, key)
        
        # Try Redis first (faster)
        if self.redis_cache:
            value = self.redis_cache.get(cache_key)
            if value is not None:
                logger.debug(f"Cache hit (Redis): {namespace}:{key}")
                return value
        
        # Try disk cache
        if self.disk_cache:
            value = self.disk_cache.get(cache_key)
            if value is not None:
                logger.debug(f"Cache hit (Disk): {namespace}:{key}")
                # Promote to Redis if available
                if self.redis_cache:
                    self.redis_cache.set(cache_key, value, self.config.default_ttl_seconds)
                return value
        
        logger.debug(f"Cache miss: {namespace}:{key}")
        return None
    
    def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache (writes to both Redis and disk)"""
        cache_key = self._compute_cache_key(namespace, key)
        ttl = ttl_seconds or self.config.default_ttl_seconds
        
        success = True
        
        # Write to Redis
        if self.redis_cache:
            if not self.redis_cache.set(cache_key, value, ttl):
                success = False
        
        # Write to disk cache
        if self.disk_cache:
            if not self.disk_cache.set(cache_key, value, ttl):
                success = False
        
        if success:
            logger.debug(f"Cache set: {namespace}:{key}")
        
        return success
    
    def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._compute_cache_key(namespace, key)
        
        success = True
        
        # Delete from Redis
        if self.redis_cache:
            if not self.redis_cache.delete(cache_key):
                success = False
        
        # Delete from disk
        if self.disk_cache:
            if not self.disk_cache.delete(cache_key):
                success = False
        
        return success
    
    def clear(self, namespace: Optional[str] = None) -> bool:
        """Clear cache entries"""
        success = True
        
        if namespace:
            # TODO: Implement namespace-specific clearing
            logger.warning("Namespace-specific cache clearing not yet implemented")
            return False
        
        # Clear all
        if self.redis_cache:
            if not self.redis_cache.clear():
                success = False
        
        if self.disk_cache:
            if not self.disk_cache.clear():
                success = False
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'config': {
                'disk_cache_enabled': self.config.enable_disk_cache,
                'redis_cache_enabled': self.config.enable_redis_cache,
                'default_ttl_seconds': self.config.default_ttl_seconds
            },
            'backends': {}
        }
        
        if self.disk_cache:
            stats['backends']['disk'] = self.disk_cache.get_stats()
        
        if self.redis_cache:
            stats['backends']['redis'] = self.redis_cache.get_stats()
        
        return stats

# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get or create global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        config = CacheConfig(
            enable_disk_cache=True,
            enable_redis_cache=os.getenv('CACHE_REDIS_ENABLED', 'false').lower() == 'true',
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            default_ttl_seconds=int(os.getenv('CACHE_TTL_SECONDS', '86400')),  # 24 hours
            disk_cache_dir=os.getenv('CACHE_DIR', './data/cache')
        )
        _cache_manager = CacheManager(config)
    return _cache_manager

# Convenience functions
def cache_get(namespace: str, key: str) -> Optional[Any]:
    """Get value from cache"""
    return get_cache_manager().get(namespace, key)

def cache_set(namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
    """Set value in cache"""
    return get_cache_manager().set(namespace, key, value, ttl_seconds)

def cache_delete(namespace: str, key: str) -> bool:
    """Delete value from cache"""
    return get_cache_manager().delete(namespace, key)

def cache_clear(namespace: Optional[str] = None) -> bool:
    """Clear cache entries"""
    return get_cache_manager().clear(namespace)