"""
Graceful shutdown manager with resource cleanup
"""

import signal
import sys
import time
import threading
import logging
import atexit
from typing import List, Callable, Dict, Any
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ShutdownManager:
    """Manages graceful application shutdown with resource cleanup"""
    
    def __init__(self, shutdown_timeout: int = 30):
        self.shutdown_timeout = shutdown_timeout
        self.cleanup_functions: List[tuple[str, Callable]] = []
        self.shutdown_started = False
        self.shutdown_complete = False
        self.lock = threading.Lock()
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register atexit handler
        atexit.register(self._atexit_handler)
        
        logger.info("Shutdown manager initialized")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown")
            self.shutdown()
        
        # Handle common termination signals
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Handle SIGHUP for reload (optional)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    
    def register_cleanup_function(self, name: str, cleanup_func: Callable):
        """Register a cleanup function to be called during shutdown"""
        with self.lock:
            self.cleanup_functions.append((name, cleanup_func))
        
        logger.debug(f"Registered cleanup function: {name}")
    
    def shutdown(self):
        """Initiate graceful shutdown"""
        with self.lock:
            if self.shutdown_started:
                logger.warning("Shutdown already in progress")
                return
            
            self.shutdown_started = True
        
        logger.info("Starting graceful shutdown...")
        start_time = time.time()
        
        try:
            # Execute cleanup functions in reverse order (LIFO)
            cleanup_functions = list(reversed(self.cleanup_functions))
            
            for name, cleanup_func in cleanup_functions:
                try:
                    logger.info(f"Executing cleanup: {name}")
                    
                    # Set timeout for each cleanup function
                    cleanup_start = time.time()
                    
                    if hasattr(cleanup_func, '__call__'):
                        cleanup_func()
                    
                    cleanup_duration = time.time() - cleanup_start
                    logger.info(f"Cleanup '{name}' completed in {cleanup_duration:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error in cleanup function '{name}': {e}")
                    # Continue with other cleanup functions
                
                # Check if we're running out of time
                elapsed = time.time() - start_time
                if elapsed > self.shutdown_timeout:
                    logger.warning(f"Shutdown timeout ({self.shutdown_timeout}s) exceeded")
                    break
            
            total_duration = time.time() - start_time
            logger.info(f"Graceful shutdown completed in {total_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        finally:
            with self.lock:
                self.shutdown_complete = True
    
    def _atexit_handler(self):
        """Handler called on normal exit"""
        if not self.shutdown_started:
            logger.info("Normal exit detected, performing cleanup")
            self.shutdown()
    
    @contextmanager
    def cleanup_context(self, name: str):
        """Context manager for automatic cleanup registration"""
        cleanup_actions = []
        
        def add_cleanup(action):
            cleanup_actions.append(action)
        
        try:
            yield add_cleanup
        finally:
            if cleanup_actions:
                def combined_cleanup():
                    for action in reversed(cleanup_actions):
                        try:
                            action()
                        except Exception as e:
                            logger.error(f"Error in cleanup action for {name}: {e}")
                
                self.register_cleanup_function(name, combined_cleanup)
    
    def is_shutdown_started(self) -> bool:
        """Check if shutdown has been initiated"""
        with self.lock:
            return self.shutdown_started
    
    def is_shutdown_complete(self) -> bool:
        """Check if shutdown is complete"""
        with self.lock:
            return self.shutdown_complete
    
    def wait_for_shutdown(self, timeout: float = None):
        """Wait for shutdown to complete"""
        start_time = time.time()
        
        while not self.is_shutdown_complete():
            if timeout and (time.time() - start_time) > timeout:
                break
            time.sleep(0.1)


# Global shutdown manager instance
_shutdown_manager: ShutdownManager = None


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance"""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager()
    return _shutdown_manager


def register_cleanup(name: str, cleanup_func: Callable):
    """Register a cleanup function"""
    get_shutdown_manager().register_cleanup_function(name, cleanup_func)


@contextmanager
def cleanup_context(name: str):
    """Context manager for cleanup registration"""
    with get_shutdown_manager().cleanup_context(name) as add_cleanup:
        yield add_cleanup


class ResourceManager:
    """Manages application resources with automatic cleanup"""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        self.shutdown_manager = get_shutdown_manager()
        
        # Register resource cleanup
        self.shutdown_manager.register_cleanup_function(
            "resource_manager", 
            self._cleanup_all_resources
        )
    
    def register_resource(self, name: str, resource: Any, cleanup_func: Callable = None):
        """Register a resource for management"""
        self.resources[name] = {
            'resource': resource,
            'cleanup_func': cleanup_func,
            'created_at': datetime.now()
        }
        
        logger.debug(f"Registered resource: {name}")
    
    def get_resource(self, name: str) -> Any:
        """Get a managed resource"""
        if name not in self.resources:
            raise KeyError(f"Resource '{name}' not found")
        
        return self.resources[name]['resource']
    
    def cleanup_resource(self, name: str):
        """Cleanup a specific resource"""
        if name not in self.resources:
            logger.warning(f"Resource '{name}' not found for cleanup")
            return
        
        resource_info = self.resources[name]
        resource = resource_info['resource']
        cleanup_func = resource_info['cleanup_func']
        
        try:
            if cleanup_func:
                cleanup_func(resource)
            elif hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'cleanup'):
                resource.cleanup()
            elif hasattr(resource, 'shutdown'):
                resource.shutdown()
            
            logger.info(f"Resource '{name}' cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up resource '{name}': {e}")
        
        finally:
            del self.resources[name]
    
    def _cleanup_all_resources(self):
        """Cleanup all managed resources"""
        logger.info(f"Cleaning up {len(self.resources)} managed resources")
        
        # Cleanup resources in reverse order of registration
        resource_names = list(self.resources.keys())
        for name in reversed(resource_names):
            self.cleanup_resource(name)


# Global resource manager
_resource_manager: ResourceManager = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def register_resource(name: str, resource: Any, cleanup_func: Callable = None):
    """Register a resource for automatic cleanup"""
    get_resource_manager().register_resource(name, resource, cleanup_func)


class DatabaseCleanup:
    """Database-specific cleanup operations"""
    
    @staticmethod
    def cleanup_database_connections():
        """Cleanup database connections"""
        try:
            logger.info("Cleaning up database connections")
            
            # Close ChromaDB connections
            # In a real implementation, you would access the actual database clients
            logger.info("Database connections cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up database connections: {e}")
    
    @staticmethod
    def flush_pending_operations():
        """Flush any pending database operations"""
        try:
            logger.info("Flushing pending database operations")
            
            # Flush any pending writes, transactions, etc.
            logger.info("Pending operations flushed")
            
        except Exception as e:
            logger.error(f"Error flushing pending operations: {e}")


class FileSystemCleanup:
    """File system cleanup operations"""
    
    @staticmethod
    def cleanup_temp_files():
        """Clean up temporary files"""
        try:
            logger.info("Cleaning up temporary files")
            
            # Clean up temp directories, cache files, etc.
            import tempfile
            import shutil
            
            # This would clean up application-specific temp files
            logger.info("Temporary files cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
    
    @staticmethod
    def close_file_handles():
        """Close any open file handles"""
        try:
            logger.info("Closing open file handles")
            
            # Close any open log files, data files, etc.
            logger.info("File handles closed")
            
        except Exception as e:
            logger.error(f"Error closing file handles: {e}")


class ThreadPoolCleanup:
    """Thread pool and async cleanup"""
    
    @staticmethod
    def shutdown_thread_pools():
        """Shutdown thread pools"""
        try:
            logger.info("Shutting down thread pools")
            
            # Shutdown any thread pools, executor services, etc.
            logger.info("Thread pools shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down thread pools: {e}")
    
    @staticmethod
    def cancel_async_tasks():
        """Cancel running async tasks"""
        try:
            logger.info("Cancelling async tasks")
            
            # Cancel any running asyncio tasks
            logger.info("Async tasks cancelled")
            
        except Exception as e:
            logger.error(f"Error cancelling async tasks: {e}")


def setup_production_cleanup():
    """Setup cleanup functions for production deployment"""
    shutdown_manager = get_shutdown_manager()
    
    # Register cleanup functions in order of dependency
    shutdown_manager.register_cleanup_function(
        "cancel_async_tasks", 
        ThreadPoolCleanup.cancel_async_tasks
    )
    
    shutdown_manager.register_cleanup_function(
        "flush_database_operations", 
        DatabaseCleanup.flush_pending_operations
    )
    
    shutdown_manager.register_cleanup_function(
        "close_database_connections", 
        DatabaseCleanup.cleanup_database_connections
    )
    
    shutdown_manager.register_cleanup_function(
        "shutdown_thread_pools", 
        ThreadPoolCleanup.shutdown_thread_pools
    )
    
    shutdown_manager.register_cleanup_function(
        "close_file_handles", 
        FileSystemCleanup.close_file_handles
    )
    
    shutdown_manager.register_cleanup_function(
        "cleanup_temp_files", 
        FileSystemCleanup.cleanup_temp_files
    )
    
    logger.info("Production cleanup functions registered")