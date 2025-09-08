"""
Resilience utilities for Vector RAG Database
Provides retry logic, circuit breakers, and graceful degradation
"""

import time
import logging
import functools
from typing import Callable, Any, Optional, Union, Type, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

# Try to import resilience libraries
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

try:
    from pybreaker import CircuitBreaker as PyBreakerCircuitBreaker
    PYBREAKER_AVAILABLE = True
except ImportError:
    PYBREAKER_AVAILABLE = False

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns"""
    # Retry configuration
    max_retry_attempts: int = 3
    retry_base_wait: float = 1.0
    retry_max_wait: float = 60.0
    retry_multiplier: float = 2.0
    
    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_expected_exception: Type[Exception] = Exception

class SimpleCircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open state
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.RLock()
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from open to half-open"""
        if self.state != CircuitBreakerState.OPEN:
            return False
        
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            
            # Reject calls if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count and close circuit
            with self._lock:
                self.failure_count = 0
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    logger.info("Circuit breaker reset to CLOSED")
            
            return result
            
        except Exception as e:
            # Check if this exception should trigger circuit breaker
            if isinstance(e, self.expected_exception):
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    # Open circuit if threshold reached
                    if self.failure_count >= self.failure_threshold:
                        self.state = CircuitBreakerState.OPEN
                        logger.warning(
                            f"Circuit breaker OPENED after {self.failure_count} failures"
                        )
            
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'failure_threshold': self.failure_threshold,
                'last_failure_time': self.last_failure_time,
                'recovery_timeout': self.recovery_timeout
            }

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

def create_retry_decorator(config: Optional[ResilienceConfig] = None):
    """Create retry decorator based on configuration"""
    config = config or ResilienceConfig()
    
    if TENACITY_AVAILABLE:
        # Use tenacity for advanced retry logic
        return retry(
            stop=stop_after_attempt(config.max_retry_attempts),
            wait=wait_exponential(
                multiplier=config.retry_multiplier,
                min=config.retry_base_wait,
                max=config.retry_max_wait
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True
        )
    else:
        # Fallback to simple retry implementation
        def simple_retry(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(config.max_retry_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < config.max_retry_attempts - 1:
                            wait_time = min(
                                config.retry_base_wait * (config.retry_multiplier ** attempt),
                                config.retry_max_wait
                            )
                            logger.warning(
                                f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            logger.error(f"All {config.max_retry_attempts} attempts failed")
                
                raise last_exception
            
            return wrapper
        
        return simple_retry

def create_circuit_breaker(config: Optional[ResilienceConfig] = None):
    """Create circuit breaker based on configuration"""
    config = config or ResilienceConfig()
    
    if PYBREAKER_AVAILABLE:
        # Use pybreaker for advanced circuit breaker
        return PyBreakerCircuitBreaker(
            fail_max=config.circuit_breaker_failure_threshold,
            reset_timeout=config.circuit_breaker_recovery_timeout,
            expected_exception=config.circuit_breaker_expected_exception
        )
    else:
        # Use simple implementation
        return SimpleCircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout,
            expected_exception=config.circuit_breaker_expected_exception
        )

def resilient_operation(
    retry_config: Optional[ResilienceConfig] = None,
    circuit_breaker_config: Optional[ResilienceConfig] = None,
    fallback_func: Optional[Callable] = None
):
    """
    Decorator that combines retry logic and circuit breaker
    
    Args:
        retry_config: Configuration for retry behavior
        circuit_breaker_config: Configuration for circuit breaker
        fallback_func: Function to call if all else fails
    """
    def decorator(func: Callable) -> Callable:
        # Create resilience components
        retry_decorator = create_retry_decorator(retry_config)
        circuit_breaker = create_circuit_breaker(circuit_breaker_config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Apply retry logic within circuit breaker protection
                if PYBREAKER_AVAILABLE and hasattr(circuit_breaker, '__call__'):
                    # pybreaker circuit breaker
                    @circuit_breaker
                    @retry_decorator
                    def protected_func():
                        return func(*args, **kwargs)
                    
                    return protected_func()
                else:
                    # Simple circuit breaker
                    @retry_decorator
                    def retried_func():
                        return func(*args, **kwargs)
                    
                    return circuit_breaker.call(retried_func)
                    
            except Exception as e:
                logger.error(f"Resilient operation failed: {e}")
                
                # Try fallback if available
                if fallback_func:
                    try:
                        logger.info("Attempting fallback function")
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                
                raise
        
        # Expose circuit breaker state if available
        if hasattr(circuit_breaker, 'get_state'):
            wrapper.get_circuit_state = circuit_breaker.get_state
        elif hasattr(circuit_breaker, 'current_state'):
            wrapper.get_circuit_state = lambda: {'state': circuit_breaker.current_state}
        
        return wrapper
    
    return decorator

# Common resilience patterns for RAG operations

def resilient_api_call(func: Callable) -> Callable:
    """Resilient decorator for API calls (OpenAI, etc.)"""
    config = ResilienceConfig(
        max_retry_attempts=3,
        retry_base_wait=1.0,
        retry_max_wait=30.0,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_recovery_timeout=60
    )
    
    return resilient_operation(
        retry_config=config,
        circuit_breaker_config=config
    )(func)

def resilient_database_call(func: Callable) -> Callable:
    """Resilient decorator for database calls"""
    config = ResilienceConfig(
        max_retry_attempts=2,
        retry_base_wait=0.5,
        retry_max_wait=10.0,
        circuit_breaker_failure_threshold=10,
        circuit_breaker_recovery_timeout=30
    )
    
    return resilient_operation(
        retry_config=config,
        circuit_breaker_config=config
    )(func)

def resilient_file_operation(func: Callable) -> Callable:
    """Resilient decorator for file operations"""
    config = ResilienceConfig(
        max_retry_attempts=2,
        retry_base_wait=0.1,
        retry_max_wait=5.0,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout=10
    )
    
    return resilient_operation(
        retry_config=config,
        circuit_breaker_config=config
    )(func)

# Graceful degradation helpers

class GracefulDegradation:
    """Helper for implementing graceful degradation patterns"""
    
    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable, *args, **kwargs):
        """Try primary function, fall back to secondary on failure"""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed ({e}), trying fallback")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise e  # Raise original exception
    
    @staticmethod
    def with_timeout(func: Callable, timeout_seconds: float, fallback_result=None):
        """Execute function with timeout, return fallback on timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            result = func()
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            logger.warning(f"Function timed out, returning fallback result")
            return fallback_result
        finally:
            signal.signal(signal.SIGALRM, old_handler)

# Health check utilities

def health_check_with_circuit_breaker(
    service_name: str,
    check_func: Callable,
    circuit_breaker: Optional[SimpleCircuitBreaker] = None
) -> Dict[str, Any]:
    """Perform health check with circuit breaker state"""
    if circuit_breaker is None:
        circuit_breaker = SimpleCircuitBreaker(failure_threshold=3, recovery_timeout=30)
    
    health_status = {
        'service': service_name,
        'status': 'unknown',
        'timestamp': time.time(),
        'circuit_breaker': circuit_breaker.get_state()
    }
    
    try:
        # Don't perform check if circuit is open
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            health_status['status'] = 'unhealthy'
            health_status['reason'] = 'circuit_breaker_open'
            return health_status
        
        # Perform health check
        result = circuit_breaker.call(check_func)
        health_status['status'] = 'healthy'
        health_status['details'] = result
        
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['error'] = str(e)
    
    return health_status