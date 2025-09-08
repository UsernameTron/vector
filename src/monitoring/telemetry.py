"""
Telemetry and Observability Module
Provides structured logging, metrics, and tracing for RAG operations
"""

import os
import time
import logging
import functools
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

# Try to import observability libraries
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

@dataclass
class TelemetryConfig:
    """Configuration for telemetry"""
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_structured_logging: bool = True
    service_name: str = "vector-rag-database"
    metrics_port: int = 8000
    log_level: str = "INFO"

class TelemetryManager:
    """Centralized telemetry management"""
    
    def __init__(self, config: Optional[TelemetryConfig] = None):
        """Initialize telemetry manager"""
        self.config = config or TelemetryConfig()
        self._tracer = None
        self._meter = None
        self._logger = None
        self._metrics = {}
        
        self._setup_logging()
        self._setup_tracing()
        self._setup_metrics()
    
    def _setup_logging(self):
        """Setup structured logging"""
        if not self.config.enable_structured_logging or not STRUCTLOG_AVAILABLE:
            self._logger = logging.getLogger(__name__)
            return
            
        try:
            # Configure structlog
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            
            # Set log level
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper()),
                format="%(message)s"
            )
            
            self._logger = structlog.get_logger(self.config.service_name)
            self._logger.info("Structured logging initialized")
            
        except Exception as e:
            # Fallback to standard logging
            self._logger = logging.getLogger(__name__)
            self._logger.warning(f"Failed to setup structured logging: {e}")
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        if not self.config.enable_tracing or not OTEL_AVAILABLE:
            return
            
        try:
            # Configure tracer provider
            trace.set_tracer_provider(TracerProvider())
            self._tracer = trace.get_tracer(
                __name__,
                version="1.0.0"
            )
            self._logger.info("OpenTelemetry tracing initialized") if self._logger else None
            
        except Exception as e:
            if self._logger:
                self._logger.warning(f"Failed to setup tracing: {e}")
    
    def _setup_metrics(self):
        """Setup OpenTelemetry metrics"""
        if not self.config.enable_metrics or not OTEL_AVAILABLE:
            return
            
        try:
            # Configure metrics provider
            metrics.set_meter_provider(MeterProvider())
            self._meter = metrics.get_meter(
                __name__,
                version="1.0.0"
            )
            
            # Create metrics instruments
            self._metrics = {
                'retrieval_duration': self._meter.create_histogram(
                    "retrieval_duration_seconds",
                    description="Time spent on retrieval operations",
                    unit="s"
                ),
                'retrieval_count': self._meter.create_counter(
                    "retrieval_total",
                    description="Total number of retrieval operations"
                ),
                'document_count': self._meter.create_up_down_counter(
                    "documents_total",
                    description="Total number of documents in database"
                ),
                'chunk_count': self._meter.create_up_down_counter(
                    "chunks_total",
                    description="Total number of chunks processed"
                ),
                'ingestion_duration': self._meter.create_histogram(
                    "ingestion_duration_seconds",
                    description="Time spent on document ingestion",
                    unit="s"
                ),
                'reranking_duration': self._meter.create_histogram(
                    "reranking_duration_seconds",
                    description="Time spent on reranking operations",
                    unit="s"
                ),
                'cache_hits': self._meter.create_counter(
                    "cache_hits_total",
                    description="Total number of cache hits"
                ),
                'cache_misses': self._meter.create_counter(
                    "cache_misses_total",
                    description="Total number of cache misses"
                ),
            }
            
            if self._logger:
                self._logger.info("OpenTelemetry metrics initialized")
                
        except Exception as e:
            if self._logger:
                self._logger.warning(f"Failed to setup metrics: {e}")
    
    def get_logger(self) -> Any:
        """Get configured logger"""
        return self._logger or logging.getLogger(__name__)
    
    def trace_operation(self, operation_name: str):
        """Decorator for tracing operations"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._tracer:
                    return func(*args, **kwargs)
                
                with self._tracer.start_as_current_span(operation_name) as span:
                    try:
                        # Add operation metadata
                        span.set_attribute("operation.name", operation_name)
                        span.set_attribute("operation.start_time", time.time())
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Add result metadata
                        if hasattr(result, '__len__'):
                            span.set_attribute("operation.result_count", len(result))
                        
                        span.set_attribute("operation.success", True)
                        return result
                        
                    except Exception as e:
                        span.set_attribute("operation.success", False)
                        span.set_attribute("operation.error", str(e))
                        span.record_exception(e)
                        raise
                        
            return wrapper
        return decorator
    
    def time_operation(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator for timing operations"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                labels = labels or {}
                
                try:
                    result = func(*args, **kwargs)
                    labels['status'] = 'success'
                    return result
                    
                except Exception as e:
                    labels['status'] = 'error'
                    labels['error_type'] = type(e).__name__
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    
                    # Record metrics
                    if metric_name in self._metrics:
                        self._metrics[metric_name].record(duration, labels)
                    
                    # Log operation
                    logger = self.get_logger()
                    logger.info(
                        "Operation completed",
                        operation=func.__name__,
                        duration_seconds=duration,
                        **labels
                    )
                        
            return wrapper
        return decorator
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        if metric_name in self._metrics and self._metrics[metric_name]:
            self._metrics[metric_name].add(value, labels or {})
    
    def increment_counter(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        if metric_name in self._metrics and self._metrics[metric_name]:
            if hasattr(self._metrics[metric_name], 'add'):
                self._metrics[metric_name].add(1, labels or {})
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get telemetry system health"""
        return {
            'telemetry': {
                'structured_logging': STRUCTLOG_AVAILABLE and self.config.enable_structured_logging,
                'tracing': OTEL_AVAILABLE and self.config.enable_tracing and self._tracer is not None,
                'metrics': OTEL_AVAILABLE and self.config.enable_metrics and self._meter is not None,
            },
            'config': {
                'service_name': self.config.service_name,
                'log_level': self.config.log_level
            }
        }

# Global telemetry manager instance
_telemetry_manager = None

def get_telemetry_manager() -> TelemetryManager:
    """Get or create global telemetry manager"""
    global _telemetry_manager
    if _telemetry_manager is None:
        config = TelemetryConfig(
            enable_tracing=os.getenv('OTEL_ENABLED', 'false').lower() == 'true',
            enable_metrics=os.getenv('OTEL_ENABLED', 'false').lower() == 'true',
            service_name=os.getenv('OTEL_SERVICE_NAME', 'vector-rag-database'),
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )
        _telemetry_manager = TelemetryManager(config)
    return _telemetry_manager

# Convenience functions
def get_logger():
    """Get configured logger"""
    return get_telemetry_manager().get_logger()

def trace_span(operation_name: str):
    """Decorator for tracing spans"""
    return get_telemetry_manager().trace_operation(operation_name)

def time_metric(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator for timing metrics"""
    return get_telemetry_manager().time_operation(metric_name, labels)

def record_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a metric value"""
    get_telemetry_manager().record_metric(metric_name, value, labels)

def increment_counter(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Increment a counter metric"""
    get_telemetry_manager().increment_counter(metric_name, labels)

# Specific decorators for RAG operations
def trace_retrieval(func: Callable) -> Callable:
    """Decorator for retrieval operations"""
    return trace_span("retrieval")(time_metric("retrieval_duration")(func))

def trace_ingestion(func: Callable) -> Callable:
    """Decorator for ingestion operations"""
    return trace_span("ingestion")(time_metric("ingestion_duration")(func))

def trace_reranking(func: Callable) -> Callable:
    """Decorator for reranking operations"""
    return trace_span("reranking")(time_metric("reranking_duration")(func))

def trace_chunking(func: Callable) -> Callable:
    """Decorator for chunking operations"""
    return trace_span("chunking")(func)