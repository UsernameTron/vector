"""
Logging service implementation
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from src.domain.interfaces import ILoggingService
from src.infrastructure.container import singleton


@singleton(ILoggingService)
class LoggingService(ILoggingService):
    """Logging service implementation using Python's logging module"""
    
    def __init__(self):
        self.logger = logging.getLogger("VectorRAG")
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            self._configure_logger()
    
    def _configure_logger(self):
        """Configure the logger with appropriate handlers and formatters"""
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    async def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log info message"""
        self._log_with_context(logging.INFO, message, context)
    
    async def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, context)
    
    async def log_error(self, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error message"""
        full_context = context or {}
        if exception:
            full_context['exception'] = str(exception)
            full_context['exception_type'] = type(exception).__name__
        
        self._log_with_context(logging.ERROR, message, full_context, exception)
    
    async def log_debug(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, context)
    
    def _log_with_context(self, level: int, message: str, context: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
        """Log message with context"""
        if context:
            # Format context as JSON for structured logging
            try:
                context_str = json.dumps(context, default=str, separators=(',', ':'))
                full_message = f"{message} | Context: {context_str}"
            except Exception:
                # Fallback if JSON serialization fails
                full_message = f"{message} | Context: {str(context)}"
        else:
            full_message = message
        
        self.logger.log(level, full_message, exc_info=exception)