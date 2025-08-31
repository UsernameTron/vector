"""Centralized logging configuration with structured logging"""

import logging
import logging.config
import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        if hasattr(record, 'agent_type'):
            log_entry['agent_type'] = record.agent_type
            
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


def setup_logging(
    level: str = None,
    log_file: str = None,
    structured: bool = False,
    request_id: str = None
) -> logging.Logger:
    """Setup centralized logging configuration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Use structured JSON logging
        request_id: Request ID for correlation
    """
    # Get configuration from environment
    level = level or os.getenv('LOG_LEVEL', 'INFO')
    log_file = log_file or os.getenv('LOG_FILE_PATH')
    structured = structured or os.getenv('LOG_STRUCTURED', '').lower() == 'true'
    
    # Create logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s'
            },
            'json': {
                '()': JSONFormatter,
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'json' if structured else 'standard',
                'stream': sys.stdout,
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': level,
                'propagate': False,
            },
            # Application specific loggers
            'vector_rag': {
                'handlers': ['console'],
                'level': level,
                'propagate': False,
            },
            'agents': {
                'handlers': ['console'],
                'level': level,
                'propagate': False,
            },
            'vector_db': {
                'handlers': ['console'],
                'level': level,
                'propagate': False,
            },
            # Third-party loggers (reduce noise)
            'werkzeug': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            },
            'chromadb': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False,
            },
        },
    }
    
    # Add file handler if specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'json' if structured else 'detailed',
            'filename': log_file,
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'encoding': 'utf-8',
        }
        
        # Add file handler to all loggers
        for logger_config in config['loggers'].values():
            logger_config['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Create and configure root logger
    logger = logging.getLogger('vector_rag')
    
    if request_id:
        # Add request ID to all log records
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id
            return record
            
        logging.setLogRecordFactory(record_factory)
    
    logger.info(f"Logging configured: level={level}, structured={structured}, file={log_file}")
    return logger


def get_logger(name: str, **extra_fields) -> logging.LoggerAdapter:
    """Get a logger with optional extra fields
    
    Args:
        name: Logger name
        **extra_fields: Additional fields to include in log records
    """
    logger = logging.getLogger(name)
    
    if extra_fields:
        return logging.LoggerAdapter(logger, extra_fields)
    
    return logger


class ContextualLogger:
    """Logger that maintains context across requests"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def bind(self, **kwargs):
        """Bind context variables to the logger"""
        self.context.update(kwargs)
        return self
    
    def _log(self, level, msg, *args, **kwargs):
        """Internal logging method with context"""
        # Merge context into kwargs
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)


def setup_flask_logging(app):
    """Setup Flask-specific logging"""
    from flask import request, g
    import uuid
    
    @app.before_request
    def before_request():
        g.request_id = str(uuid.uuid4())
        g.start_time = datetime.now()
        
        logger = get_logger('vector_rag.requests')
        logger.info(
            f"Request started: {request.method} {request.path}",
            extra={
                'request_id': g.request_id,
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', ''),
            }
        )
    
    @app.after_request
    def after_request(response):
        duration = (datetime.now() - g.start_time).total_seconds() * 1000
        
        logger = get_logger('vector_rag.requests')
        logger.info(
            f"Request completed: {response.status_code} in {duration:.2f}ms",
            extra={
                'request_id': g.request_id,
                'status_code': response.status_code,
                'duration_ms': duration,
            }
        )
        
        # Add request ID header
        response.headers['X-Request-ID'] = g.request_id
        return response
    
    return app


# Usage examples:
if __name__ == "__main__":
    # Basic setup
    logger = setup_logging(level='INFO', structured=True)
    logger.info("Application starting")
    
    # Contextual logging
    ctx_logger = ContextualLogger('vector_rag.agent')
    ctx_logger.bind(agent_type='research', user_id='user123')
    ctx_logger.info("Agent processing request")
    
    # With extra fields
    adapter_logger = get_logger('vector_rag.db', collection_name='documents')
    adapter_logger.info("Database operation completed")