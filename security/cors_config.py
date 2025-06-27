"""
CORS configuration for Flask application
Secure CORS setup with specific allowed origins
"""

from flask_cors import CORS
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SecureCORS:
    """Secure CORS configuration management"""
    
    # Default allowed origins for development
    DEFAULT_ALLOWED_ORIGINS = [
        'http://localhost:3000',
        'http://localhost:8000',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:8000'
    ]
    
    # Production origins should be set via environment variables
    @staticmethod
    def get_allowed_origins() -> List[str]:
        """Get allowed origins from environment or defaults"""
        env_origins = os.getenv('CORS_ALLOWED_ORIGINS')
        
        if env_origins:
            # Parse comma-separated origins from environment
            origins = [origin.strip() for origin in env_origins.split(',')]
            logger.info(f"Using CORS origins from environment: {origins}")
            return origins
        else:
            logger.warning("No CORS_ALLOWED_ORIGINS set, using development defaults")
            return SecureCORS.DEFAULT_ALLOWED_ORIGINS
    
    @staticmethod
    def get_cors_config() -> Dict[str, Any]:
        """Get secure CORS configuration"""
        allowed_origins = SecureCORS.get_allowed_origins()
        
        config = {
            'origins': allowed_origins,
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allow_headers': [
                'Content-Type',
                'Authorization',
                'X-Requested-With',
                'Accept',
                'Origin'
            ],
            'expose_headers': [
                'Content-Range',
                'X-Content-Range',
                'X-Total-Count'
            ],
            'supports_credentials': True,
            'max_age': 3600,  # Cache preflight requests for 1 hour
        }
        
        return config
    
    @staticmethod
    def init_cors(app):
        """Initialize CORS with secure configuration"""
        config = SecureCORS.get_cors_config()
        
        cors = CORS(app, **config)
        
        logger.info("CORS initialized with secure configuration")
        logger.debug(f"Allowed origins: {config['origins']}")
        
        return cors
    
    @staticmethod
    def validate_origin(origin: str) -> bool:
        """Validate if origin is allowed"""
        allowed_origins = SecureCORS.get_allowed_origins()
        return origin in allowed_origins