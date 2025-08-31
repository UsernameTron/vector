"""Application configuration management"""

import os
import secrets
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseConfig(ABC):
    """Base configuration class"""
    
    # Security
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or secrets.token_urlsafe(32)
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or secrets.token_urlsafe(32)
    
    # Application settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # Database
    CHROMA_PERSIST_DIRECTORY = os.environ.get('CHROMA_PERSIST_DIRECTORY', './chroma_db')
    DEFAULT_COLLECTION_NAME = os.environ.get('DEFAULT_COLLECTION_NAME', 'vector_rag_collection')
    
    # OpenAI
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ALLOWED_ORIGINS', '*').split(',')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE_PATH = os.environ.get('LOG_FILE_PATH')
    LOG_STRUCTURED = os.environ.get('LOG_STRUCTURED', 'false').lower() == 'true'
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('RATELIMIT_STORAGE_URL', 'memory://')
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per hour')
    
    @abstractmethod
    def init_app(self, app):
        """Initialize application with configuration"""
        pass


class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    
    DEBUG = True
    TESTING = False
    
    # Relaxed settings for development
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8000', 'http://127.0.0.1:3000', 'http://127.0.0.1:8000']
    LOG_LEVEL = 'DEBUG'
    
    def init_app(self, app):
        """Development-specific initialization"""
        app.logger.info("🛠️  Running in DEVELOPMENT mode")
        app.logger.warning("⚠️  Debug mode enabled - not suitable for production!")


class TestingConfig(BaseConfig):
    """Testing configuration"""
    
    DEBUG = False
    TESTING = True
    
    # Use in-memory databases for testing
    CHROMA_PERSIST_DIRECTORY = ':memory:'
    
    # Test-specific settings
    SECRET_KEY = 'test-secret-key'
    JWT_SECRET_KEY = 'test-jwt-key'
    OPENAI_API_KEY = 'sk-test-key-for-testing'
    
    # Disable rate limiting in tests
    RATELIMIT_ENABLED = False
    
    def init_app(self, app):
        """Testing-specific initialization"""
        app.logger.info("🧪 Running in TESTING mode")


class ProductionConfig(BaseConfig):
    """Production configuration"""
    
    DEBUG = False
    TESTING = False
    
    # Strict security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Production logging
    LOG_STRUCTURED = True
    
    def init_app(self, app):
        """Production-specific initialization"""
        app.logger.info("🚀 Running in PRODUCTION mode")
        
        # Validate critical settings
        if not self.OPENAI_API_KEY or self.OPENAI_API_KEY.startswith('sk-demo'):
            app.logger.error("❌ Invalid OpenAI API key for production!")
        
        if self.SECRET_KEY == 'dev-key-change-in-production':
            app.logger.error("❌ Default secret key detected in production!")
        
        # Additional security headers
        @app.after_request
        def security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response


class CleanArchitectureConfig(BaseConfig):
    """Configuration for clean architecture implementation"""
    
    DEBUG = False
    TESTING = False
    
    # Clean architecture specific settings
    USE_DEPENDENCY_INJECTION = True
    USE_DOMAIN_EVENTS = True
    ENABLE_SWAGGER_DOCS = True
    
    def init_app(self, app):
        """Clean architecture specific initialization"""
        app.logger.info("🏗️  Running with CLEAN ARCHITECTURE")
        app.logger.info("🔧 Dependency injection enabled")


# Configuration mapping
configs = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'clean': CleanArchitectureConfig,
}


def get_config(config_name: Optional[str] = None) -> BaseConfig:
    """Get configuration by name or from environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    config_class = configs.get(config_name)
    if config_class is None:
        raise ValueError(f"Invalid configuration name: {config_name}")
    
    return config_class()


class AppMode:
    """Application mode constants"""
    DEVELOPMENT = 'development'
    TESTING = 'testing'
    PRODUCTION = 'production'
    CLEAN_ARCHITECTURE = 'clean'


class FeatureFlags:
    """Feature flags for different app modes"""
    
    def __init__(self, config: BaseConfig):
        self.config = config
    
    @property
    def use_full_rag_system(self) -> bool:
        """Whether to use the full UnifiedAI RAG system"""
        return hasattr(self.config, 'USE_FULL_RAG') and self.config.USE_FULL_RAG
    
    @property
    def use_clean_architecture(self) -> bool:
        """Whether to use clean architecture implementation"""
        return isinstance(self.config, CleanArchitectureConfig)
    
    @property
    def enable_middleware(self) -> bool:
        """Whether to enable custom middleware"""
        return not self.config.TESTING
    
    @property
    def enable_swagger(self) -> bool:
        """Whether to enable Swagger documentation"""
        return getattr(self.config, 'ENABLE_SWAGGER_DOCS', False) or self.config.DEBUG
    
    @property
    def use_structured_logging(self) -> bool:
        """Whether to use structured JSON logging"""
        return getattr(self.config, 'LOG_STRUCTURED', False)


# Usage examples and validation
if __name__ == "__main__":
    # Test all configurations
    for name, config_class in configs.items():
        print(f"\n=== {name.upper()} CONFIG ===")
        config = config_class()
        print(f"DEBUG: {config.DEBUG}")
        print(f"SECRET_KEY length: {len(config.SECRET_KEY)}")
        print(f"CHROMA_DIR: {config.CHROMA_PERSIST_DIRECTORY}")
        print(f"LOG_LEVEL: {config.LOG_LEVEL}")
        
        # Test feature flags
        flags = FeatureFlags(config)
        print(f"Clean Architecture: {flags.use_clean_architecture}")
        print(f"Swagger Enabled: {flags.enable_swagger}")
        print(f"Structured Logging: {flags.use_structured_logging}")