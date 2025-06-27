"""
Environment-specific configuration management
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    chroma_persist_directory: str = "./chroma_db"
    collection_name: str = "vector_rag_collection"
    connection_timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            chroma_persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY', cls.chroma_persist_directory),
            collection_name=os.getenv('DEFAULT_COLLECTION_NAME', cls.collection_name),
            connection_timeout=int(os.getenv('DB_CONNECTION_TIMEOUT', cls.connection_timeout)),
            max_retries=int(os.getenv('DB_MAX_RETRIES', cls.max_retries))
        )


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    jwt_secret_key: str = ""
    jwt_expiration_hours: int = 24
    encryption_secret: str = ""
    cors_origins: str = "*"
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        return cls(
            secret_key=os.getenv('FLASK_SECRET_KEY', ''),
            jwt_secret_key=os.getenv('JWT_SECRET_KEY', ''),
            jwt_expiration_hours=int(os.getenv('JWT_EXPIRATION_HOURS', cls.jwt_expiration_hours)),
            encryption_secret=os.getenv('ENCRYPTION_SECRET', ''),
            cors_origins=os.getenv('CORS_ALLOWED_ORIGINS', cls.cors_origins),
            rate_limit_enabled=os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            rate_limit_per_minute=int(os.getenv('RATE_LIMIT_PER_MINUTE', cls.rate_limit_per_minute))
        )
    
    def validate(self) -> list[str]:
        """Validate security configuration"""
        errors = []
        
        if not self.secret_key:
            errors.append("FLASK_SECRET_KEY is required")
        elif len(self.secret_key) < 32:
            errors.append("FLASK_SECRET_KEY should be at least 32 characters")
        
        if not self.jwt_secret_key:
            errors.append("JWT_SECRET_KEY is required")
        elif len(self.jwt_secret_key) < 32:
            errors.append("JWT_SECRET_KEY should be at least 32 characters")
        
        if not self.encryption_secret:
            errors.append("ENCRYPTION_SECRET is required")
        elif len(self.encryption_secret) < 32:
            errors.append("ENCRYPTION_SECRET should be at least 32 characters")
        
        return errors


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        return cls(
            level=os.getenv('LOG_LEVEL', cls.level).upper(),
            format=os.getenv('LOG_FORMAT', cls.format),
            file_path=os.getenv('LOG_FILE_PATH'),
            max_bytes=int(os.getenv('LOG_MAX_BYTES', cls.max_bytes)),
            backup_count=int(os.getenv('LOG_BACKUP_COUNT', cls.backup_count))
        )


@dataclass
class MonitoringConfig:
    """Monitoring and health check configuration"""
    health_check_enabled: bool = True
    metrics_enabled: bool = True
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        return cls(
            health_check_enabled=os.getenv('HEALTH_CHECK_ENABLED', 'true').lower() == 'true',
            metrics_enabled=os.getenv('METRICS_ENABLED', 'true').lower() == 'true',
            prometheus_enabled=os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true',
            prometheus_port=int(os.getenv('PROMETHEUS_PORT', cls.prometheus_port))
        )


@dataclass
class ExternalServicesConfig:
    """External services configuration"""
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_timeout: int = 60
    redis_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'ExternalServicesConfig':
        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            openai_api_base=os.getenv('OPENAI_API_BASE', cls.openai_api_base),
            openai_timeout=int(os.getenv('OPENAI_TIMEOUT', cls.openai_timeout)),
            redis_url=os.getenv('REDIS_URL'),
            elasticsearch_url=os.getenv('ELASTICSEARCH_URL')
        )


@dataclass
class EnvironmentConfig:
    """Complete environment configuration"""
    environment: str = "development"
    debug: bool = False
    testing: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    external_services: ExternalServicesConfig = field(default_factory=ExternalServicesConfig)
    
    @classmethod
    def from_env(cls) -> 'EnvironmentConfig':
        """Create configuration from environment variables"""
        env = os.getenv('FLASK_ENV', 'development').lower()
        
        config = cls(
            environment=env,
            debug=env == 'development',
            testing=env == 'testing',
            host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', '8000')),
            
            database=DatabaseConfig.from_env(),
            security=SecurityConfig.from_env(),
            logging=LoggingConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            external_services=ExternalServicesConfig.from_env()
        )
        
        return config
    
    @classmethod
    def from_file(cls, config_file: str) -> 'EnvironmentConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Create config from file data
            config = cls()
            
            # Update fields from JSON
            for key, value in data.items():
                if hasattr(config, key):
                    if isinstance(getattr(config, key), (DatabaseConfig, SecurityConfig, LoggingConfig, MonitoringConfig, ExternalServicesConfig)):
                        # Handle nested config objects
                        nested_config = getattr(config, key)
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(nested_config, nested_key, nested_value)
                    else:
                        setattr(config, key, value)
            
            logger.info(f"Configuration loaded from file: {config_file}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_file}")
            return cls.from_env()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
            return cls.from_env()
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            return cls.from_env()
    
    def validate(self) -> list[str]:
        """Validate configuration"""
        errors = []
        
        # Validate security configuration
        security_errors = self.security.validate()
        errors.extend(security_errors)
        
        # Validate database configuration
        if not self.database.chroma_persist_directory:
            errors.append("Database persist directory is required")
        
        # Validate external services for production
        if self.environment == 'production':
            if not self.external_services.openai_api_key:
                errors.append("OpenAI API key is required for production")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                # Handle nested dataclass objects
                result[key] = value.__dict__
            else:
                result[key] = value
        
        return result
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        try:
            config_dict = self.to_dict()
            
            # Remove sensitive data before saving
            if 'security' in config_dict:
                sensitive_fields = ['secret_key', 'jwt_secret_key', 'encryption_secret']
                for field in sensitive_fields:
                    if field in config_dict['security']:
                        config_dict['security'][field] = '[REDACTED]'
            
            if 'external_services' in config_dict:
                if 'openai_api_key' in config_dict['external_services']:
                    config_dict['external_services']['openai_api_key'] = '[REDACTED]'
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to file: {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
            raise


def get_config() -> EnvironmentConfig:
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development').lower()
    
    # Try to load from environment-specific config file
    config_files = [
        f'config/config.{env}.json',
        'config/config.json',
        f'.env.{env}',
        '.env'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            if config_file.endswith('.json'):
                config = EnvironmentConfig.from_file(config_file)
            else:
                # Load .env file
                try:
                    from dotenv import load_dotenv
                    load_dotenv(config_file)
                    logger.info(f"Loaded environment variables from: {config_file}")
                except ImportError:
                    logger.warning("python-dotenv not installed, skipping .env file")
                except Exception as e:
                    logger.warning(f"Error loading .env file {config_file}: {e}")
                
                config = EnvironmentConfig.from_env()
            break
    else:
        # No config file found, use environment variables only
        config = EnvironmentConfig.from_env()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.warning(f"Configuration validation errors: {errors}")
        if config.environment == 'production':
            raise ValueError(f"Invalid production configuration: {errors}")
    
    logger.info(f"Configuration loaded for environment: {config.environment}")
    return config


def create_config_template():
    """Create a configuration template file"""
    template_config = EnvironmentConfig()
    
    # Set example values
    template_config.environment = "production"
    template_config.security.secret_key = "your-secret-key-here-32-chars-min"
    template_config.security.jwt_secret_key = "your-jwt-secret-here-32-chars-min"
    template_config.security.encryption_secret = "your-encryption-secret-32-chars"
    template_config.external_services.openai_api_key = "your-openai-api-key"
    
    template_config.save_to_file("config/config.template.json")
    logger.info("Configuration template created: config/config.template.json")