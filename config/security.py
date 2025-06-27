"""
Security configuration for Flask application
Secure secret key generation and management
"""

import os
import secrets
import json
from typing import Dict, Any
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration management"""
    
    @staticmethod
    def generate_secure_secret_key() -> str:
        """Generate a cryptographically secure secret key"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_encryption_secret() -> str:
        """Generate encryption secret for API key encryption"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def get_flask_config() -> Dict[str, Any]:
        """Get Flask security configuration"""
        config = {
            # Session configuration
            'SECRET_KEY': os.getenv('FLASK_SECRET_KEY') or SecurityConfig.generate_secure_secret_key(),
            'SESSION_COOKIE_SECURE': os.getenv('FLASK_ENV') == 'production',
            'SESSION_COOKIE_HTTPONLY': True,
            'SESSION_COOKIE_SAMESITE': 'Lax',
            'PERMANENT_SESSION_LIFETIME': timedelta(hours=24),
            
            # Security headers
            'SEND_FILE_MAX_AGE_DEFAULT': timedelta(hours=1),
            
            # Upload configuration
            'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB max file upload
            
            # JWT configuration
            'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY') or SecurityConfig.generate_secure_secret_key(),
            'JWT_EXPIRATION_HOURS': int(os.getenv('JWT_EXPIRATION_HOURS', '24')),
            
            # Rate limiting
            'RATELIMIT_STORAGE_URL': os.getenv('REDIS_URL', 'memory://'),
            'RATELIMIT_DEFAULT': '100 per hour',
            
            # CSRF protection
            'WTF_CSRF_ENABLED': True,
            'WTF_CSRF_TIME_LIMIT': 3600,
        }
        
        return config
    
    @staticmethod
    def save_config_template(file_path: str = '.env.template'):
        """Save environment configuration template"""
        template = """# Flask Security Configuration
FLASK_SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_here
ENCRYPTION_SECRET=your_encryption_secret_here

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection

# OpenAI Configuration (encrypted)
OPENAI_API_KEY_ENCRYPTED=your_encrypted_openai_key_here

# JWT Configuration
JWT_EXPIRATION_HOURS=24

# Environment
FLASK_ENV=development

# Rate Limiting
REDIS_URL=redis://localhost:6379/0
"""
        
        try:
            with open(file_path, 'w') as f:
                f.write(template)
            logger.info(f"Configuration template saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save config template: {e}")
    
    @staticmethod
    def generate_development_config() -> Dict[str, str]:
        """Generate development configuration with secure defaults"""
        config = {
            'FLASK_SECRET_KEY': SecurityConfig.generate_secure_secret_key(),
            'JWT_SECRET_KEY': SecurityConfig.generate_secure_secret_key(),
            'ENCRYPTION_SECRET': SecurityConfig.generate_encryption_secret(),
            'CORS_ALLOWED_ORIGINS': 'http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000',
            'FLASK_ENV': 'development',
            'JWT_EXPIRATION_HOURS': '24'
        }
        
        return config
    
    @staticmethod
    def save_development_env(file_path: str = '.env.development'):
        """Save development environment file with secure secrets"""
        config = SecurityConfig.generate_development_config()
        
        env_content = ""
        for key, value in config.items():
            env_content += f"{key}={value}\n"
        
        try:
            with open(file_path, 'w') as f:
                f.write(env_content)
            logger.info(f"Development environment file saved to {file_path}")
            logger.warning("Remember to add .env.development to .gitignore!")
        except Exception as e:
            logger.error(f"Failed to save development config: {e}")
    
    @staticmethod
    def validate_security_config() -> Dict[str, bool]:
        """Validate current security configuration"""
        checks = {
            'flask_secret_key_set': bool(os.getenv('FLASK_SECRET_KEY')),
            'jwt_secret_key_set': bool(os.getenv('JWT_SECRET_KEY')),
            'encryption_secret_set': bool(os.getenv('ENCRYPTION_SECRET')),
            'cors_origins_configured': bool(os.getenv('CORS_ALLOWED_ORIGINS')),
            'production_mode': os.getenv('FLASK_ENV') == 'production',
        }
        
        # Check secret key strength
        flask_secret = os.getenv('FLASK_SECRET_KEY', '')
        checks['flask_secret_strong'] = len(flask_secret) >= 32
        
        jwt_secret = os.getenv('JWT_SECRET_KEY', '')
        checks['jwt_secret_strong'] = len(jwt_secret) >= 32
        
        encryption_secret = os.getenv('ENCRYPTION_SECRET', '')
        checks['encryption_secret_strong'] = len(encryption_secret) >= 32
        
        return checks