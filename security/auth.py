"""
Authentication and authorization middleware for Flask application
Includes JWT authentication, role-based access control, and API key encryption
"""

import jwt
import secrets
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from flask import request, jsonify, current_app
from functools import wraps
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Secure API key encryption and decryption using environment-based secrets"""
    
    @staticmethod
    def _get_encryption_key() -> bytes:
        """Generate encryption key from environment secret"""
        secret = os.getenv('ENCRYPTION_SECRET')
        if not secret:
            raise ValueError("ENCRYPTION_SECRET environment variable not set")
        
        # Use PBKDF2 to derive a key from the secret
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'vector_rag_salt',  # In production, use a random salt stored securely
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
        return key
    
    @staticmethod
    def encrypt_api_key(api_key: str) -> str:
        """Encrypt API key for secure storage"""
        try:
            key = APIKeyManager._get_encryption_key()
            fernet = Fernet(key)
            encrypted_key = fernet.encrypt(api_key.encode())
            return base64.urlsafe_b64encode(encrypted_key).decode()
        except Exception as e:
            logger.error(f"API key encryption failed: {e}")
            raise
    
    @staticmethod
    def decrypt_api_key(encrypted_api_key: str) -> str:
        """Decrypt API key for use"""
        try:
            key = APIKeyManager._get_encryption_key()
            fernet = Fernet(key)
            encrypted_data = base64.urlsafe_b64decode(encrypted_api_key.encode())
            decrypted_key = fernet.decrypt(encrypted_data)
            return decrypted_key.decode()
        except Exception as e:
            logger.error(f"API key decryption failed: {e}")
            raise


def encrypt_api_key(api_key: str) -> str:
    """Public function to encrypt API key"""
    return APIKeyManager.encrypt_api_key(api_key)


def decrypt_api_key(encrypted_api_key: str) -> str:
    """Public function to decrypt API key"""
    return APIKeyManager.decrypt_api_key(encrypted_api_key)


class JWTAuth:
    """JWT-based authentication with role-based access control"""
    
    # User roles and their permissions
    ROLES = {
        'admin': ['read', 'write', 'delete', 'manage_users'],
        'user': ['read', 'write'],
        'readonly': ['read']
    }
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize JWT authentication with app"""
        self.app = app
        # Set default secret key if not provided
        if not app.config.get('JWT_SECRET_KEY'):
            app.config['JWT_SECRET_KEY'] = self.generate_secure_secret()
        
        if not app.config.get('JWT_EXPIRATION_HOURS'):
            app.config['JWT_EXPIRATION_HOURS'] = 24
    
    @staticmethod
    def generate_secure_secret() -> str:
        """Generate a cryptographically secure secret key"""
        return secrets.token_urlsafe(32)
    
    def generate_token(self, user_id: str, role: str = 'user', permissions: List[str] = None) -> str:
        """Generate JWT token for user"""
        if role not in self.ROLES:
            raise ValueError(f"Invalid role: {role}")
        
        # Use role permissions if custom permissions not provided
        if permissions is None:
            permissions = self.ROLES[role]
        
        payload = {
            'user_id': user_id,
            'role': role,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.app.config['JWT_EXPIRATION_HOURS'])
        }
        
        token = jwt.encode(
            payload,
            self.app.config['JWT_SECRET_KEY'],
            algorithm='HS256'
        )
        
        logger.info(f"Generated JWT token for user {user_id} with role {role}")
        return token
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(
                token,
                self.app.config['JWT_SECRET_KEY'],
                algorithms=['HS256']
            )
            return True, payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return False, {'error': 'Token expired'}
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return False, {'error': 'Invalid token'}
    
    def has_permission(self, payload: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission"""
        user_permissions = payload.get('permissions', [])
        return required_permission in user_permissions
    
    def is_admin(self, payload: Dict[str, Any]) -> bool:
        """Check if user has admin role"""
        return payload.get('role') == 'admin'


def require_auth(permission: str = None, admin_only: bool = False):
    """Decorator for requiring authentication and authorization"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get token from Authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'Authorization header required'}), 401
            
            try:
                # Extract token from "Bearer <token>" format
                token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else auth_header
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
            
            # Initialize JWT auth
            jwt_auth = JWTAuth(current_app)
            
            # Verify token
            is_valid, result = jwt_auth.verify_token(token)
            if not is_valid:
                return jsonify({'error': result.get('error', 'Authentication failed')}), 401
            
            payload = result
            
            # Check admin requirement
            if admin_only and not jwt_auth.is_admin(payload):
                return jsonify({'error': 'Admin access required'}), 403
            
            # Check specific permission
            if permission and not jwt_auth.has_permission(payload, permission):
                return jsonify({'error': f'Permission required: {permission}'}), 403
            
            # Add user info to request context
            request.current_user = {
                'user_id': payload['user_id'],
                'role': payload['role'],
                'permissions': payload['permissions']
            }
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def create_default_users() -> Dict[str, str]:
    """Create default user tokens for development/testing"""
    jwt_auth = JWTAuth()
    
    # Generate secure secret for app config
    secret_key = jwt_auth.generate_secure_secret()
    
    users = {
        'admin': jwt_auth.generate_token('admin_user', 'admin'),
        'user': jwt_auth.generate_token('regular_user', 'user'),
        'readonly': jwt_auth.generate_token('readonly_user', 'readonly')
    }
    
    return {
        'secret_key': secret_key,
        'tokens': users
    }


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        salt, password_hash = hashed.split(':')
        computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return computed_hash == password_hash
    except ValueError:
        return False