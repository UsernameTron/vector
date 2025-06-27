#!/usr/bin/env python3
"""
Security setup script for Vector RAG Database
Generates secure configuration and demonstrates usage
"""

import os
import sys
import logging
from config.security import SecurityConfig
from security.auth import encrypt_api_key, JWTAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main setup function"""
    print("🔐 Vector RAG Database - Security Setup")
    print("=" * 50)
    
    # Generate development environment
    print("\n1. Generating secure development environment...")
    SecurityConfig.save_development_env('.env.development')
    
    # Generate configuration template
    print("\n2. Creating configuration template...")
    SecurityConfig.save_config_template('.env.template')
    
    # Demonstrate API key encryption
    print("\n3. API Key Encryption Demo...")
    demo_api_key = "sk-demo-key-1234567890abcdef"
    
    try:
        # Set a demo encryption secret
        os.environ['ENCRYPTION_SECRET'] = SecurityConfig.generate_encryption_secret()
        
        encrypted_key = encrypt_api_key(demo_api_key)
        print(f"Original API Key: {demo_api_key}")
        print(f"Encrypted API Key: {encrypted_key}")
        print("✅ API key encryption working correctly!")
        
    except Exception as e:
        print(f"❌ API key encryption failed: {e}")
    
    # Generate JWT tokens
    print("\n4. Generating demo JWT tokens...")
    try:
        # Set demo JWT secret
        os.environ['JWT_SECRET_KEY'] = SecurityConfig.generate_secure_secret_key()
        
        jwt_auth = JWTAuth()
        
        tokens = {
            'admin': jwt_auth.generate_token('admin_user', 'admin'),
            'user': jwt_auth.generate_token('regular_user', 'user'),
            'readonly': jwt_auth.generate_token('readonly_user', 'readonly')
        }
        
        print("Demo JWT Tokens:")
        for role, token in tokens.items():
            print(f"  {role}: {token[:50]}...")
        
        print("✅ JWT token generation working correctly!")
        
    except Exception as e:
        print(f"❌ JWT token generation failed: {e}")
    
    # Security configuration validation
    print("\n5. Validating security configuration...")
    config_status = SecurityConfig.validate_security_config()
    
    print("Security Configuration Status:")
    for check, status in config_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check}: {status}")
    
    # Installation instructions
    print("\n6. Next Steps:")
    print("=" * 30)
    print("1. Install security dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Copy and configure environment:")
    print("   cp .env.development .env")
    print("   # Edit .env with your actual configuration")
    print()
    print("3. Run the secure application:")
    print("   python app_secure.py")
    print()
    print("4. Test authentication:")
    print("   # Login with demo credentials:")
    print("   # admin/admin123, user/user123, readonly/readonly123")
    print()
    print("🔒 Security features implemented:")
    print("   • Input validation and sanitization")
    print("   • JWT authentication with RBAC")
    print("   • API key encryption")
    print("   • Secure CORS configuration")
    print("   • File upload validation")
    print("   • Security headers")
    print("   • Rate limiting ready")
    print()
    print("⚠️  Remember to:")
    print("   • Change default passwords in production")
    print("   • Set proper CORS origins")
    print("   • Use HTTPS in production")
    print("   • Configure proper logging")
    print("   • Set up rate limiting with Redis")


if __name__ == "__main__":
    main()