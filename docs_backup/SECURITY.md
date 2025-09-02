# Security Implementation Guide

This document outlines the comprehensive security features implemented in the Vector RAG Database application.

## 🔐 Security Features

### 1. Input Validation & Sanitization (`security/validators.py`)
- **HTML Sanitization**: Prevents XSS attacks using `bleach` library
- **SQL Injection Detection**: Pattern-based detection of SQL injection attempts
- **Input Length Validation**: Configurable min/max length checks
- **JSON Payload Validation**: Comprehensive JSON input validation
- **Query Parameter Sanitization**: URL parameter validation and cleaning

**Usage:**
```python
from security import validate_input

@app.route('/api/endpoint', methods=['POST'])
@validate_input(required_json_fields=['field1', 'field2'])
def secure_endpoint():
    # Your endpoint logic here
    pass
```

### 2. JWT Authentication & RBAC (`security/auth.py`)
- **Role-Based Access Control**: Admin, User, ReadOnly roles
- **JWT Token Management**: Secure token generation and validation
- **Permission System**: Granular permissions (read, write, delete, manage_users)
- **Token Expiration**: Configurable token lifetime
- **Secure Secret Generation**: Cryptographically secure secret keys

**Roles & Permissions:**
- `admin`: read, write, delete, manage_users
- `user`: read, write
- `readonly`: read

**Usage:**
```python
from security import require_auth

@app.route('/api/secure', methods=['GET'])
@require_auth('read')
def secure_endpoint():
    user = request.current_user  # Access current user info
    return jsonify({'user': user})

@app.route('/api/admin', methods=['GET'])
@require_auth(admin_only=True)
def admin_endpoint():
    # Admin-only functionality
    pass
```

### 3. API Key Encryption (`security/auth.py`)
- **Environment-Based Encryption**: Uses `ENCRYPTION_SECRET` from environment
- **PBKDF2 Key Derivation**: Secure key derivation from secrets
- **Fernet Encryption**: Symmetric encryption for API keys
- **Base64 Encoding**: Safe storage format

**Usage:**
```python
from security import encrypt_api_key, decrypt_api_key

# Encrypt API key for storage
encrypted = encrypt_api_key("your-api-key")

# Decrypt for use
api_key = decrypt_api_key(encrypted)
```

### 4. File Upload Validation (`security/file_validator.py`)
- **File Type Validation**: MIME type checking with `python-magic`
- **Size Limits**: Configurable file size restrictions
- **Filename Security**: Path traversal prevention
- **Content Scanning**: Malicious pattern detection
- **Virus Signature Detection**: Basic malware signature checking
- **Hash Calculation**: SHA-256 file hashing

**Supported File Types:**
- Text: txt, md, rtf
- Documents: pdf, doc, docx, odt
- Data: json, csv, xml

**Usage:**
```python
from security import validate_file_upload

@app.route('/upload', methods=['POST'])
@validate_file_upload(max_size=10*1024*1024)  # 10MB limit
def upload_file():
    validation_result = request.file_validation
    # File is validated and safe to process
    pass
```

### 5. CORS Configuration (`security/cors_config.py`)
- **Environment-Based Origins**: Configure allowed origins via `CORS_ALLOWED_ORIGINS`
- **Secure Defaults**: Development and production configurations
- **Credential Support**: Controlled credential handling
- **Method Restrictions**: Limited to necessary HTTP methods

**Configuration:**
```bash
# Environment variable
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### 6. Security Configuration (`config/security.py`)
- **Secure Secret Generation**: Cryptographically secure keys
- **Flask Security Headers**: XSS, CSRF, clickjacking protection
- **Session Security**: Secure session configuration
- **Configuration Validation**: Security setting verification

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Secure Configuration
```bash
python setup_security.py
```

### 3. Configure Environment
```bash
cp .env.development .env
# Edit .env with your settings
```

### 4. Run Secure Application
```bash
python app_secure.py
```

## 🔑 Authentication Flow

### 1. Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

Response:
```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": "admin",
  "role": "admin",
  "expires_in": 86400
}
```

### 2. Authenticated Request
```bash
curl -X GET http://localhost:8000/api/agents \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🛡️ Security Headers

The application automatically adds these security headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'`

## 🔧 Configuration

### Environment Variables

```bash
# Flask Security
FLASK_SECRET_KEY=your-secure-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
ENCRYPTION_SECRET=your-encryption-secret

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# JWT Settings
JWT_EXPIRATION_HOURS=24

# File Upload
MAX_UPLOAD_SIZE=52428800  # 50MB in bytes

# Environment
FLASK_ENV=production  # or development
```

### Security Validation

Check your security configuration:
```python
from config.security import SecurityConfig

status = SecurityConfig.validate_security_config()
print(status)
```

## 🎯 Production Checklist

- [ ] Generate strong secret keys
- [ ] Configure proper CORS origins
- [ ] Enable HTTPS/TLS
- [ ] Set up rate limiting
- [ ] Configure logging and monitoring
- [ ] Change default passwords
- [ ] Set up proper user management
- [ ] Configure database security
- [ ] Set up backup encryption
- [ ] Enable audit logging

## 🚨 Security Considerations

1. **Default Credentials**: Change default login credentials immediately
2. **HTTPS Only**: Always use HTTPS in production
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Logging**: Monitor and log security events
5. **Updates**: Keep dependencies updated for security patches
6. **Secrets Management**: Use proper secrets management in production
7. **Database Security**: Secure your vector database storage
8. **Backup Security**: Encrypt backups and limit access

## 🐛 Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:
1. Do not create public issues for security vulnerabilities
2. Contact the maintainers privately
3. Provide detailed information about the vulnerability
4. Allow time for the issue to be fixed before public disclosure

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Flask Security Best Practices](https://flask.palletsprojects.com/en/2.0.x/security/)
- [JWT Security Best Practices](https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/)
- [Python Security Guidelines](https://python.org/dev/security/)