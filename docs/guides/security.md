# Security Guide

## Security Best Practices

### API Key Management

#### Environment Variables
Never commit API keys to version control:

```bash
# .env file (add to .gitignore)
OPENAI_API_KEY=sk-...
API_SECRET_KEY=your-secret-key

# Load in application
from dotenv import load_dotenv
load_dotenv()
```

#### Key Rotation
- Rotate API keys regularly (every 90 days)
- Implement key versioning
- Maintain audit logs of key usage

### Authentication & Authorization

#### API Key Authentication
```python
# Middleware for API key validation
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

#### JWT Implementation
```python
# Generate JWT token
def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
```

### Input Validation

#### Request Validation
```python
# Validate and sanitize input
def validate_input(data):
    # Remove dangerous characters
    cleaned = bleach.clean(data, strip=True)
    
    # Validate against schema
    schema = {
        'agent_type': {'type': 'string', 'allowed': VALID_AGENTS},
        'query': {'type': 'string', 'maxlength': 1000}
    }
    v = Validator(schema)
    if not v.validate(data):
        raise ValidationError(v.errors)
    
    return cleaned
```

#### File Upload Security
```python
ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'json'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file):
    if file.content_length > MAX_FILE_SIZE:
        raise ValueError("File too large")
    
    # Check file signature (magic numbers)
    file_header = file.read(512)
    file.seek(0)
    
    if not is_valid_file_type(file_header):
        raise ValueError("Invalid file type")
```

### Data Protection

#### Encryption at Rest
```python
# Encrypt sensitive data before storage
from cryptography.fernet import Fernet

def encrypt_data(data):
    key = os.environ.get('ENCRYPTION_KEY')
    f = Fernet(key)
    return f.encrypt(data.encode())

def decrypt_data(encrypted_data):
    key = os.environ.get('ENCRYPTION_KEY')
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode()
```

#### Data Sanitization
```python
# Remove PII from documents
def sanitize_document(content):
    # Remove email addresses
    content = re.sub(r'\S+@\S+', '[EMAIL]', content)
    
    # Remove phone numbers
    content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', content)
    
    # Remove SSN
    content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', content)
    
    return content
```

### Network Security

#### HTTPS Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # Strong SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

#### CORS Configuration
```python
from flask_cors import CORS

# Restrict CORS to specific origins
CORS(app, origins=[
    "https://your-domain.com",
    "https://app.your-domain.com"
])
```

#### Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: get_remote_address(),
    default_limits=["100 per minute", "1000 per hour"]
)

@app.route('/api/chat')
@limiter.limit("10 per minute")
def chat():
    # Handle chat request
    pass
```

### Logging & Monitoring

#### Security Logging
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure security logger
security_logger = logging.getLogger('security')
handler = RotatingFileHandler('security.log', maxBytes=10485760, backupCount=10)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
security_logger.addHandler(handler)

# Log security events
def log_security_event(event_type, details):
    security_logger.warning(f"Security Event: {event_type} - {details}")
```

#### Audit Trail
```python
def audit_log(user_id, action, resource, result):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'action': action,
        'resource': resource,
        'result': result,
        'ip_address': request.remote_addr
    }
    # Store in database or log file
    audit_logger.info(json.dumps(log_entry))
```

### Vulnerability Prevention

#### SQL Injection Prevention
Use parameterized queries:
```python
# Safe query
cursor.execute(
    "SELECT * FROM documents WHERE id = %s",
    (document_id,)
)

# Never do this
cursor.execute(
    f"SELECT * FROM documents WHERE id = '{document_id}'"
)
```

#### XSS Prevention
```python
from markupsafe import escape

@app.route('/display')
def display():
    user_input = request.args.get('input', '')
    # Escape user input before rendering
    safe_input = escape(user_input)
    return render_template('display.html', content=safe_input)
```

#### CSRF Protection
```python
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
```

### Dependency Security

#### Regular Updates
```bash
# Check for security vulnerabilities
pip install safety
safety check

# Update dependencies
pip install --upgrade -r requirements.txt
```

#### Dependency Scanning
```yaml
# GitHub Actions workflow
name: Security Scan
on: [push, pull_request]
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run security scan
        run: |
          pip install safety
          safety check --json
```

### Deployment Security

#### Environment Separation
```python
# Use different configs for different environments
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    SSL_REDIRECT = True
```

#### Secrets Management
```bash
# Use secret management services
# AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id prod/vectorrag/api-keys

# HashiCorp Vault
vault kv get secret/vectorrag/api-keys
```

### Security Checklist

#### Pre-Deployment
- [ ] All secrets in environment variables
- [ ] Input validation implemented
- [ ] Authentication required for sensitive endpoints
- [ ] Rate limiting configured
- [ ] HTTPS enabled
- [ ] Security headers configured
- [ ] Dependencies updated
- [ ] Security scan passed

#### Post-Deployment
- [ ] Monitor security logs
- [ ] Regular security audits
- [ ] Penetration testing
- [ ] Incident response plan
- [ ] Backup and recovery tested
- [ ] Access controls reviewed

### Incident Response

#### Response Plan
1. **Detect**: Monitor logs and alerts
2. **Contain**: Isolate affected systems
3. **Investigate**: Determine scope and impact
4. **Remediate**: Fix vulnerabilities
5. **Recover**: Restore normal operations
6. **Learn**: Document and improve

#### Emergency Contacts
- Security Team: security@your-domain.com
- On-call Engineer: +1-XXX-XXX-XXXX
- Management: management@your-domain.com

### Compliance

#### GDPR Compliance
- Data minimization
- Right to erasure
- Data portability
- Privacy by design

#### Data Retention
```python
# Automatic data cleanup
def cleanup_old_data():
    cutoff_date = datetime.now() - timedelta(days=90)
    Document.query.filter(Document.created_at < cutoff_date).delete()
    db.session.commit()
```