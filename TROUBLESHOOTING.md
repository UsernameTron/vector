# Troubleshooting Guide

This guide helps resolve common issues with the Vector RAG Database application.

## 🚨 Common Issues and Solutions

### 1. Application Won't Start

#### Issue: Import Errors
```
ImportError: No module named 'chromadb'
```

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# For development with all optional dependencies
pip install -r requirements.txt[dev]

# If specific dependencies fail, install individually
pip install chromadb openai flask flask-cors
```

#### Issue: Vector Database Initialization Failed
```
Vector database initialization failed: [Errno 13] Permission denied
```

**Solutions:**
1. **Permission Issues:**
   ```bash
   # Check directory permissions
   ls -la ./chroma_db/
   
   # Fix permissions
   chmod 755 ./chroma_db/
   
   # Or use alternative directory
   export CHROMA_PERSIST_DIRECTORY="$HOME/.vector_rag_db"
   ```

2. **Directory Conflicts:**
   ```bash
   # Remove corrupted database
   rm -rf ./chroma_db/
   
   # Restart application (will recreate)
   python app_robust.py
   ```

### 2. Authentication Issues

#### Issue: JWT Token Errors
```
JWT token expired / Invalid token
```

**Solutions:**
1. **Generate New Secrets:**
   ```bash
   python setup_security.py
   ```

2. **Manual Secret Generation:**
   ```python
   from config.security import SecurityConfig
   print("JWT Secret:", SecurityConfig.generate_secure_secret_key())
   ```

3. **Environment Configuration:**
   ```bash
   # Add to .env file
   JWT_SECRET_KEY=your_generated_secret_here
   JWT_EXPIRATION_HOURS=24
   ```

### 3. File Upload Problems

#### Issue: File Validation Failures
```
File validation failed: Malicious content detected
```

**Solutions:**
1. **Check File Content:**
   - Remove any script tags from documents
   - Avoid files with embedded macros
   - Use plain text formats when possible

2. **Adjust Security Settings:**
   ```python
   # In app_robust.py, modify validation call
   @enhanced_validate_file_upload(
       max_size=50*1024*1024,  # Increase size limit
       security_level='medium'  # Reduce security level
   )
   ```

3. **Whitelist Specific Files:**
   ```python
   # Add to security config
   ALLOWED_EXTENSIONS = ['txt', 'md', 'pdf', 'docx']
   ```

#### Issue: File Size Limits
```
File too large: exceeds maximum size
```

**Solutions:**
1. **Increase Limits:**
   ```python
   # In Flask config
   app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
   ```

2. **Compress Files:**
   - Use text format instead of binary
   - Remove unnecessary formatting
   - Split large documents

### 4. Search and Database Issues

#### Issue: Search Returns No Results
```
Search query returned 0 results
```

**Solutions:**
1. **Check Database Status:**
   ```bash
   curl http://localhost:8000/api/health
   ```

2. **Verify Documents:**
   ```bash
   curl http://localhost:8000/api/documents
   ```

3. **Database Recovery:**
   ```python
   # In Python console
   from vector_db_robust import VectorDatabase
   db = VectorDatabase()
   status = db.health_check()
   print(status)
   ```

#### Issue: Database Corruption
```
Database operations failed: corrupted database
```

**Solutions:**
1. **Automatic Recovery:**
   ```bash
   # Application will attempt auto-recovery
   # Check logs for recovery status
   tail -f app.log
   ```

2. **Manual Recovery:**
   ```bash
   # Backup corrupted database
   mv ./chroma_db ./chroma_db_backup_$(date +%s)
   
   # Restart application
   python app_robust.py
   ```

3. **Import Backup Data:**
   ```python
   # Restore from backup if available
   from vector_db_robust import VectorDatabase
   db = VectorDatabase()
   
   # Re-import documents
   with open('backup_documents.json', 'r') as f:
       docs = json.load(f)
       for doc in docs:
           db.add_document(doc['content'], doc['title'])
   ```

### 5. AI Agent Issues

#### Issue: Agents Not Available
```
AI agents are not available
```

**Solutions:**
1. **Check API Key:**
   ```bash
   # Verify OpenAI API key
   echo $OPENAI_API_KEY
   
   # Or check encrypted key
   echo $OPENAI_API_KEY_ENCRYPTED
   ```

2. **Manual API Key Setup:**
   ```python
   from security.auth import encrypt_api_key
   
   # Encrypt your API key
   encrypted = encrypt_api_key("your-openai-api-key")
   print(f"OPENAI_API_KEY_ENCRYPTED={encrypted}")
   ```

3. **Fallback Mode:**
   ```python
   # Disable AI agents for basic functionality
   AGENTS_AVAILABLE = False
   ```

### 6. Performance Issues

#### Issue: Slow Response Times
```
Requests timing out or very slow
```

**Solutions:**
1. **Database Optimization:**
   ```python
   # Reduce search result limits
   limit = min(limit, 10)  # Max 10 results
   
   # Add pagination
   offset = request.args.get('offset', 0, type=int)
   ```

2. **Memory Management:**
   ```bash
   # Monitor memory usage
   ps aux | grep python
   
   # Increase available memory
   export PYTHONHASHSEED=0
   ```

3. **Cleanup Resources:**
   ```python
   # Force cleanup
   from utils.error_handler import FileCleanupManager
   FileCleanupManager.cleanup_all()
   ```

### 7. Security Issues

#### Issue: CORS Errors
```
Access to fetch blocked by CORS policy
```

**Solutions:**
1. **Update CORS Origins:**
   ```bash
   # In .env file
   CORS_ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
   ```

2. **Development Mode:**
   ```python
   # For development only
   from flask_cors import CORS
   CORS(app, origins="*")  # Allow all origins
   ```

#### Issue: File Upload Security Blocks
```
File contains potentially harmful content
```

**Solutions:**
1. **Review File Content:**
   - Remove JavaScript code
   - Remove macros from Office documents
   - Use plain text formats

2. **Adjust Security Level:**
   ```python
   # Lower security for trusted sources
   security_level = 'low'  # or 'medium'
   ```

## 🔧 Diagnostic Commands

### Health Check
```bash
curl -X GET http://localhost:8000/api/health | jq '.'
```

### Application Status
```bash
curl -X GET http://localhost:8000/api/status | jq '.'
```

### Database Status
```python
from vector_db_robust import VectorDatabase
db = VectorDatabase()
print(db.get_status())
print(db.health_check())
```

### Dependency Check
```python
from utils.error_handler import DependencyManager
print(DependencyManager.get_dependency_status())
```

### Security Configuration
```python
from config.security import SecurityConfig
print(SecurityConfig.validate_security_config())
```

## 📋 Recovery Procedures

### Complete Application Reset
```bash
# 1. Stop application
pkill -f "python.*app_robust.py"

# 2. Backup data
mkdir backup_$(date +%s)
cp -r chroma_db backup_$(date +%s)/ 2>/dev/null || true

# 3. Clean temporary files
rm -rf /tmp/temp_*
rm -rf uploads/

# 4. Reset database
rm -rf chroma_db/

# 5. Regenerate security config
python setup_security.py

# 6. Restart application
python app_robust.py
```

### Dependency Reset
```bash
# Create new virtual environment
python -m venv venv_new
source venv_new/bin/activate  # or venv_new\Scripts\activate on Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test installation
python -c "import chromadb, openai, flask; print('Dependencies OK')"
```

### Data Recovery
```python
# Export current data
from vector_db_robust import VectorDatabase
import json

try:
    db = VectorDatabase()
    docs = db.get_all_documents()
    
    with open('data_export.json', 'w') as f:
        json.dump(docs, f, indent=2)
    
    print(f"Exported {len(docs)} documents")
except Exception as e:
    print(f"Export failed: {e}")
```

## 🆘 Getting Help

### Log Analysis
```bash
# View recent errors
tail -n 50 app.log | grep ERROR

# View application startup
tail -n 100 app.log | grep -A5 -B5 "initialization"

# Monitor real-time logs
tail -f app.log
```

### Environment Debugging
```bash
# Check Python environment
python --version
pip list | grep -E "(flask|chromadb|openai)"

# Check environment variables
env | grep -E "(FLASK|CHROMA|OPENAI|JWT)"

# Check file permissions
ls -la chroma_db/ uploads/ 2>/dev/null || echo "Directories not found"
```

### Performance Monitoring
```python
# Memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
print(f"CPU: {process.cpu_percent()}%")
```

## 📞 Support

If you continue experiencing issues:

1. **Check the logs** (`app.log`) for detailed error messages
2. **Verify your configuration** with the health check endpoint
3. **Try the recovery procedures** appropriate to your issue
4. **Check GitHub issues** for known problems
5. **Contact support** with:
   - Error messages from logs
   - Your configuration (without secrets)
   - Steps to reproduce the issue
   - System information (OS, Python version)

Remember to **never share secret keys or tokens** when requesting support!