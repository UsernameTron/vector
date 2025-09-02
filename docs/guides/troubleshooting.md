# Troubleshooting Guide

## Common Issues and Solutions

### Application Won't Start

#### ImportError: No module named 'chromadb'

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### Port Already in Use

```bash
# Find process using port
lsof -i :5001  # Linux/Mac
netstat -ano | findstr :5001  # Windows

# Kill the process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

### Vector Database Issues

#### Permission Denied Error

```bash
# Fix permissions
chmod -R 755 ./chroma_db/

# Or delete and recreate
rm -rf ./chroma_db/
python app_unified.py --mode production
```

#### Database Corruption

```bash
# Backup existing data if needed
cp -r chroma_db chroma_db_backup

# Remove corrupted database
rm -rf chroma_db/

# Restart application
python app_unified.py --mode production
```

### OpenAI API Issues

#### Invalid API Key

```bash
# Verify .env file
cat .env | grep OPENAI_API_KEY

# Test API key
python -c "import openai; openai.api_key='your_key'; print(openai.Model.list())"
```

#### Rate Limiting

- Implement exponential backoff
- Use caching for repeated queries
- Consider upgrading OpenAI plan

### File Upload Problems

#### File Too Large

```bash
# Increase upload limit in app configuration
# Edit app_unified.py or set environment variable
export MAX_CONTENT_LENGTH=16777216  # 16MB
```

#### Unsupported File Format

Supported formats:
- CSV, TSV, TXT
- PDF (with PyPDF2)
- JSON
- Excel (with openpyxl)

### Agent Response Issues

#### Agent Not Responding

1. Check OpenAI API status
2. Verify network connectivity
3. Check application logs
4. Restart the application

#### Poor Response Quality

- Improve document quality
- Add more context to queries
- Use appropriate agent for task
- Clear conversation history if needed

### Performance Issues

#### Slow Response Times

```bash
# Check system resources
top  # Linux/Mac
taskmgr  # Windows

# Optimize ChromaDB
# Reduce collection size
# Index optimization
```

#### High Memory Usage

```bash
# Monitor memory
ps aux | grep python

# Limit document batch size
# Implement pagination
# Regular garbage collection
```

### Frontend Issues

#### Page Not Loading

1. Clear browser cache
2. Check console for errors (F12)
3. Verify server is running
4. Check CORS configuration

#### WebSocket Connection Failed

```bash
# Check firewall settings
# Verify WebSocket support
# Check proxy configuration
```

### Docker Issues

#### Container Won't Start

```bash
# Check logs
docker logs vector-rag

# Verify environment variables
docker exec vector-rag env

# Rebuild image
docker build -t vector-rag-db . --no-cache
```

### SSL/HTTPS Issues

#### Certificate Errors

```bash
# Update certificates
pip install --upgrade certifi

# For macOS
brew install ca-certificates
```

## Debugging Tips

### Enable Debug Mode

```bash
# Set environment variable
export FLASK_ENV=development
export LOG_LEVEL=DEBUG

# Or use development mode
python app_unified.py --mode development
```

### Check Logs

```bash
# Application logs
tail -f app.log

# System logs
journalctl -u vector-rag -f  # systemd
docker logs -f vector-rag  # Docker
```

### Test Endpoints

```bash
# Health check
curl http://localhost:5001/health

# List agents
curl http://localhost:5001/api/agents

# Test chat endpoint
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"agent_type":"research","query":"test"}'
```

## Getting Help

### Log Collection Script

```bash
#!/bin/bash
# collect_logs.sh
mkdir -p debug_logs
cp app.log debug_logs/
cp .env debug_logs/env_sample.txt
python --version > debug_logs/python_version.txt
pip freeze > debug_logs/requirements_installed.txt
tar -czf debug_logs.tar.gz debug_logs/
```

### Reporting Issues

When reporting issues, include:
1. Error messages and stack traces
2. Steps to reproduce
3. Environment details (OS, Python version)
4. Configuration files (without secrets)
5. Recent changes made

### Community Support

- GitHub Issues: Report bugs and request features
- Documentation: Check latest docs for updates
- FAQ: Common questions and answers

## Recovery Procedures

### Full System Reset

```bash
# Backup important data
cp -r chroma_db chroma_db_backup
cp .env .env.backup

# Clean installation
rm -rf venv chroma_db __pycache__
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Restore configuration
cp .env.backup .env

# Start fresh
python app_unified.py --mode production
```

### Database Recovery

```bash
# From backup
cp -r backups/chroma_db_20240101 chroma_db

# Rebuild indexes
python scripts/rebuild_indexes.py
```