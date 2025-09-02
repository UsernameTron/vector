# Production Deployment Guide

## Deployment Options

### 1. Docker Deployment (Recommended)

```bash
# Build Docker image
docker build -t vector-rag-db .

# Run container
docker run -d \
  -p 5001:5001 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --name vector-rag \
  vector-rag-db
```

### 2. Systemd Service (Linux)

Create `/etc/systemd/system/vector-rag.service`:

```ini
[Unit]
Description=Vector RAG Database
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/vector-rag
Environment="PATH=/opt/vector-rag/venv/bin"
ExecStart=/opt/vector-rag/venv/bin/python app_unified.py --mode production
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable vector-rag
sudo systemctl start vector-rag
```

### 3. Gunicorn with Nginx

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -c gunicorn.conf.py app_unified:app
```

Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
FLASK_ENV=production
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection
LOG_LEVEL=INFO
```

## Security Checklist

- [ ] Set strong OPENAI_API_KEY
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable CORS restrictions
- [ ] Implement authentication
- [ ] Regular security updates
- [ ] Monitor logs

## Performance Tuning

### Gunicorn Workers
```python
# gunicorn.conf.py
workers = 4  # 2-4 x CPU cores
worker_class = 'sync'
worker_connections = 1000
```

### Database Optimization
- Periodic vector index optimization
- Regular cleanup of old documents
- Monitor ChromaDB performance

## Monitoring

### Health Checks
```bash
# Automated health check
curl http://localhost:5001/health
```

### Logging
```bash
# View logs
journalctl -u vector-rag -f
```

## Backup Strategy

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups/vector-rag"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
cp -r chroma_db "$BACKUP_DIR/chroma_db_$DATE"

# Backup config
cp .env "$BACKUP_DIR/env_$DATE"
```

## Scaling Considerations

- Use external ChromaDB instance for multiple app servers
- Implement Redis for caching
- Use CDN for static assets
- Consider load balancer for high traffic