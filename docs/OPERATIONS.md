# RAG System Operations Runbook

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Initial Setup](#initial-setup)
3. [Document Ingestion](#document-ingestion)
4. [Index Management](#index-management)
5. [Evaluation](#evaluation)
6. [Deployment](#deployment)
7. [Rollback Procedures](#rollback-procedures)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 50GB+ for vector store
- Network: Stable internet for API calls

### Software
- Python 3.10+
- Docker 20.10+
- Redis (optional, for caching)
- PostgreSQL (optional, for metadata)

## Initial Setup

### 1. Clone and Install Dependencies
```bash
# Clone repository
git clone <repository-url>
cd vector-rag-database

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.template .env

# Edit configuration
vi .env

# Required variables:
# - OPENAI_API_KEY
# - FLASK_SECRET_KEY
# - JWT_SECRET_KEY
# - CHROMA_PERSIST_DIRECTORY
```

### 3. Initialize Database
```bash
# Create ChromaDB directory
mkdir -p ./chroma_db

# Initialize collections
python -c "from vector_db import VectorDatabase; db = VectorDatabase()"
```

## Document Ingestion

### Manual Ingestion (Current)
```bash
# Start Flask server
python app.py

# Add document via API
curl -X POST http://localhost:5001/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Document content here",
    "title": "Document Title",
    "source": "manual"
  }'
```

### Batch Ingestion (After Implementation)
```bash
# Ingest directory of documents
make ingest DOCS_DIR=/path/to/documents

# Or using script directly
python scripts/ingest.py \
  --input-dir /path/to/documents \
  --file-types pdf,txt,md \
  --batch-size 10 \
  --dedupe
```

## Index Management

### Build/Rebuild Index
```bash
# Full rebuild
make index-rebuild

# Incremental update
make index-update

# Optimize index
make index-optimize
```

### Backup Index
```bash
# Create backup
make backup-index BACKUP_DIR=/backups/$(date +%Y%m%d)

# Manual backup
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_db
```

### Restore Index
```bash
# Restore from backup
make restore-index BACKUP_FILE=/backups/20250906/chroma_backup.tar.gz

# Manual restore
tar -xzf chroma_backup_20250906.tar.gz
```

## Evaluation

### Run Evaluation Suite
```bash
# Full evaluation
make eval

# Specific metrics
python scripts/eval.py --metrics recall,precision,latency

# Generate report
make eval-report OUTPUT=reports/eval_$(date +%Y%m%d).json
```

### Benchmark Performance
```bash
# Run benchmarks
make bench

# Latency testing
python scripts/bench.sh --queries 1000 --concurrency 10

# Load testing
make load-test USERS=100 DURATION=300
```

## Deployment

### Development Environment
```bash
# Using Docker Compose
docker-compose -f docker-compose.dev.yml up

# Direct Python
python app.py --mode development
```

### Production Deployment
```bash
# Build Docker image
docker build -t vector-rag:latest .

# Run with Docker Compose
docker-compose up -d

# Using deployment script
./deployment/deploy.sh production

# With Gunicorn
gunicorn -c gunicorn.conf.py wsgi:app
```

### Health Checks
```bash
# Check service health
curl http://localhost:5001/health

# Detailed status
curl http://localhost:5001/api/status
```

## Rollback Procedures

### Application Rollback
```bash
# Stop current deployment
docker-compose down

# Restore previous version
git checkout <previous-tag>
docker build -t vector-rag:rollback .
docker run -d --name vector-rag-rollback vector-rag:rollback

# Verify rollback
curl http://localhost:5001/health
```

### Index Rollback
```bash
# Stop services
docker-compose stop

# Restore previous index
mv ./chroma_db ./chroma_db.failed
tar -xzf /backups/last-known-good.tar.gz

# Restart services
docker-compose start
```

## Monitoring

### Log Locations
```
Application Logs: ./logs/app.log
Access Logs: ./logs/access.log
Error Logs: ./logs/error.log
ChromaDB Logs: ./chroma_db/chroma.log
```

### Key Metrics to Monitor
```bash
# Query latency
tail -f logs/app.log | grep "Search query"

# Error rate
grep ERROR logs/app.log | wc -l

# Document count
python -c "from vector_db import VectorDatabase; print(VectorDatabase().get_stats())"
```

### Alerts Configuration
```yaml
# alerting.yml (example)
alerts:
  - name: high_latency
    condition: p99_latency > 2000ms
    action: email
  
  - name: error_rate
    condition: error_rate > 1%
    action: slack
  
  - name: disk_space
    condition: disk_usage > 80%
    action: pagerduty
```

## Troubleshooting

### Common Issues

#### ChromaDB Connection Failed
```bash
# Check if ChromaDB directory exists
ls -la ./chroma_db

# Reset ChromaDB
rm -rf ./chroma_db
mkdir ./chroma_db
python -c "from vector_db import VectorDatabase; VectorDatabase()"
```

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Restart with memory limits
docker run -m 4g vector-rag:latest

# Clear caches
python scripts/clear_cache.py
```

#### Slow Queries
```bash
# Analyze query performance
python scripts/analyze_slow_queries.py

# Optimize index
python scripts/optimize_index.py

# Increase cache size
export CACHE_SIZE=1000
```

#### API Rate Limiting
```bash
# Check current usage
python scripts/check_api_usage.py

# Implement backoff
export RETRY_DELAY=5
export MAX_RETRIES=3
```

### Emergency Procedures

#### Complete System Reset
```bash
# Stop all services
docker-compose down
pkill -f "python app"

# Backup current state
tar -czf emergency_backup_$(date +%Y%m%d_%H%M%S).tar.gz .

# Clean state
rm -rf ./chroma_db
rm -rf ./logs/*
rm -rf ./__pycache__

# Reinitialize
python scripts/init_system.py
```

#### Data Recovery
```bash
# List available backups
ls -la /backups/

# Verify backup integrity
tar -tzf /backups/backup.tar.gz

# Restore specific backup
./scripts/restore.sh /backups/backup.tar.gz
```

## Maintenance Windows

### Weekly Maintenance
```bash
# Every Sunday 2 AM
0 2 * * 0 /opt/rag/scripts/weekly_maintenance.sh
```

### Monthly Tasks
```bash
# First Sunday of month
0 3 1-7 * 0 /opt/rag/scripts/monthly_maintenance.sh
```

### Maintenance Script
```bash
#!/bin/bash
# maintenance.sh

# Backup current state
make backup

# Optimize indexes
make index-optimize

# Clean old logs
find ./logs -name "*.log" -mtime +30 -delete

# Update dependencies
pip list --outdated
pip-audit

# Generate health report
make health-report
```

## Contact Information

### Escalation Path
1. On-call Engineer: Check PagerDuty
2. Team Lead: [Contact Info]
3. Platform Team: [Contact Info]

### Resources
- Documentation: /docs
- Runbooks: /docs/runbooks
- Architecture: /docs/architecture.mmd
- API Docs: http://localhost:5001/api/docs

---
Last Updated: September 6, 2025
