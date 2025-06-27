# Multi-stage Docker build for Vector RAG Database
# Optimized for production deployment with security and performance

# Build stage
FROM python:3.11-slim as builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    pkg-config \
    git \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/
COPY requirements-prod.txt /tmp/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt && \
    pip install -r /tmp/requirements-prod.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FLASK_ENV=production \
    GUNICORN_WORKERS=4 \
    GUNICORN_TIMEOUT=120 \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libmagic1 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app /var/log/vector-rag /var/run/vector-rag && \
    chown -R appuser:appuser /app /var/log/vector-rag /var/run/vector-rag

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create data directories
RUN mkdir -p chroma_db uploads && \
    chown -R appuser:appuser chroma_db uploads

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default command
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:application"]

# Development stage
FROM production as development

# Switch back to root for dev dependencies
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    less \
    htop \
    telnet \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install pytest pytest-flask pytest-cov black flake8 mypy

# Set development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=1

# Switch back to app user
USER appuser

# Override CMD for development
CMD ["python", "app_prod.py"]