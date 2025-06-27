"""
Gunicorn configuration for production deployment
"""

import os
import multiprocessing
from pathlib import Path

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Restart workers if they haven't processed a request in this many seconds
timeout = 120
keepalive = 5

# Threading
threads = 2

# Logging
log_dir = Path("/var/log/vector-rag")
log_dir.mkdir(exist_ok=True, parents=True)

accesslog = str(log_dir / "access.log") if log_dir.exists() else "-"
errorlog = str(log_dir / "error.log") if log_dir.exists() else "-"
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'vector-rag-app'

# Server mechanics
daemon = False
pidfile = '/var/run/vector-rag/gunicorn.pid'
user = os.getenv('GUNICORN_USER', None)
group = os.getenv('GUNICORN_GROUP', None)
tmp_upload_dir = None

# SSL (uncomment and configure for HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Application-specific settings
preload_app = True
enable_stdio_inheritance = True

# Worker lifecycle hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Vector RAG Database server")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading Vector RAG Database server")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Vector RAG Database server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGTERM."""
    worker.log.info("Worker received INT or TERM signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal (pid: %s)", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug("%s %s", req.method, req.uri)

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    worker.log.debug("Response: %s", resp.status)

def child_exit(server, worker):
    """Called just after a worker has been exited, in the master process."""
    server.log.info("Worker exited (pid: %s)", worker.pid)

def worker_exit(server, worker):
    """Called just after a worker has been exited, in the worker process."""
    worker.log.info("Worker exiting (pid: %s)", worker.pid)

def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info("Number of workers changed from %s to %s", old_value, new_value)

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Shutting down Vector RAG Database server")

# Performance tuning
worker_tmp_dir = "/dev/shm"  # Use RAM disk for better performance
forwarded_allow_ips = "*"
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}