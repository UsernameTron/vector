#!/bin/bash

# Vector RAG Database Deployment Script
# Production deployment automation

set -euo pipefail

# Configuration
PROJECT_NAME="vector-rag-database"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"
BACKUP_DIR="/opt/backups/vector-rag"
LOG_FILE="/var/log/deploy-vector-rag.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check if environment file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        warning "Environment file $ENV_FILE not found, creating from template..."
        if [[ -f ".env.template" ]]; then
            cp .env.template "$ENV_FILE"
            warning "Please edit $ENV_FILE with your configuration"
        else
            error "No environment template found"
        fi
    fi
    
    success "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    sudo mkdir -p "$BACKUP_DIR"
    sudo mkdir -p /var/log/vector-rag
    sudo mkdir -p /var/run/vector-rag
    
    # Set permissions
    sudo chown -R $USER:$USER "$BACKUP_DIR" 2>/dev/null || true
    
    success "Directories created"
}

# Backup existing data
backup_data() {
    log "Creating backup..."
    
    BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup Docker volumes if they exist
    if docker volume ls | grep -q vector-rag; then
        log "Backing up Docker volumes..."
        
        # Create temporary container to backup volumes
        docker run --rm \
            -v vector-rag_vector_data:/source/data \
            -v vector-rag_vector_logs:/source/logs \
            -v "$BACKUP_PATH":/backup \
            alpine:latest \
            sh -c "cd /source && tar czf /backup/volumes_$BACKUP_TIMESTAMP.tar.gz ."
        
        success "Data backup completed: $BACKUP_PATH"
    else
        log "No existing volumes found, skipping backup"
    fi
    
    # Keep only last 5 backups
    find "$BACKUP_DIR" -name "backup_*" -type d | sort | head -n -5 | xargs rm -rf 2>/dev/null || true
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" pull
    else
        docker compose -f "$COMPOSE_FILE" pull
    fi
    
    success "Images pulled successfully"
}

# Deploy application
deploy() {
    log "Deploying application..."
    
    # Stop existing containers
    log "Stopping existing containers..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down --remove-orphans || true
    else
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down --remove-orphans || true
    fi
    
    # Start new containers
    log "Starting new containers..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    else
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    fi
    
    success "Application deployed successfully"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Wait for application to start
    sleep 30
    
    # Check if containers are running
    if command -v docker-compose &> /dev/null; then
        RUNNING_CONTAINERS=$(docker-compose -f "$COMPOSE_FILE" ps --services --filter status=running)
    else
        RUNNING_CONTAINERS=$(docker compose -f "$COMPOSE_FILE" ps --services --filter status=running)
    fi
    
    if [[ -z "$RUNNING_CONTAINERS" ]]; then
        error "No containers are running"
    fi
    
    # Check application health endpoint
    for i in {1..10}; do
        if curl -f http://localhost:8000/health/live &> /dev/null; then
            success "Application health check passed"
            return 0
        fi
        log "Health check attempt $i/10 failed, retrying..."
        sleep 10
    done
    
    error "Application health check failed after 10 attempts"
}

# View logs
view_logs() {
    log "Recent application logs:"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" logs --tail=50 vector-rag-app
    else
        docker compose -f "$COMPOSE_FILE" logs --tail=50 vector-rag-app
    fi
}

# Cleanup old images
cleanup() {
    log "Cleaning up old Docker images..."
    
    # Remove dangling images
    docker image prune -f &> /dev/null || true
    
    # Remove old images (keep last 3)
    OLD_IMAGES=$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | grep "$PROJECT_NAME" | tail -n +4 | awk '{print $1}')
    if [[ -n "$OLD_IMAGES" ]]; then
        echo "$OLD_IMAGES" | xargs docker rmi &> /dev/null || true
    fi
    
    success "Cleanup completed"
}

# Show usage
usage() {
    echo "Vector RAG Database Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  deploy      Full deployment (default)"
    echo "  start       Start services"
    echo "  stop        Stop services"
    echo "  restart     Restart services"
    echo "  status      Show service status"
    echo "  logs        Show application logs"
    echo "  backup      Create data backup"
    echo "  health      Run health check"
    echo "  cleanup     Clean up old images"
    echo "  help        Show this help"
    echo
    echo "Examples:"
    echo "  $0 deploy           # Full deployment"
    echo "  $0 logs             # View logs"
    echo "  $0 health           # Check health"
}

# Main deployment function
main_deploy() {
    log "Starting Vector RAG Database deployment..."
    
    check_prerequisites
    create_directories
    backup_data
    pull_images
    deploy
    health_check
    cleanup
    
    success "Deployment completed successfully!"
    success "Application is available at: http://localhost:8000"
    success "Health check: http://localhost:8000/health"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main_deploy
        ;;
    "start")
        log "Starting services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
        else
            docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
        fi
        success "Services started"
        ;;
    "stop")
        log "Stopping services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" down
        else
            docker compose -f "$COMPOSE_FILE" down
        fi
        success "Services stopped"
        ;;
    "restart")
        log "Restarting services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
        else
            docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
        fi
        success "Services restarted"
        ;;
    "status")
        log "Service status:"
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" ps
        else
            docker compose -f "$COMPOSE_FILE" ps
        fi
        ;;
    "logs")
        view_logs
        ;;
    "backup")
        backup_data
        ;;
    "health")
        health_check
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        usage
        ;;
    *)
        error "Unknown command: $1"
        usage
        exit 1
        ;;
esac