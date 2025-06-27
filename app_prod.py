"""
Production Vector RAG Database Application
Production-ready application with comprehensive monitoring, health checks, and graceful shutdown
"""

import asyncio
import logging
import sys
import os
from flask import Flask

# Configuration
from config.environment import get_config, EnvironmentConfig

# Infrastructure imports
from src.infrastructure.container import ServiceContainer, get_container, set_container
from src.infrastructure.repositories.chroma_document_repository import ChromaDocumentRepository, ChromaVectorStore
from src.infrastructure.services.logging_service import LoggingService
from src.infrastructure.services.event_publisher import EventPublisher
from src.infrastructure.monitoring.health_monitor import HealthMonitor
from src.infrastructure.shutdown import get_shutdown_manager, setup_production_cleanup, register_resource

# Domain interfaces
from src.domain.interfaces import IDocumentRepository, IVectorStore, ILoggingService, IEventPublisher, IHealthMonitor

# Application services
from src.application.services.document_service import DocumentService

# Presentation layer
from src.presentation.controllers.document_controller import document_bp
from src.presentation.controllers.health_controller import health_bp
from src.presentation.responses import create_response_builder


def create_app() -> Flask:
    """Production application factory function"""
    config = get_config()
    
    # Setup logging
    log_level = getattr(logging, config.logging.level.upper())
    logging.basicConfig(level=log_level, format=config.logging.format)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Vector RAG Database - Production")
    
    # Initialize Flask app
    app = Flask(__name__)
    app.config.update({
        'SECRET_KEY': config.security.secret_key,
        'DEBUG': config.debug,
        'TESTING': config.testing
    })
    
    # Register blueprints
    app.register_blueprint(document_bp)
    app.register_blueprint(health_bp)
    
    return app


if __name__ == '__main__':
    app = create_app()
    config = get_config()
    app.run(host=config.host, port=config.port, debug=config.debug)