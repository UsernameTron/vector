"""
Clean Architecture Vector RAG Database Application
Main application entry point with dependency injection and layered architecture
"""

import asyncio
import logging
import sys
from flask import Flask
from typing import Dict, Any

# Infrastructure imports
from src.infrastructure.container import ServiceContainer, get_container, set_container
from src.infrastructure.repositories.chroma_document_repository import (
    ChromaDocumentRepository, ChromaVectorStore
)
from src.infrastructure.services.logging_service import LoggingService
from src.infrastructure.services.event_publisher import EventPublisher

# Domain interfaces
from src.domain.interfaces import (
    IDocumentRepository, IVectorStore, ILoggingService, IEventPublisher,
    IFileRepository, IFileValidationService, ICacheService
)

# Application services
from src.application.services.document_service import DocumentService
from src.application.services.ai_agent_service import AIAgentService
from src.application.services.file_processing_service import FileProcessingService

# Presentation layer
from src.presentation.controllers.document_controller import document_bp
from src.presentation.responses import create_response_builder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ApplicationBootstrapper:
    """Bootstraps the application with dependency injection"""
    
    def __init__(self):
        self.container = ServiceContainer()
        self.app = None
    
    def configure_services(self):
        """Configure dependency injection container"""
        logger.info("Configuring services...")
        
        # Infrastructure services
        self.container.register_singleton(ILoggingService, LoggingService)
        self.container.register_singleton(IEventPublisher, EventPublisher)
        
        # Data access
        self.container.register_singleton(IVectorStore, ChromaVectorStore)
        self.container.register_singleton(IDocumentRepository, ChromaDocumentRepository)
        
        # Application services
        self.container.register_scoped(DocumentService)
        
        # Validate all registrations
        issues = self.container.validate_registrations()
        if issues:
            logger.warning(f"Service registration issues: {issues}")
        
        logger.info("Services configured successfully")
    
    def create_flask_app(self) -> Flask:
        """Create and configure Flask application"""
        logger.info("Creating Flask application...")
        
        app = Flask(__name__)
        
        # Configure Flask
        app.config.update({
            'SECRET_KEY': 'your-secret-key-here',  # Should come from environment
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': True
        })
        
        # Register error handlers
        self._register_error_handlers(app)
        
        # Register blueprints
        app.register_blueprint(document_bp)
        
        # Add health check endpoint
        self._add_health_check(app)
        
        # Add middleware
        self._add_middleware(app)
        
        self.app = app
        logger.info("Flask application created successfully")
        return app
    
    def _register_error_handlers(self, app: Flask):
        """Register global error handlers"""
        
        @app.errorhandler(404)
        def not_found(error):
            response_builder = create_response_builder()
            response = response_builder.not_found("Endpoint", request.path if 'request' in globals() else "unknown")
            return response.to_dict(), 404
        
        @app.errorhandler(405)
        def method_not_allowed(error):
            response_builder = create_response_builder()
            response = response_builder.error(
                error_detail={
                    "code": "METHOD_NOT_ALLOWED",
                    "message": "Method not allowed for this endpoint"
                }
            )
            return response.to_dict(), 405
        
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            response_builder = create_response_builder()
            response = response_builder.system_error("Internal server error")
            return response.to_dict(), 500
    
    def _add_health_check(self, app: Flask):
        """Add health check endpoint"""
        
        @app.route('/health', methods=['GET'])
        @app.route('/api/health', methods=['GET'])
        def health_check():
            response_builder = create_response_builder()
            
            try:
                # Check container status
                container_status = {
                    "registered_services": len(self.container.get_registered_services()),
                    "validation_issues": self.container.validate_registrations()
                }
                
                # Check database connection
                try:
                    vector_store = self.container.resolve(IVectorStore)
                    db_health = asyncio.run(vector_store.get_health_status())
                    db_status = {
                        "healthy": db_health.healthy,
                        "message": db_health.message
                    }
                except Exception as e:
                    db_status = {
                        "healthy": False,
                        "message": f"Database check failed: {e}"
                    }
                
                health_data = {
                    "status": "healthy" if db_status["healthy"] else "unhealthy",
                    "timestamp": "2024-01-01T00:00:00Z",  # Would use datetime.now()
                    "components": {
                        "database": db_status,
                        "container": container_status
                    }
                }
                
                status_code = 200 if db_status["healthy"] else 503
                response = response_builder.success(health_data)
                return response.to_dict(), status_code
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                response = response_builder.system_error(f"Health check failed: {e}")
                return response.to_dict(), 500
    
    def _add_middleware(self, app: Flask):
        """Add middleware for request processing"""
        
        @app.before_request
        def before_request():
            """Before request middleware"""
            # Clear scoped instances for new request
            self.container.clear_scoped()
        
        @app.after_request
        def after_request(response):
            """After request middleware"""
            # Add standard headers
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            
            return response
    
    def initialize_application(self) -> Flask:
        """Initialize the complete application"""
        logger.info("Initializing Vector RAG Database Application (Clean Architecture)")
        
        try:
            # Configure dependency injection
            self.configure_services()
            
            # Set global container
            set_container(self.container)
            
            # Create Flask app
            app = self.create_flask_app()
            
            logger.info("Application initialized successfully")
            return app
            
        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            raise


def create_app() -> Flask:
    """Application factory function"""
    bootstrapper = ApplicationBootstrapper()
    return bootstrapper.initialize_application()


# Mock implementations for missing interfaces (for demonstration)
class MockFileRepository:
    """Mock file repository implementation"""
    
    async def save_file(self, file_upload, content: bytes):
        return file_upload
    
    async def get_file_metadata(self, file_id: str):
        return None
    
    async def get_file_content(self, file_id: str):
        return None
    
    async def delete_file(self, file_id: str):
        return True
    
    async def list_files(self, user_id=None):
        return []


class MockFileValidationService:
    """Mock file validation service"""
    
    async def validate_file(self, file_data: bytes, filename: str):
        return {"valid": True, "errors": []}
    
    async def scan_for_threats(self, file_data: bytes):
        return {"safe": True, "threats": []}


class MockCacheService:
    """Mock cache service implementation"""
    
    def __init__(self):
        self._cache = {}
    
    async def get(self, key: str):
        return self._cache.get(key)
    
    async def set(self, key: str, value, ttl=None):
        self._cache[key] = value
        return True
    
    async def delete(self, key: str):
        self._cache.pop(key, None)
        return True
    
    async def clear(self):
        self._cache.clear()
        return True


# Register mock implementations
def configure_mock_services(container: ServiceContainer):
    """Configure mock services for missing implementations"""
    container.register_singleton(IFileRepository, MockFileRepository)
    container.register_singleton(IFileValidationService, MockFileValidationService)
    container.register_singleton(ICacheService, MockCacheService)


if __name__ == '__main__':
    try:
        # Create application
        app = create_app()
        
        # Configure mock services
        container = get_container()
        configure_mock_services(container)
        
        # Display startup information
        logger.info("=" * 60)
        logger.info("Vector RAG Database - Clean Architecture")
        logger.info("=" * 60)
        logger.info("Features:")
        logger.info("✓ Clean Architecture with separation of concerns")
        logger.info("✓ Dependency Injection container")
        logger.info("✓ Repository pattern for data access")
        logger.info("✓ Service layer with single responsibilities")
        logger.info("✓ Standardized API responses and error handling")
        logger.info("✓ Domain-driven design with entities and interfaces")
        logger.info("=" * 60)
        
        # Validate service registrations
        issues = container.validate_registrations()
        if issues:
            logger.warning(f"Service validation issues: {issues}")
        else:
            logger.info("All services validated successfully")
        
        # Start the application
        logger.info("Starting application on http://localhost:8000")
        app.run(host='0.0.0.0', port=8000, debug=True)
        
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)