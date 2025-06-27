"""
Domain interfaces defining contracts for external dependencies
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .entities import (
    Document, SearchResult, SearchQuery, User, AgentResponse, 
    FileUpload, HealthStatus, SystemStatus, AgentType
)


class IDocumentRepository(ABC):
    """Interface for document data access"""
    
    @abstractmethod
    async def create(self, document: Document) -> Document:
        """Create a new document"""
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, offset: int = 0, limit: int = 100) -> List[Document]:
        """Get all documents with pagination"""
        pass
    
    @abstractmethod
    async def update(self, document: Document) -> Document:
        """Update existing document"""
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document by ID"""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search documents"""
        pass
    
    @abstractmethod
    async def get_count(self) -> int:
        """Get total document count"""
        pass


class IUserRepository(ABC):
    """Interface for user data access"""
    
    @abstractmethod
    async def create(self, user: User) -> User:
        """Create a new user"""
        pass
    
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        pass
    
    @abstractmethod
    async def get_all(self, offset: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination"""
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        """Update existing user"""
        pass
    
    @abstractmethod
    async def delete(self, user_id: str) -> bool:
        """Delete user by ID"""
        pass


class IFileRepository(ABC):
    """Interface for file storage and metadata"""
    
    @abstractmethod
    async def save_file(self, file_upload: FileUpload, content: bytes) -> FileUpload:
        """Save file and metadata"""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, file_id: str) -> Optional[FileUpload]:
        """Get file metadata by ID"""
        pass
    
    @abstractmethod
    async def get_file_content(self, file_id: str) -> Optional[bytes]:
        """Get file content by ID"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """Delete file and metadata"""
        pass
    
    @abstractmethod
    async def list_files(self, user_id: Optional[str] = None) -> List[FileUpload]:
        """List files, optionally filtered by user"""
        pass


class IVectorStore(ABC):
    """Interface for vector database operations"""
    
    @abstractmethod
    async def add_document(self, document: Document) -> bool:
        """Add document to vector store"""
        pass
    
    @abstractmethod
    async def search_similar(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store"""
        pass
    
    @abstractmethod
    async def update_document(self, document: Document) -> bool:
        """Update document in vector store"""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """Get vector store health status"""
        pass


class IAIAgent(ABC):
    """Interface for AI agent operations"""
    
    @abstractmethod
    async def process_query(self, query: str, context: Optional[str] = None) -> AgentResponse:
        """Process query and return response"""
        pass
    
    @abstractmethod
    def get_agent_type(self) -> AgentType:
        """Get agent type"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> HealthStatus:
        """Get agent health status"""
        pass


class IAuthenticationService(ABC):
    """Interface for authentication operations"""
    
    @abstractmethod
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with credentials"""
        pass
    
    @abstractmethod
    async def generate_token(self, user: User) -> str:
        """Generate authentication token"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate token and return user"""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke authentication token"""
        pass


class IAuthorizationService(ABC):
    """Interface for authorization operations"""
    
    @abstractmethod
    async def check_permission(self, user: User, permission: str, resource: Optional[str] = None) -> bool:
        """Check if user has permission for resource"""
        pass
    
    @abstractmethod
    async def get_user_permissions(self, user: User) -> List[str]:
        """Get all permissions for user"""
        pass


class IFileValidationService(ABC):
    """Interface for file validation operations"""
    
    @abstractmethod
    async def validate_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate file content and metadata"""
        pass
    
    @abstractmethod
    async def scan_for_threats(self, file_data: bytes) -> Dict[str, Any]:
        """Scan file for security threats"""
        pass


class INotificationService(ABC):
    """Interface for notification operations"""
    
    @abstractmethod
    async def send_notification(self, user_id: str, message: str, notification_type: str = "info") -> bool:
        """Send notification to user"""
        pass
    
    @abstractmethod
    async def send_system_alert(self, message: str, severity: str = "warning") -> bool:
        """Send system alert"""
        pass


class ICacheService(ABC):
    """Interface for caching operations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass


class IHealthMonitor(ABC):
    """Interface for health monitoring operations"""
    
    @abstractmethod
    async def check_component_health(self, component_name: str) -> HealthStatus:
        """Check health of specific component"""
        pass
    
    @abstractmethod
    async def get_system_status(self) -> SystemStatus:
        """Get overall system status"""
        pass
    
    @abstractmethod
    async def register_health_check(self, component_name: str, check_function) -> bool:
        """Register custom health check function"""
        pass


class IEventPublisher(ABC):
    """Interface for event publishing"""
    
    @abstractmethod
    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Publish domain event"""
        pass


class IConfigurationService(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        pass
    
    @abstractmethod
    async def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values"""
        pass


class ILoggingService(ABC):
    """Interface for logging operations"""
    
    @abstractmethod
    async def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log info message"""
        pass
    
    @abstractmethod
    async def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message"""
        pass
    
    @abstractmethod
    async def log_error(self, message: str, exception: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error message"""
        pass
    
    @abstractmethod
    async def log_debug(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message"""
        pass