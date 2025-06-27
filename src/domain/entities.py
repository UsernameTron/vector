"""
Domain entities representing core business objects
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


class DocumentStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UserRole(Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class AgentType(Enum):
    """AI Agent types"""
    RESEARCH = "research"
    CEO = "ceo"
    PERFORMANCE = "performance"
    COACHING = "coaching"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    CONTACT_CENTER = "contact_center"


@dataclass
class Document:
    """Domain entity representing a document"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    source: str = "unknown"
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if not self.title:
            self.title = f"Document {self.id[:8]}"
        
        # Update metadata
        self.metadata.update({
            'content_length': len(self.content),
            'word_count': len(self.content.split()),
            'content_hash': hash(self.content)
        })
    
    def mark_as_processed(self):
        """Mark document as successfully processed"""
        self.status = DocumentStatus.COMPLETED
        self.updated_at = datetime.now()
    
    def mark_as_failed(self, error: str = None):
        """Mark document as failed to process"""
        self.status = DocumentStatus.FAILED
        self.updated_at = datetime.now()
        if error:
            self.metadata['error'] = error
    
    def update_content(self, new_content: str):
        """Update document content and metadata"""
        if not new_content:
            raise ValueError("Content cannot be empty")
        
        self.content = new_content
        self.updated_at = datetime.now()
        self.metadata.update({
            'content_length': len(new_content),
            'word_count': len(new_content.split()),
            'content_hash': hash(new_content)
        })


@dataclass
class SearchResult:
    """Domain entity representing a search result"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    distance: Optional[float] = None
    
    def __post_init__(self):
        """Validate search result"""
        if not 0 <= self.relevance_score <= 1:
            raise ValueError("Relevance score must be between 0 and 1")


@dataclass
class SearchQuery:
    """Domain entity representing a search query"""
    query: str
    limit: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate search query"""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        if self.limit <= 0 or self.limit > 100:
            raise ValueError("Limit must be between 1 and 100")


@dataclass
class User:
    """Domain entity representing a user"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate user"""
        if not self.username:
            raise ValueError("Username cannot be empty")
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        permissions = {
            UserRole.ADMIN: ['read', 'write', 'delete', 'manage_users'],
            UserRole.USER: ['read', 'write'],
            UserRole.READONLY: ['read']
        }
        return permission in permissions.get(self.role, [])
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now()


@dataclass
class AgentResponse:
    """Domain entity representing an AI agent response"""
    agent_type: AgentType
    query: str
    response: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate agent response"""
        if not self.query:
            raise ValueError("Query cannot be empty")
        if not self.response:
            raise ValueError("Response cannot be empty")


@dataclass
class FileUpload:
    """Domain entity representing a file upload"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    original_filename: str = ""
    content_type: str = ""
    size_bytes: int = 0
    hash_sha256: str = ""
    uploaded_by: Optional[str] = None
    uploaded_at: datetime = field(default_factory=datetime.now)
    validation_result: Dict[str, Any] = field(default_factory=dict)
    processing_status: DocumentStatus = DocumentStatus.PENDING
    
    def __post_init__(self):
        """Validate file upload"""
        if not self.filename:
            raise ValueError("Filename cannot be empty")
        if self.size_bytes <= 0:
            raise ValueError("File size must be greater than 0")
    
    def is_valid(self) -> bool:
        """Check if file passed validation"""
        return self.validation_result.get('valid', False)
    
    def mark_as_processed(self, document_id: str):
        """Mark file as successfully processed"""
        self.processing_status = DocumentStatus.COMPLETED
        self.validation_result['document_id'] = document_id


@dataclass
class HealthStatus:
    """Domain entity representing system health status"""
    component: str
    healthy: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate health status"""
        if not self.component:
            raise ValueError("Component name cannot be empty")


@dataclass
class SystemStatus:
    """Domain entity representing overall system status"""
    status: str
    uptime: str
    components: List[HealthStatus] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return all(component.healthy for component in self.components)
    
    def add_component_status(self, component_status: HealthStatus):
        """Add component health status"""
        self.components.append(component_status)
    
    def get_component_status(self, component_name: str) -> Optional[HealthStatus]:
        """Get status for specific component"""
        for component in self.components:
            if component.component == component_name:
                return component
        return None