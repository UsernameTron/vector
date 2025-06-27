# Clean Architecture Implementation

This document describes the redesigned Vector RAG Database application using Clean Architecture principles with proper separation of concerns, dependency injection, and modular design.

## 🏗️ Architecture Overview

The application follows Clean Architecture (also known as Hexagonal Architecture or Ports and Adapters) with these layers:

```
┌─────────────────────────────────────────────────┐
│                 Presentation Layer              │
│            (Controllers, API Routes)           │
├─────────────────────────────────────────────────┤
│                Application Layer                │
│          (Services, Use Cases, DTOs)           │
├─────────────────────────────────────────────────┤
│                  Domain Layer                   │
│        (Entities, Interfaces, Rules)           │
├─────────────────────────────────────────────────┤
│               Infrastructure Layer              │
│      (Repositories, External Services)         │
└─────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
src/
├── domain/                          # Core business layer
│   ├── entities.py                  # Business entities
│   └── interfaces.py                # Abstract interfaces
├── application/                     # Business logic layer
│   └── services/                    # Application services
│       ├── document_service.py      # Document management
│       ├── ai_agent_service.py      # AI agent operations
│       └── file_processing_service.py # File handling
├── infrastructure/                  # External dependencies
│   ├── container.py                 # Dependency injection
│   ├── repositories/                # Data access implementations
│   │   └── chroma_document_repository.py
│   └── services/                    # Infrastructure services
│       ├── logging_service.py
│       └── event_publisher.py
└── presentation/                    # API layer
    ├── controllers/                 # API controllers
    │   └── document_controller.py
    └── responses.py                 # Standardized responses
```

## 🔧 Key Components

### 1. Domain Layer (`src/domain/`)

The innermost layer containing business entities and rules.

#### Entities (`entities.py`)
Core business objects with behavior and validation:

```python
@dataclass
class Document:
    id: str
    title: str
    content: str
    status: DocumentStatus
    created_at: datetime
    
    def mark_as_processed(self):
        self.status = DocumentStatus.COMPLETED
        self.updated_at = datetime.now()
```

#### Interfaces (`interfaces.py`)
Abstract contracts for external dependencies:

```python
class IDocumentRepository(ABC):
    @abstractmethod
    async def create(self, document: Document) -> Document: pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[Document]: pass
```

### 2. Application Layer (`src/application/`)

Business logic and use case orchestration.

#### Services
Each service has a single responsibility:

- **DocumentService**: Document CRUD operations, validation, business rules
- **AIAgentService**: AI agent interactions, context management
- **FileProcessingService**: File upload handling, validation, content extraction

```python
@scoped()
class DocumentService:
    def __init__(
        self,
        document_repository: IDocumentRepository,
        event_publisher: IEventPublisher,
        logging_service: ILoggingService
    ):
        # Dependencies injected via constructor
```

### 3. Infrastructure Layer (`src/infrastructure/`)

External system integrations and concrete implementations.

#### Dependency Injection Container (`container.py`)
Manages object creation and dependency resolution:

```python
container = ServiceContainer()
container.register_singleton(IDocumentRepository, ChromaDocumentRepository)
container.register_scoped(DocumentService)

# Automatic dependency resolution
document_service = container.resolve(DocumentService)
```

#### Repository Implementations
Concrete data access implementations:

```python
@singleton(IDocumentRepository)
class ChromaDocumentRepository(IDocumentRepository):
    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store
```

### 4. Presentation Layer (`src/presentation/`)

API controllers and response formatting.

#### Standardized Responses (`responses.py`)
Consistent API response format:

```python
@dataclass
class ApiResponse:
    status: ResponseStatus
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: List[ErrorDetail] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    pagination: Optional[PaginationInfo] = None
```

#### Controllers (`controllers/`)
HTTP request handling with dependency injection:

```python
@document_bp.route('', methods=['POST'])
async def create_document():
    container = get_container()
    document_service = container.resolve(DocumentService)
    
    document = await document_service.create_document(...)
    return jsonify(response_builder.success(data=document).to_dict())
```

## 🔗 Dependency Injection

The application uses a custom dependency injection container supporting:

### Service Lifetimes

- **Singleton**: One instance for the entire application
- **Scoped**: One instance per request/scope
- **Transient**: New instance every time

```python
# Registration
container.register_singleton(ILoggingService, LoggingService)
container.register_scoped(DocumentService)
container.register_transient(IEmailService, EmailService)

# Decorators
@singleton(IDocumentRepository)
class ChromaDocumentRepository: ...

@scoped()
class DocumentService: ...
```

### Automatic Resolution

Constructor injection with type hints:

```python
class DocumentService:
    def __init__(
        self,
        document_repository: IDocumentRepository,  # Auto-resolved
        event_publisher: IEventPublisher,          # Auto-resolved
        logging_service: ILoggingService           # Auto-resolved
    ):
        # Dependencies automatically injected
```

## 📝 API Response Standards

### Success Response Format

```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "uuid-here"
}
```

### Error Response Format

```json
{
  "status": "error",
  "errors": [
    {
      "code": "VALIDATION_FAILED",
      "message": "Title is required",
      "field": "title"
    }
  ],
  "message": "Validation failed",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "uuid-here"
}
```

### Paginated Response Format

```json
{
  "status": "success",
  "data": [ ... ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_count": 100,
    "total_pages": 5,
    "has_next": true,
    "has_previous": false
  }
}
```

## 🚀 Usage Examples

### Starting the Application

```bash
python app_clean_architecture.py
```

### API Endpoints

#### Create Document
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Document",
    "content": "Document content here",
    "metadata": {"category": "research"}
  }'
```

#### Search Documents
```bash
curl -X POST http://localhost:8000/api/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search term",
    "limit": 10
  }'
```

#### Get Document
```bash
curl http://localhost:8000/api/documents/{document_id}
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## 🧪 Testing the Architecture

### Unit Testing Services

```python
# Test with mocked dependencies
def test_document_service():
    mock_repo = Mock(spec=IDocumentRepository)
    mock_events = Mock(spec=IEventPublisher)
    mock_logging = Mock(spec=ILoggingService)
    
    service = DocumentService(mock_repo, mock_events, mock_logging)
    
    # Test business logic in isolation
    result = await service.create_document("Title", "Content")
    
    mock_repo.create.assert_called_once()
```

### Integration Testing

```python
# Test with real container
def test_document_creation_integration():
    container = ServiceContainer()
    # Configure test services
    
    service = container.resolve(DocumentService)
    result = await service.create_document("Title", "Content")
    
    assert result.id is not None
    assert result.status == DocumentStatus.COMPLETED
```

## 🔍 Benefits of This Architecture

### 1. **Separation of Concerns**
- Domain logic isolated from infrastructure
- Business rules independent of frameworks
- Clear boundaries between layers

### 2. **Testability**
- Easy to mock dependencies
- Unit tests for business logic
- Integration tests for full flows

### 3. **Maintainability**
- Single responsibility principle
- Loose coupling via interfaces
- High cohesion within services

### 4. **Flexibility**
- Easy to swap implementations
- Database-agnostic domain layer
- Framework-independent business logic

### 5. **Scalability**
- Service-oriented architecture
- Horizontal scaling capabilities
- Microservice migration path

## 🔧 Configuration

### Environment Variables

```bash
# Database configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection

# Application configuration
FLASK_ENV=production
LOG_LEVEL=INFO

# External services
OPENAI_API_KEY=your-api-key
```

### Service Registration

```python
def configure_services(container: ServiceContainer):
    # Core services
    container.register_singleton(ILoggingService, LoggingService)
    container.register_singleton(IEventPublisher, EventPublisher)
    
    # Data access
    container.register_singleton(IVectorStore, ChromaVectorStore)
    container.register_singleton(IDocumentRepository, ChromaDocumentRepository)
    
    # Application services
    container.register_scoped(DocumentService)
    container.register_scoped(AIAgentService)
    container.register_scoped(FileProcessingService)
```

## 📚 Additional Features

### 1. **Event Publishing**
Domain events for loose coupling:

```python
await self.event_publisher.publish_event(
    "document.created",
    {"document_id": document.id, "created_by": user_id}
)
```

### 2. **Structured Logging**
Contextual logging throughout the application:

```python
await self.logging_service.log_info(
    "Document created",
    {"document_id": doc.id, "user_id": user_id}
)
```

### 3. **Health Monitoring**
Comprehensive health checks:

```bash
curl http://localhost:8000/health
```

### 4. **Error Handling**
Standardized error responses with recovery suggestions:

```python
raise ValidationException("title", "Title is required")
# Automatically converted to proper HTTP response
```

## 🔄 Migration from Previous Architecture

### Key Changes

1. **Modular Design**: Separated concerns into distinct layers
2. **Dependency Injection**: Replaced global state with proper DI
3. **Interface Segregation**: Defined clear contracts between layers
4. **Standardized Responses**: Consistent API response format
5. **Service Pattern**: Single responsibility services
6. **Repository Pattern**: Abstracted data access

### Migration Steps

1. Extract domain entities from existing code
2. Define interfaces for external dependencies
3. Implement repositories and services
4. Create dependency injection configuration
5. Update controllers to use new services
6. Test and validate functionality

This clean architecture provides a solid foundation for maintaining and extending the Vector RAG Database application while ensuring code quality, testability, and maintainability.