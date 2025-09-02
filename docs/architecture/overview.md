# Architecture Overview

## System Architecture

The Vector RAG Database follows a modular architecture with multiple implementation patterns to support different deployment scenarios.

## Core Components

### 1. AI Agent System

The system includes 8 specialized AI agents, each optimized for specific domains:

- **ResearchAgent**: Deep analysis and information synthesis
- **CEOAgent**: Strategic planning and executive decisions
- **PerformanceAgent**: System optimization and analytics
- **CoachingAgent**: Guidance and skill development
- **CodeAnalyzerAgent**: Code analysis and technical reviews
- **TriageAgent**: Issue prioritization and routing
- **BusinessIntelligenceAgent**: Data analytics and business insights
- **ContactCenterDirectorAgent**: Call center operations and metrics

Each agent:
- Inherits from `BaseAgent` abstract class
- Has specialized system prompts and context
- Integrates with the vector database for RAG capabilities
- Supports conversation history management

### 2. Vector Database Layer

**Primary Implementation:** ChromaDB

- **VectorDatabase** (`vector_db.py`): Basic vector storage and retrieval
- **VectorDatabaseEnhanced** (`vector_db_enhanced.py`): Advanced features with metadata
- **VectorDatabaseRobust** (`vector_db_robust.py`): Production-ready with error handling

Key features:
- Document embedding using OpenAI embeddings
- Similarity search with configurable top-k results
- Metadata filtering and management
- Persistent storage with automatic recovery

### 3. Application Layers

#### Clean Architecture Implementation (`src/`)

Following Domain-Driven Design principles:

**Domain Layer** (`src/domain/`)
- `entities/`: Core business entities (Document, Agent, User)
- `interfaces/`: Repository and service interfaces
- `value_objects/`: Immutable value objects

**Application Layer** (`src/application/`)
- `services/`: Business logic services
  - `DocumentService`: Document CRUD operations
  - `AIAgentService`: Agent interaction management
  - `FileProcessingService`: File upload and parsing
- `use_cases/`: Specific business use cases

**Infrastructure Layer** (`src/infrastructure/`)
- `repositories/`: Data access implementations
  - `ChromaDocumentRepository`: Vector database access
- `container.py`: Dependency injection container
- `config.py`: Configuration management

**Presentation Layer** (`src/presentation/`)
- `controllers/`: API endpoint controllers
- `middleware/`: Request/response middleware
- `responses/`: Standardized API responses

### 4. Application Entry Points

Multiple application files for different scenarios:

- **app_unified.py**: Main unified application with mode switching
- **app_production.py**: Production-optimized version
- **app_clean_architecture.py**: Clean architecture implementation
- **app.py**: Basic Flask application
- **desktop_launcher.py**: GUI launcher application

## Design Patterns

### Dependency Injection

Custom DI container supporting:
- Singleton, Scoped, and Transient lifetimes
- Automatic dependency resolution via type hints
- Decorator-based service registration

Example:
```python
@container.register(ServiceLifetime.SINGLETON)
class DocumentService:
    def __init__(self, repository: IDocumentRepository):
        self.repository = repository
```

### Repository Pattern

Abstracts data access with interfaces:
```python
class IDocumentRepository(ABC):
    @abstractmethod
    async def create(self, document: Document) -> Document:
        pass
```

### Factory Pattern

Agent creation through factory methods:
```python
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str) -> BaseAgent:
        # Returns appropriate agent instance
```

## Data Flow

1. **Request Reception**: API endpoint receives request
2. **Validation**: Middleware validates request data
3. **Controller Processing**: Controller handles business logic
4. **Service Layer**: Service orchestrates operations
5. **Repository Access**: Data retrieved/stored via repositories
6. **Vector Search**: ChromaDB performs similarity search
7. **Agent Processing**: AI agent generates response
8. **Response Formation**: Standardized response returned

## Security Architecture

### Authentication & Authorization
- API key validation middleware
- Role-based access control (RBAC)
- JWT token support for session management

### Data Protection
- Input sanitization and validation
- SQL injection prevention
- XSS protection in responses
- Secure file upload handling

### Network Security
- CORS configuration
- Rate limiting
- Request size limits
- HTTPS enforcement in production

## Scalability Considerations

### Horizontal Scaling
- Stateless application design
- External vector database
- Load balancer compatible

### Performance Optimization
- Connection pooling
- Caching layer support
- Async processing capabilities
- Batch document processing

### Monitoring & Observability
- Structured logging
- Health check endpoints
- Performance metrics
- Error tracking

## Technology Stack

### Backend
- **Framework**: Flask 2.0+
- **Language**: Python 3.7+
- **Vector DB**: ChromaDB
- **AI/ML**: OpenAI API
- **Embeddings**: OpenAI text-embedding-ada-002

### Frontend
- **Framework**: Vanilla JavaScript
- **Styling**: Cyberpunk-themed CSS
- **UI Components**: Custom components
- **State Management**: Local state

### Infrastructure
- **Container**: Docker support
- **Process Manager**: Gunicorn
- **Environment**: Python virtual environment
- **Configuration**: Environment variables

## Deployment Architecture

### Development Mode
- Flask development server
- Debug logging enabled
- Hot reload support
- Local file storage

### Production Mode
- Gunicorn WSGI server
- Production logging
- Error handling middleware
- Persistent storage

### Clean Architecture Mode
- Full DI container
- Swagger documentation
- Advanced middleware
- Complete separation of concerns

## Database Schema

### Vector Collections
```
Collection: vector_rag_collection
├── Documents
│   ├── id: string (UUID)
│   ├── content: string
│   ├── embedding: vector[1536]
│   └── metadata: dict
└── Indexes
    └── similarity_index
```

### Metadata Structure
```json
{
  "title": "Document Title",
  "source": "upload|manual|api",
  "timestamp": "ISO 8601",
  "content_length": 1234,
  "agent_type": "research|ceo|...",
  "user_id": "optional"
}
```

## Future Architecture Considerations

- Microservices decomposition
- Event-driven architecture
- GraphQL API layer
- Real-time WebSocket support
- Distributed vector database
- Multi-tenant support