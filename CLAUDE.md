# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Production setup with guided configuration
./setup_production.sh

# Install dependencies manually
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and add OPENAI_API_KEY
```

### Running the Application
```bash
# Production server (recommended)
python app_production.py  # Runs on port 5001

# Development versions
python app.py            # Basic Flask app on port 5000
python app_demo.py       # Demo version
python app_clean_architecture.py  # Clean architecture version on port 8000

# Desktop launcher
python desktop_launcher.py  # GUI launcher with status monitoring
```

### Testing
```bash
# Run tests (pytest available in requirements.txt)
pytest

# Health check
curl http://localhost:5001/health
```

### Linting and Code Quality
```bash
# Format code
black .

# Lint code
flake8
```

## Architecture Overview

This is a **Vector RAG Database** application with specialized AI agents and document processing capabilities. The system uses multiple architectural patterns:

### Main Architecture (Clean Architecture)
- **Domain Layer** (`src/domain/`): Business entities and interfaces
- **Application Layer** (`src/application/services/`): Business logic services
- **Infrastructure Layer** (`src/infrastructure/`): Data access, external services, dependency injection
- **Presentation Layer** (`src/presentation/controllers/`): API controllers and responses

### Key Components

**AI Agent System**: 6-8 specialized agents (Research, CEO, Performance, Coaching, Business Intelligence, Contact Center) with RAG capabilities
- Agent implementations in `agents.py`
- Each agent has vector database integration via `VectorDatabase` class

**Vector Database**: ChromaDB-based document storage and retrieval
- Primary implementation: `vector_db.py`
- Enhanced version: `vector_db_enhanced.py` 
- Robust version: `vector_db_robust.py`

**Application Entry Points**:
- `app_production.py`: Production-ready version (port 5001)
- `app_clean_architecture.py`: Clean architecture implementation (port 8000)
- `app.py`: Basic Flask application (port 5000)

### Key Services (Clean Architecture)
- **DocumentService**: Document CRUD, validation, business rules
- **AIAgentService**: Agent interactions and context management
- **FileProcessingService**: File upload handling and content extraction
- **ChromaDocumentRepository**: Vector database data access layer

## Environment Configuration

Required environment variables in `.env`:
```bash
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=production
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection
```

## API Structure

### Standard Response Format
All APIs return structured responses with `status`, `data`, `message`, `timestamp`, and optional `errors`/`pagination` fields.

### Key Endpoints
- `/health` - System health and agent status
- `/api/agents` - List available AI agents
- `/api/chat` - Chat with specific agent
- `/api/documents` - Document management (Clean Architecture version)
- `/api/search` - Vector search functionality

## File Organization Patterns

**Multiple App Versions**: The codebase maintains several application versions for different use cases:
- Production (`app_production.py`)
- Clean Architecture (`app_clean_architecture.py`) 
- Demo/Development (`app.py`, `app_demo.py`)
- Security-focused (`app_secure.py`)

**Launcher System**: Desktop integration with GUI launcher (`desktop_launcher.py`) and shell scripts for cross-platform deployment.

**Documentation**: Extensive documentation in `README.md`, `PRODUCTION_README.md`, `CLEAN_ARCHITECTURE.md` covering different deployment scenarios.

## Development Notes

- **Dependency Injection**: Clean architecture version uses custom DI container (`src/infrastructure/container.py`)
- **Testing**: Uses pytest framework - tests can be run with `pytest` command
- **Vector Database**: Persists to `./chroma_db` directory by default
- **Security**: Security setup available via `setup_security.py`
- **Production**: WSGI deployment ready with Gunicorn configuration