# Vector RAG Database

A cyberpunk-themed application featuring specialized AI agents with RAG capabilities, powered by ChromaDB and OpenAI.

## 🚀 Quick Start

```bash
# Option 1: Desktop Launcher (Recommended)
./launch.sh

# Option 2: Manual Setup
pip install -r requirements.txt
cp .env.template .env
# Edit .env and add OPENAI_API_KEY
python app_unified.py --mode production
```

Visit http://localhost:5001 to access the interface.

## 🤖 Key Features

- **8 Specialized AI Agents**: Research, CEO, Performance, Coaching, Code Analyzer, Triage, Business Intelligence, Contact Center
- **Vector Database Integration**: ChromaDB for intelligent document retrieval
- **Clean Architecture**: Proper separation of concerns with Domain, Application, Infrastructure, and Presentation layers
- **Multiple Deployment Modes**: Production, Development, and Clean Architecture modes
- **Desktop Launcher**: GUI application for easy startup and management
- **Secure API Endpoints**: Authentication and validation middleware

## 📚 Documentation

- [Installation Guide](docs/guides/installation.md) - Complete setup instructions
- [Architecture Overview](docs/architecture/overview.md) - System design and patterns
- [Deployment Guide](docs/deployment/production.md) - Production deployment
- [API Reference](docs/reference/api.md) - Endpoint documentation
- [Troubleshooting](docs/guides/troubleshooting.md) - Common issues and solutions
- [Security](docs/guides/security.md) - Security considerations

## 🧪 Development

```bash
# Run tests
pytest

# Run specific test file
pytest tests/unit/test_file_parser.py

# Format code
black .

# Lint code
flake8

# Run development server
python app_unified.py --mode development
```

## 📋 System Requirements

- Python 3.7+
- OpenAI API key
- 2GB RAM minimum
- 500MB disk space

## 🏗️ Project Structure

```
vector-rag-database/
├── app_unified.py          # Main unified application
├── agents.py               # AI agent implementations
├── vector_db.py            # Vector database integration
├── src/                    # Clean architecture implementation
│   ├── domain/            # Business entities
│   ├── application/       # Business logic
│   ├── infrastructure/    # External services
│   └── presentation/      # API controllers
├── static/                # Frontend assets
├── templates/             # HTML templates
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## 🚢 Deployment Options

1. **Desktop Application**: Use `desktop_launcher.py` for GUI-based management
2. **Docker**: Production-ready containerized deployment
3. **Manual**: Direct Python execution with customizable modes

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 💬 Support

For issues and questions, please use the GitHub issue tracker.