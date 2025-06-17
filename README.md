# Vector RAG Database

🤖 **Specialized AI Agent Platform with Vector Database Integration**

A standalone cyberpunk-themed application featuring 6 specialized AI agents with RAG (Retrieval-Augmented Generation) capabilities, powered by ChromaDB vector storage and OpenAI embeddings.

## 🌟 Featured Agents

### Core Business Intelligence
- **🎯 Research Agent** - Deep analysis, market research, and information synthesis
- **👔 CEO Agent** - Strategic planning, executive decisions, and high-level coordination

### Performance & Operations  
- **📊 Performance Agent** - System optimization, analytics, and performance monitoring
- **🎓 Coaching Agent** - AI-powered guidance, mentoring, and skill development

### Specialized Intelligence
- **💼 Business Intelligence Agent** - Data analytics, KPIs, and business insights
- **📞 Contact Center Director Agent** - Call center metrics, operations, and customer analytics

## ✨ Key Features

- **Vector RAG Integration** - Upload documents and chat with AI agents using your data
- **Cyberpunk Light Blue Theme** - Modern, professional interface with cyberpunk aesthetics
- **Real-time Agent Selection** - Choose the perfect specialist for each task
- **Document Management** - Upload, process, and manage knowledge base
- **Export Capabilities** - Download conversation history and insights
- **Advanced Analytics** - Track agent performance and system metrics

## 🚀 Quick Start

### Option 1: Desktop Launcher (Recommended)
**Easy one-click startup with GUI interface:**

```bash
# Make scripts executable (Linux/Mac)
chmod +x desktop_launcher.py launch.sh

# Launch with GUI (Linux/Mac)
./launch.sh

# Or run directly
python3 desktop_launcher.py
```

**Windows users:**
- Double-click `launch.bat` or run `python desktop_launcher.py`

### Option 2: Manual Command Line
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY

# Run the application
python app.py
```

Visit `http://localhost:5000` to access the Vector RAG Database interface.

## 🖥️ Desktop Launcher Features

The desktop launcher provides:
- **One-click startup** - Automatic dependency checking and server launch
- **Browser integration** - Automatically opens web interface when ready
- **Status monitoring** - Real-time system status and health checks
- **Easy shutdown** - Clean server termination with confirmation
- **Error handling** - Helpful error messages and troubleshooting tips

**Launcher Requirements:**
- Python 3.7+
- Tkinter (usually included with Python)
- All project dependencies (automatically checked)

## 🏗️ Architecture

Built on the modular agent framework from the UnifiedAIPlatform, following Phase 2 strategy for standalone deployment with specialized business intelligence capabilities.

## 📊 Agent Capabilities

Each agent is optimized for specific domains while maintaining full RAG integration for context-aware responses based on your uploaded documents.
