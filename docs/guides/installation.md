# Installation Guide

## Prerequisites

- Python 3.7 or higher
- Git
- OpenAI API key
- 2GB RAM minimum
- 500MB available disk space

## Standard Installation

### 1. Clone the Repository

```bash
git clone https://github.com/UsernameTron/vector.git
cd vector-rag-database
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env file and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

## Running the Application

### Option 1: Desktop Launcher (Recommended)

The easiest way to run the application with a GUI interface:

```bash
# Make scripts executable (macOS/Linux)
chmod +x launch.sh desktop_launcher.py

# Run the launcher
./launch.sh
# Or directly:
python desktop_launcher.py
```

**Windows:**
```cmd
# Double-click launch.bat
# Or run:
python desktop_launcher.py
```

### Option 2: Command Line

Run the unified application with different modes:

```bash
# Production mode (port 5001)
python app_unified.py --mode production

# Development mode with debug logging
python app_unified.py --mode development

# Clean architecture mode
python app_unified.py --mode clean
```

### Option 3: Direct Application Files

```bash
# Production version
python app_production.py

# Basic version
python app.py

# Clean architecture version
python app_clean_architecture.py
```

## macOS-Specific Instructions

### System Requirements

- macOS 10.14 (Mojave) or later
- Xcode Command Line Tools
- Homebrew (recommended)

### Installation Steps

1. **Install Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew (if not installed):**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install Python 3 (if needed):**
   ```bash
   brew install python3
   ```

4. **Follow standard installation steps above**

### Troubleshooting macOS Issues

- **Permission errors:** Use `sudo` for system-wide installation
- **SSL certificate errors:** Update certificates with `brew install ca-certificates`
- **Port already in use:** Kill existing process with `lsof -ti:5001 | xargs kill -9`

## Windows-Specific Instructions

### System Requirements

- Windows 10 or later
- Python 3.7+ from python.org
- Git for Windows

### Installation Steps

1. **Install Python:**
   - Download from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Install Git:**
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Open Command Prompt or PowerShell**

4. **Follow standard installation steps above**

### Troubleshooting Windows Issues

- **'python' not recognized:** Ensure Python is in PATH
- **Permission denied:** Run as Administrator
- **SSL errors:** Install certificates manually

## Linux-Specific Instructions

### Ubuntu/Debian

```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Follow standard installation steps
```

### Fedora/RHEL

```bash
# Install dependencies
sudo dnf install python3 python3-pip git

# Follow standard installation steps
```

### Arch Linux

```bash
# Install dependencies
sudo pacman -S python python-pip git

# Follow standard installation steps
```

## Verifying Installation

### 1. Check Server Health

```bash
curl http://localhost:5001/health
```

Expected response:
```json
{
  "status": "healthy",
  "agents_available": 8,
  "vector_db_available": true
}
```

### 2. Access Web Interface

Open your browser and navigate to:
- Production: http://localhost:5001
- Development: http://localhost:5000

### 3. Test API Endpoints

```bash
# List available agents
curl http://localhost:5001/api/agents

# Check documents
curl http://localhost:5001/api/documents
```

## Common Installation Issues

### Issue: ModuleNotFoundError

**Solution:** Ensure virtual environment is activated and requirements installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: OpenAI API Key Error

**Solution:** Verify your .env file contains a valid API key:
```bash
cat .env | grep OPENAI_API_KEY
```

### Issue: Port Already in Use

**Solution:** Find and kill the process using the port:
```bash
# Find process
lsof -i :5001  # macOS/Linux
netstat -ano | findstr :5001  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### Issue: ChromaDB Initialization Error

**Solution:** Delete the existing database and restart:
```bash
rm -rf chroma_db/
python app_unified.py --mode production
```

## Next Steps

- Read the [Architecture Overview](../architecture/overview.md)
- Review [API Reference](../reference/api.md)
- Check [Deployment Guide](../deployment/production.md) for production setup
- See [Troubleshooting](troubleshooting.md) for more solutions