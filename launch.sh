#!/bin/bash
# Vector RAG Database Desktop Launcher Script

echo "🚀 Vector RAG Database - Unified Application Launcher"
echo "======================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "app_unified.py" ] && [ ! -f "desktop_launcher.py" ]; then
    echo "❌ Error: Vector RAG Database files not found"
    echo "Please run this script from the Vector RAG Database directory"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "📝 Please install Python 3.8+ to continue"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check for desktop launcher (GUI)
if [ -f "desktop_launcher.py" ]; then
    echo "🖥️  Launching GUI Desktop Application..."
    echo ""
    python3 desktop_launcher.py
else
    # Fallback to command-line unified app
    echo "📱 GUI launcher not found, starting unified application..."
    echo ""
    
    # Detect mode from environment or use production
    MODE=${FLASK_ENV:-production}
    PORT=${FLASK_PORT:-5001}
    
    echo "🎯 Mode: $MODE"
    echo "🌐 Port: $PORT" 
    echo ""
    echo "Starting Vector RAG Database..."
    echo "Access at: http://localhost:$PORT"
    echo ""
    
    python3 app_unified.py --mode $MODE --port $PORT --host 127.0.0.1
fi

echo ""
echo "Vector RAG Database session ended."
read -p "Press Enter to exit..."
