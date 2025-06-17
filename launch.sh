#!/bin/bash
# Vector RAG Database Desktop Launcher Script

echo "🚀 Starting Vector RAG Database Launcher..."
echo "Current directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "desktop_launcher.py" ]; then
    echo "❌ Error: desktop_launcher.py not found"
    echo "Please run this script from the Vector RAG Database directory"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    read -p "Press Enter to exit..."
    exit 1
fi

# Launch the desktop application
echo "🎯 Launching Vector RAG Database..."
python3 desktop_launcher.py

echo "Launcher closed."
read -p "Press Enter to exit..."
