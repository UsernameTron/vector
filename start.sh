#!/bin/bash

# Vector RAG Database - Startup Script

echo "🚀 Starting Vector RAG Database Application"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up environment file..."
    cp .env.template .env
    echo "Please edit .env file with your OpenAI API key before running the application."
    echo "You can get your API key from: https://platform.openai.com/api-keys"
    read -p "Press Enter to continue once you've updated the .env file..."
fi

# Create chroma_db directory if it doesn't exist
mkdir -p chroma_db

echo "✅ Setup complete!"
echo ""
echo "🌟 To start the application:"
echo "   1. Make sure your OpenAI API key is set in .env file"
echo "   2. Run: python app.py"
echo "   3. Open your browser to: http://localhost:5000"
echo ""
echo "🤖 Available AI Agents:"
echo "   • Research Agent - Data analysis and insights"
echo "   • CEO Agent - Strategic leadership and vision"
echo "   • Performance Agent - Optimization and metrics"
echo "   • Coaching Agent - Development and mentoring"
echo "   • Business Intelligence Agent - Analytics and BI"
echo "   • Contact Center Director Agent - Customer service operations"
echo ""
echo "Happy coding! 🎉"
