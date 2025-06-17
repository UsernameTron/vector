#!/bin/bash

# Vector RAG Database - Production Setup Script
# This script sets up the production environment with proper API key configuration

echo "🚀 Vector RAG Database - Production Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}📝 $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "app_production.py" ]; then
    print_error "app_production.py not found. Please run this script from the vector-rag-database directory."
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

print_status "Python 3 found"

# Check for pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is required but not installed."
    exit 1
fi

print_status "pip3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_info "Installing Python dependencies..."
pip install -r requirements.txt

print_status "Dependencies installed"

# Check current .env file
if [ -f ".env" ]; then
    current_key=$(grep "OPENAI_API_KEY=" .env | cut -d'=' -f2)
    if [ "$current_key" = "sk-demo-key-placeholder-for-testing" ] || [ -z "$current_key" ]; then
        print_warning "Demo API key detected in .env file"
        need_api_key=true
    else
        print_status "OpenAI API key found in .env file"
        need_api_key=false
    fi
else
    print_warning ".env file not found"
    need_api_key=true
fi

# Handle API key setup
if [ "$need_api_key" = true ]; then
    echo ""
    echo "🔑 OpenAI API Key Setup Required"
    echo "--------------------------------"
    echo ""
    echo "To use the full functionality of the Vector RAG Database, you need an OpenAI API key."
    echo ""
    echo "Steps to get your API key:"
    echo "1. Go to: https://platform.openai.com/api-keys"
    echo "2. Sign in or create an OpenAI account"
    echo "3. Click 'Create new secret key'"
    echo "4. Copy the generated key"
    echo ""
    
    read -p "Do you have your OpenAI API key ready? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        read -p "Enter your OpenAI API key: " -s api_key
        echo ""
        
        if [ ! -z "$api_key" ]; then
            # Update or create .env file
            if [ -f ".env" ]; then
                # Replace existing key
                if grep -q "OPENAI_API_KEY=" .env; then
                    sed -i '' "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
                else
                    echo "OPENAI_API_KEY=$api_key" >> .env
                fi
            else
                # Create new .env file
                cat > .env << EOF
# Environment Variables for Vector RAG Database
OPENAI_API_KEY=$api_key
FLASK_ENV=production
FLASK_DEBUG=False
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection
SECRET_KEY=$(openssl rand -hex 32)
EOF
            fi
            print_status "API key configured successfully!"
        else
            print_error "No API key provided."
            need_api_key=true
        fi
    else
        print_warning "API key setup skipped. You can manually edit the .env file later."
        # Create .env with placeholder if it doesn't exist
        if [ ! -f ".env" ]; then
            cat > .env << EOF
# Environment Variables for Vector RAG Database
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=production
FLASK_DEBUG=False
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection
SECRET_KEY=$(openssl rand -hex 32)
EOF
            print_info ".env file created with placeholder values"
        fi
    fi
fi

# Test the setup
echo ""
print_info "Testing the setup..."

# Test Python imports
python3 -c "
import sys
try:
    import flask
    import openai
    import chromadb
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "Package imports successful"
else
    print_error "Package import failed"
    exit 1
fi

# Test OpenAI API key if provided
if [ "$need_api_key" = false ] || [ ! -z "$api_key" ]; then
    print_info "Testing OpenAI API connection..."
    python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    client = OpenAI()
    models = client.models.list()
    print('✅ OpenAI API connection successful')
except Exception as e:
    print(f'⚠️  OpenAI API test failed: {e}')
    print('   The application will run in limited mode')
"
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p chroma_db
mkdir -p logs
mkdir -p uploads

print_status "Directories created"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
print_status "Vector RAG Database is ready to run"
echo ""
echo "🚀 To start the application:"
echo "   source venv/bin/activate"
echo "   python3 app_production.py"
echo ""
echo "🌐 The application will be available at:"
echo "   http://localhost:5001"
echo ""

if [ "$need_api_key" = true ] && [ -z "$api_key" ]; then
    print_warning "Remember to configure your OpenAI API key in the .env file for full functionality!"
    echo "   Edit .env and replace 'your_openai_api_key_here' with your actual API key"
fi

echo ""
print_info "🎯 Available AI Agents:"
echo "   • Research Agent - Market intelligence and analysis"
echo "   • CEO Agent - Strategic planning and coordination"
echo "   • Performance Agent - Analytics and optimization"
echo "   • Coaching Agent - Guidance and development"
echo "   • Business Intelligence Agent - Data analytics"
echo "   • Contact Center Agent - Operations management"
echo ""

print_status "Setup completed successfully! 🚀"
