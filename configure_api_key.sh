#!/bin/bash

# OpenAI API Key Configuration Script for Vector RAG Database

echo "🔑 Vector RAG Database - OpenAI API Key Setup"
echo "============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}📝 $1${NC}"; }

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    print_info "Creating .env file from template..."
    
    cat > .env << 'EOF'
# Environment Variables for Vector RAG Database
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=production
FLASK_DEBUG=False
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_COLLECTION_NAME=vector_rag_collection
SECRET_KEY=change-this-to-a-random-secret-key-in-production
MAX_CONTENT_LENGTH=16777216
EOF
    
    print_status ".env file created"
fi

# Check current API key
current_key=$(grep "OPENAI_API_KEY=" .env | cut -d'=' -f2)

if [ "$current_key" = "your_openai_api_key_here" ] || [ "$current_key" = "sk-demo-key-placeholder-for-testing" ] || [ -z "$current_key" ]; then
    echo "🔑 OpenAI API Key Required"
    echo "-------------------------"
    echo ""
    echo "Your Vector RAG Database needs a valid OpenAI API key to function properly."
    echo ""
    echo "📋 How to get your API key:"
    echo "1. Visit: https://platform.openai.com/api-keys"
    echo "2. Sign in to your OpenAI account (or create one)"
    echo "3. Click 'Create new secret key'"
    echo "4. Give it a name (e.g., 'Vector RAG Database')"
    echo "5. Copy the generated key (starts with 'sk-')"
    echo ""
    
    # Check if user wants to set it now
    read -p "Do you have your OpenAI API key ready to configure? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔐 Please enter your OpenAI API key:"
        echo "(Your input will be hidden for security)"
        read -s api_key
        echo ""
        
        # Validate the key format
        if [[ $api_key =~ ^sk-[a-zA-Z0-9]{32,}$ ]]; then
            # Update the .env file
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                sed -i '' "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
            else
                # Linux
                sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
            fi
            
            print_status "API key configured successfully!"
            
            # Test the API key
            echo ""
            print_info "Testing your API key..."
            
            # Create a simple test script
            cat > test_api_key.py << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Test the connection
    models = client.models.list()
    print("✅ API key is valid and working!")
    print(f"📊 You have access to {len(models.data)} models")
    
except Exception as e:
    print(f"❌ API key test failed: {e}")
    exit(1)
EOF
            
            # Run the test if Python is available
            if command -v python3 &> /dev/null; then
                python3 test_api_key.py
                test_result=$?
                rm -f test_api_key.py
                
                if [ $test_result -eq 0 ]; then
                    print_status "API key verification successful!"
                else
                    print_warning "API key verification failed. Please check your key."
                fi
            else
                print_info "Python not found. API key saved but not tested."
                rm -f test_api_key.py
            fi
            
        else
            print_error "Invalid API key format. OpenAI keys should start with 'sk-'"
            print_info "Please run this script again with a valid API key."
            exit 1
        fi
        
    else
        print_warning "API key setup skipped."
        print_info "You can run this script again later or manually edit the .env file."
        echo ""
        print_info "To manually configure:"
        echo "1. Edit the .env file in this directory"
        echo "2. Replace 'your_openai_api_key_here' with your actual API key"
        echo "3. Save the file and restart the application"
    fi
    
else
    print_status "OpenAI API key is already configured!"
    
    # Offer to test it
    read -p "Would you like to test the current API key? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Testing API key..."
        
        cat > test_api_key.py << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    models = client.models.list()
    print("✅ API key is valid and working!")
    print(f"📊 You have access to {len(models.data)} models")
except Exception as e:
    print(f"❌ API key test failed: {e}")
    exit(1)
EOF
        
        if command -v python3 &> /dev/null; then
            python3 test_api_key.py
            rm -f test_api_key.py
        else
            print_warning "Python not found. Cannot test API key."
            rm -f test_api_key.py
        fi
    fi
fi

echo ""
echo "🚀 Next Steps:"
echo "=============="
echo ""
print_info "1. Install dependencies (if you haven't already):"
echo "   ./setup_production.sh"
echo ""
print_info "2. Start the Vector RAG Database:"
echo "   python3 app_demo.py"
echo ""
print_info "3. Open your browser to:"
echo "   http://localhost:5001"
echo ""
print_status "Your Vector RAG Database with 6 AI agents will be ready! 🎯"
echo ""
