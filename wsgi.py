"""
WSGI entry point for production deployment
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging for WSGI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/vector-rag/app.log', mode='a') if os.path.exists('/var/log/vector-rag') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

try:
    # Import the application factory
    from app_production import create_app
    
    # Create the WSGI application
    application = create_app()
    
    logger.info("WSGI application created successfully")
    
except Exception as e:
    logger.error(f"Failed to create WSGI application: {e}")
    raise

if __name__ == "__main__":
    # This allows running with python wsgi.py for testing
    application.run(host='0.0.0.0', port=8000, debug=False)