#!/usr/bin/env python3
"""
Unified Vector RAG Database Application
Single configurable application replacing all app variants
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import configuration
from app_config import get_config, FeatureFlags, AppMode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorRAGApplication:
    """Unified Vector RAG application with configurable modes"""
    
    def __init__(self, config_name: str = None):
        self.config_name = config_name or os.environ.get('FLASK_ENV', 'development')
        self.config = get_config(self.config_name)
        self.feature_flags = FeatureFlags(self.config)
        
        self.app = None
        self.vector_db = None
        self.agents = {}
        self.logger = None
        
        # Component availability flags
        self.openai_available = False
        self.vector_db_available = False
        self.full_rag_available = False
    
    def create_app(self) -> Flask:
        """Create and configure the Flask application"""
        self.app = Flask(__name__)
        
        # Apply configuration
        self.app.config.from_object(self.config)
        self.config.init_app(self.app)
        
        # Setup CORS
        CORS(self.app, origins=self.config.CORS_ORIGINS)
        
        # Setup logging
        self._setup_logging()
        
        # Setup middleware (if enabled)
        if self.feature_flags.enable_middleware:
            self._setup_middleware()
        
        # Initialize components based on feature flags
        if self.feature_flags.use_clean_architecture:
            self._setup_clean_architecture()
        else:
            self._setup_standard_architecture()
        
        # Setup API documentation (if enabled)
        if self.feature_flags.enable_swagger:
            self._setup_swagger()
        
        # Register routes
        self._register_routes()
        
        return self.app
    
    def _setup_logging(self):
        """Setup application logging"""
        try:
            from logging_config import setup_logging, setup_flask_logging
            
            self.logger = setup_logging(
                level=self.config.LOG_LEVEL,
                log_file=self.config.LOG_FILE_PATH,
                structured=self.feature_flags.use_structured_logging
            )
            
            if not self.config.TESTING:
                setup_flask_logging(self.app)
                
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=getattr(logging, self.config.LOG_LEVEL),
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            self.logger = logging.getLogger('vector_rag')
    
    def _setup_middleware(self):
        """Setup middleware components"""
        try:
            from middleware.error_handlers import register_error_handlers
            from middleware.validation import setup_validation_middleware
            
            register_error_handlers(self.app)
            setup_validation_middleware(self.app)
            
            self.logger.info("✅ Middleware registered successfully")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Middleware not available: {e}")
        except Exception as e:
            self.logger.error(f"❌ Failed to setup middleware: {e}")
    
    def _setup_clean_architecture(self):
        """Setup clean architecture components"""
        try:
            # Import clean architecture components
            sys.path.insert(0, os.path.join(current_dir, 'src'))
            from src.infrastructure.container import get_container
            from src.presentation.controllers.document_controller import document_bp
            from src.presentation.controllers.health_controller import health_bp
            
            # Register blueprints
            self.app.register_blueprint(document_bp)
            self.app.register_blueprint(health_bp)
            
            self.logger.info("🏗️ Clean architecture components loaded")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Clean architecture components not available: {e}")
            self._setup_standard_architecture()
    
    def _setup_standard_architecture(self):
        """Setup standard architecture components"""
        # Initialize vector database
        self._initialize_vector_db()
        
        # Initialize agents
        self._initialize_agents()
        
        # Check for full RAG system
        if self.feature_flags.use_full_rag_system:
            self._initialize_full_rag()
    
    def _initialize_vector_db(self):
        """Initialize vector database"""
        try:
            from vector_db import VectorDatabase
            
            self.vector_db = VectorDatabase()
            self.vector_db_available = True
            
            self.logger.info("✅ Vector database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector database: {e}")
            self.vector_db_available = False
    
    def _initialize_agents(self):
        """Initialize AI agents"""
        try:
            from agents import (
                ResearchAgent, CEOAgent, PerformanceAgent, 
                CoachingAgent, BusinessIntelligenceAgent, 
                ContactCenterDirectorAgent
            )
            
            if self.vector_db:
                self.agents = {
                    'research': ResearchAgent(self.vector_db),
                    'ceo': CEOAgent(self.vector_db),
                    'performance': PerformanceAgent(self.vector_db),
                    'coaching': CoachingAgent(self.vector_db),
                    'business_intelligence': BusinessIntelligenceAgent(self.vector_db),
                    'contact_center': ContactCenterDirectorAgent(self.vector_db)
                }
            else:
                # Create simple fallback agents
                self._create_simple_agents()
            
            self.logger.info(f"✅ {len(self.agents)} AI agents initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agents: {e}")
            self._create_simple_agents()
    
    def _create_simple_agents(self):
        """Create simple fallback agents"""
        class SimpleAgent:
            def __init__(self, name: str, role: str):
                self.name = name
                self.role = role
                self.description = f"Simple {name} for basic functionality"
                self.capabilities = [role]
            
            def query(self, message: str) -> str:
                return f"I'm {self.name}. I received your message: {message}"
        
        self.agents = {
            'research': SimpleAgent('Research Agent', 'research'),
            'ceo': SimpleAgent('CEO Agent', 'strategy'),
            'performance': SimpleAgent('Performance Agent', 'analytics'),
            'coaching': SimpleAgent('Coaching Agent', 'guidance'),
            'business_intelligence': SimpleAgent('BI Agent', 'business'),
            'contact_center': SimpleAgent('Contact Agent', 'support')
        }
    
    def _initialize_full_rag(self):
        """Initialize full RAG system if available"""
        try:
            # Try to import UnifiedAI components
            sys.path.insert(0, '/Users/cpconnor/projects/UnifiedAIPlatform/RAG')
            from search_system import SearchSystem
            from unified_agent_system import UnifiedAgentManager
            
            # Initialize full RAG system
            # Implementation would depend on available system
            self.full_rag_available = True
            self.logger.info("✅ Full RAG system available")
            
        except ImportError:
            self.logger.info("ℹ️ Full RAG system not available, using local components")
            self.full_rag_available = False
    
    def _setup_swagger(self):
        """Setup Swagger API documentation"""
        try:
            from flask_restx import Api, Resource, fields
            
            api = Api(
                self.app,
                version='1.0',
                title='Vector RAG Database API',
                description='AI-powered document management and chat system',
                doc='/api/docs/',
                prefix='/api'
            )
            
            # Define models
            chat_model = api.model('ChatRequest', {
                'message': fields.String(required=True, description='User message'),
                'agent_type': fields.String(required=True, description='Agent type'),
                'context_limit': fields.Integer(description='Context limit for RAG')
            })
            
            self.api = api
            self.chat_model = chat_model
            
            self.logger.info("📚 Swagger documentation enabled at /api/docs/")
            
        except ImportError:
            self.logger.warning("⚠️ Flask-RESTX not available, Swagger disabled")
    
    def _register_routes(self):
        """Register all application routes"""
        # Health check
        @self.app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy' if self.vector_db_available else 'degraded',
                'mode': self.config_name,
                'agents_available': len(self.agents),
                'vector_db_available': self.vector_db_available,
                'full_rag_available': self.full_rag_available,
                'timestamp': self._current_timestamp()
            })
        
        # Agents list
        @self.app.route('/api/agents')
        def agents():
            agent_list = []
            for key, agent in self.agents.items():
                agent_list.append({
                    'id': key,
                    'name': agent.name,
                    'role': getattr(agent, 'role', key),
                    'description': getattr(agent, 'description', ''),
                    'capabilities': getattr(agent, 'capabilities', [])
                })
            
            return jsonify({'agents': agent_list})
        
        # Chat endpoint
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            message = data.get('message')
            agent_type = data.get('agent_type')
            
            if not message:
                return jsonify({'error': 'Message is required'}), 400
            
            if agent_type not in self.agents:
                return jsonify({'error': 'Invalid agent type'}), 400
            
            try:
                agent = self.agents[agent_type]
                response = agent.query(message)
                
                return jsonify({
                    'response': response,
                    'agent': agent_type,
                    'timestamp': self._current_timestamp()
                })
                
            except Exception as e:
                self.logger.error(f"Chat error: {e}")
                return jsonify({'error': 'Agent temporarily unavailable'}), 500
        
        # Search endpoint
        @self.app.route('/api/search', methods=['POST'])
        def search():
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            query = data.get('query')
            limit = data.get('limit', 10)
            
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            try:
                if self.vector_db:
                    results = self.vector_db.search(query, limit)
                else:
                    results = []
                
                return jsonify({
                    'results': results,
                    'query': query,
                    'count': len(results),
                    'timestamp': self._current_timestamp()
                })
                
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                return jsonify({'error': 'Search temporarily unavailable'}), 500
        
        # Upload endpoint
        @self.app.route('/api/upload', methods=['POST'])
        def upload():
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            title = data.get('title')
            content = data.get('content')
            
            if not content:
                return jsonify({'error': 'Content is required'}), 400
            
            if not content.strip():
                return jsonify({'error': 'Content cannot be empty'}), 400
            
            try:
                if self.vector_db:
                    doc_id = self.vector_db.add_document(content, title or 'Untitled')
                else:
                    doc_id = f"doc_{self._current_timestamp()}"
                
                return jsonify({
                    'document_id': doc_id,
                    'title': title,
                    'timestamp': self._current_timestamp()
                })
                
            except Exception as e:
                self.logger.error(f"Upload error: {e}")
                return jsonify({'error': 'Upload failed'}), 500
        
        # Status endpoint
        @self.app.route('/api/status')
        def status():
            return jsonify({
                'application': 'Vector RAG Database',
                'version': '2.0.0',
                'mode': self.config_name,
                'features': {
                    'clean_architecture': self.feature_flags.use_clean_architecture,
                    'full_rag': self.full_rag_available,
                    'swagger_docs': self.feature_flags.enable_swagger,
                    'structured_logging': self.feature_flags.use_structured_logging
                },
                'components': {
                    'vector_db': self.vector_db_available,
                    'agents': len(self.agents),
                    'middleware': self.feature_flags.enable_middleware
                },
                'timestamp': self._current_timestamp()
            })
        
        # Root route
        @self.app.route('/')
        def index():
            try:
                return render_template('index.html')
            except Exception:
                # Fallback if templates not available
                return jsonify({
                    'message': f'Vector RAG Database API ({self.config_name} mode)',
                    'health': '/health',
                    'agents': '/api/agents',
                    'docs': '/api/docs/' if self.feature_flags.enable_swagger else None
                })
    
    def _current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def run(self, **kwargs):
        """Run the application"""
        if not self.app:
            self.create_app()
        
        # Default run configuration
        default_kwargs = {
            'host': os.environ.get('FLASK_HOST', '127.0.0.1'),
            'port': int(os.environ.get('FLASK_PORT', '5001')),
            'debug': self.config.DEBUG,
        }
        default_kwargs.update(kwargs)
        
        self.logger.info(f"🚀 Starting Vector RAG Database ({self.config_name} mode)")
        self.logger.info(f"🌐 Server: http://{default_kwargs['host']}:{default_kwargs['port']}")
        self.logger.info(f"🤖 Agents: {len(self.agents)} available")
        self.logger.info(f"🗄️ Vector DB: {'✅' if self.vector_db_available else '❌'}")
        
        self.app.run(**default_kwargs)


def create_app(config_name: str = None) -> Flask:
    """Application factory function"""
    app_instance = VectorRAGApplication(config_name)
    return app_instance.create_app()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector RAG Database - Unified Application')
    parser.add_argument('--mode', choices=['development', 'testing', 'production', 'clean'],
                      default=os.environ.get('FLASK_ENV', 'development'),
                      help='Application mode')
    parser.add_argument('--host', default=os.environ.get('FLASK_HOST', '127.0.0.1'),
                      help='Host to bind to')
    parser.add_argument('--port', type=int, default=int(os.environ.get('FLASK_PORT', '5001')),
                      help='Port to bind to')
    
    args = parser.parse_args()
    
    # Create and run application
    app_instance = VectorRAGApplication(args.mode)
    app_instance.create_app()
    app_instance.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()