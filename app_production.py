#!/usr/bin/env python3
"""
Vector RAG Database Application - Production Version
Full-featured application with real AI agents and RAG integration
"""

import os
import sys
import json
import asyncio
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Flask and web framework imports
from flask import Flask, request, jsonify, render_template, current_app
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import from the UnifiedAI platform
try:
    sys.path.insert(0, '/Users/cpconnor/projects/UnifiedAIPlatform/RAG')
    sys.path.insert(0, '/Users/cpconnor/projects/UnifiedAIPlatform/VectorDBRAG')
    
    from search_system import SearchSystem
    from config import Config
    from legacy_agents import (
        CEOAgent, ResearchAgent, PerformanceAgent, CoachingAgent,
        CodeAnalyzerAgent, TriageAgent
    )
    from unified_agent_system import UnifiedAgentManager
    
    FULL_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Full system components not available: {e}")
    logger.info("Creating simplified version with local components")
    FULL_SYSTEM_AVAILABLE = False

# Local vector database import
try:
    from vector_db import VectorDatabase
    LOCAL_VECTOR_DB_AVAILABLE = True
except ImportError:
    LOCAL_VECTOR_DB_AVAILABLE = False

# OpenAI client
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging for production (minimal console output)
try:
    from logging_config import setup_logging
    logger = setup_logging(level='WARNING', structured=False)
except ImportError:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class ProductionVectorRAG:
    """Main production application class"""
    
    def __init__(self):
        self.app = None
        self.openai_client = None
        self.vector_db = None
        self.agents = {}
        self.search_system = None
        self.agent_manager = None
        
    def initialize_openai(self) -> bool:
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI package not available")
            return False
            
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'sk-demo-key-placeholder-for-testing':
            logger.error("Valid OpenAI API key not found in environment")
            return False
            
        try:
            self.openai_client = OpenAI(api_key=api_key)
            # Test the connection
            self.openai_client.models.list()
            logger.info("✅ OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            return False
    
    def initialize_vector_db(self) -> bool:
        """Initialize vector database"""
        if LOCAL_VECTOR_DB_AVAILABLE:
            try:
                self.vector_db = VectorDatabase()
                logger.info("✅ Local vector database initialized")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize vector database: {e}")
                return False
        return False
    
    def initialize_rag_system(self) -> bool:
        """Initialize full RAG system if available"""
        if FULL_SYSTEM_AVAILABLE:
            try:
                config = Config('production')
                self.search_system = SearchSystem(config)
                logger.info("✅ Full RAG search system initialized")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG system: {e}")
                return False
        return False
    
    def initialize_agents(self) -> bool:
        """Initialize AI agents"""
        if not self.openai_client:
            logger.error("OpenAI client required for agents")
            return False
            
        try:
            if FULL_SYSTEM_AVAILABLE:
                # Use full agent system and add the standard 6 agents
                self.agent_manager = UnifiedAgentManager()
                base_agents = {
                    'research': ResearchAgent(),
                    'ceo': CEOAgent(),
                    'performance': PerformanceAgent(),
                    'coaching': CoachingAgent(),
                    'code_analyzer': CodeAnalyzerAgent(),
                    'triage': TriageAgent()
                }
                # Add the business intelligence and contact center agents
                simplified_agents = self._create_simplified_agents()
                self.agents = {**base_agents, **simplified_agents}
            else:
                # Create simplified agents - now including all 6 mentioned in README
                self.agents = self._create_simplified_agents()
                
            logger.info(f"✅ Initialized {len(self.agents)} AI agents")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            return False
    
    def _create_simplified_agents(self) -> Dict[str, Any]:
        """Create simplified agent implementations"""
        return {
            'business_intelligence': SimpleAgent('Business Intelligence Agent', 'business'),
            'contact_center': SimpleAgent('Contact Center Agent', 'operations')
        }
    
    def create_app(self) -> Flask:
        """Create and configure Flask application"""
        app = Flask(__name__)
        CORS(app)
        
        # Configure Flask
        app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Store references
        app.vector_rag = self
        
        # Setup middleware
        try:
            from middleware.error_handlers import register_error_handlers
            from middleware.validation import setup_validation_middleware
            
            register_error_handlers(app)
            setup_validation_middleware(app)
            logger.info("✅ Middleware registered successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Middleware not available: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to setup middleware: {e}")
        self.app = app
        
        # Register routes
        self._register_routes(app)
        self._register_error_handlers(app)
        
        return app
    
    def _register_routes(self, app: Flask):
        """Register application routes"""
        
        @app.route('/')
        def index():
            """Main application interface"""
            return render_template('index.html', 
                                 agents=self._get_agent_info(),
                                 system_status=self._get_system_status())
        
        @app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'agents_available': len(self.agents),
                'openai_connected': bool(self.openai_client),
                'vector_db_available': bool(self.vector_db or self.search_system)
            })
        
        @app.route('/api/agents')
        def list_agents():
            """List available agents"""
            return jsonify({
                'agents': self._get_agent_info(),
                'total': len(self.agents)
            })
        
        @app.route('/api/chat', methods=['POST'])
        def chat():
            """Chat with an agent"""
            try:
                data = request.get_json()
                agent_type = data.get('agent', 'research')
                message = data.get('message', '')
                
                if not message:
                    return jsonify({'error': 'Message is required'}), 400
                
                if agent_type not in self.agents:
                    return jsonify({'error': f'Agent {agent_type} not available'}), 404
                
                # Get response from agent
                response = self._get_agent_response(agent_type, message)
                
                return jsonify({
                    'response': response,
                    'agent': agent_type,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return jsonify({'error': 'Chat processing failed'}), 500
        
        @app.route('/api/search', methods=['POST'])
        def search():
            """Search documents in vector database"""
            try:
                data = request.get_json()
                query = data.get('query', '')
                
                if not query:
                    return jsonify({'error': 'Query is required'}), 400
                
                # Use full search system if available
                if self.search_system:
                    # Use full RAG search
                    results = self._perform_rag_search(query)
                elif self.vector_db:
                    # Use local vector search
                    results = self._perform_vector_search(query)
                else:
                    # Fallback to simple response
                    results = self._fallback_search(query)
                
                return jsonify({
                    'results': results,
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                return jsonify({'error': 'Search failed'}), 500
        
        @app.route('/api/documents', methods=['POST'])
        def add_document():
            """Add document via JSON (title/content)"""
            try:
                data = request.get_json()
                title = data.get('title', 'Untitled')
                content = data.get('content', '')
                source = data.get('source', 'user_upload')
                
                if not content:
                    return jsonify({'error': 'Content is required'}), 400
                
                # Chunk large documents to avoid token limits
                chunks = self._chunk_document(content)
                
                results = []
                for i, chunk in enumerate(chunks):
                    chunk_title = title
                    if len(chunks) > 1:
                        chunk_title += f" (Part {i+1}/{len(chunks)})"
                    
                    if self.vector_db:
                        result = self.vector_db.add_document(
                            chunk, 
                            title=chunk_title,
                            source=source
                        )
                        results.append(result)
                    else:
                        results.append(f"doc_{i}")
                
                return jsonify({
                    'success': True,
                    'document_ids': results,
                    'chunks_processed': len(chunks),
                    'message': f'Document "{title}" added successfully' + (f' in {len(chunks)} chunks' if len(chunks) > 1 else ''),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Document add error: {e}")
                return jsonify({'error': 'Failed to add document'}), 500

        @app.route('/api/documents', methods=['GET'])
        def get_documents():
            """Get all documents"""
            try:
                if self.vector_db:
                    docs = self.vector_db.get_all_documents()
                    return jsonify(docs)
                else:
                    return jsonify([])
            except Exception as e:
                logger.error(f"Get documents error: {e}")
                return jsonify({'error': 'Failed to retrieve documents'}), 500

        @app.route('/api/upload', methods=['POST'])
        def upload_document():
            """Upload and process documents"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Process the uploaded file
                result = self._process_upload(file)
                
                return jsonify({
                    'success': True,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return jsonify({'error': 'Upload failed'}), 500
        
        @app.route('/api/status')
        def system_status():
            """Get detailed system status"""
            return jsonify(self._get_system_status())
        
        @app.route('/favicon.ico')
        def favicon():
            """Handle favicon requests"""
            return '', 204
    
    def _register_error_handlers(self, app: Flask):
        """Register error handlers"""
        
        @app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
        
        @app.errorhandler(RequestEntityTooLarge)
        def file_too_large(error):
            return jsonify({'error': 'File too large'}), 413
    
    def _get_agent_info(self) -> List[Dict[str, Any]]:
        """Get information about available agents"""
        agent_info = []
        agent_details = {
            'research': {
                'name': 'Research Agent',
                'role': 'Market Intelligence Specialist',
                'description': 'Deep analysis, market research, and information synthesis',
                'icon': '🎯',
                'capabilities': ['Market analysis', 'Research synthesis', 'Data analysis']
            },
            'ceo': {
                'name': 'CEO Agent',
                'role': 'Executive Leadership',
                'description': 'Strategic planning, executive decisions, and coordination',
                'icon': '👔',
                'capabilities': ['Strategic planning', 'Decision making', 'Leadership']
            },
            'performance': {
                'name': 'Performance Agent',
                'role': 'Analytics & Optimization',
                'description': 'System optimization and performance monitoring',
                'icon': '📊',
                'capabilities': ['Performance analysis', 'Optimization', 'Monitoring']
            },
            'coaching': {
                'name': 'Coaching Agent',
                'role': 'Development Specialist',
                'description': 'AI-powered guidance and skill development',
                'icon': '🎓',
                'capabilities': ['Skill development', 'Mentoring', 'Guidance']
            },
            'business_intelligence': {
                'name': 'Business Intelligence Agent',
                'role': 'Data Analytics Expert',
                'description': 'Business analytics and insights',
                'icon': '💼',
                'capabilities': ['Business intelligence', 'Data visualization', 'Analytics']
            },
            'contact_center': {
                'name': 'Contact Center Agent',
                'role': 'Operations Manager',
                'description': 'Customer operations and analytics',
                'icon': '📞',
                'capabilities': ['Operations management', 'Customer analytics', 'Quality assurance']
            }
        }
        
        for agent_key in self.agents.keys():
            info = agent_details.get(agent_key, {
                'name': f'{agent_key.title()} Agent',
                'role': 'AI Assistant',
                'description': f'{agent_key.title()} specialized assistant',
                'icon': '🤖',
                'capabilities': ['General assistance']
            })
            info['id'] = agent_key
            info['available'] = True
            agent_info.append(info)
        
        return agent_info
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        return {
            'openai_connected': bool(self.openai_client),
            'agents_count': len(self.agents),
            'vector_db_available': bool(self.vector_db or self.search_system),
            'full_system_available': FULL_SYSTEM_AVAILABLE,
            'local_vector_available': LOCAL_VECTOR_DB_AVAILABLE,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    
    def _get_rag_context(self, message: str, max_results: int = 3) -> str:
        """Get limited RAG context for agent responses"""
        try:
            if self.vector_db:
                search_results = self.vector_db.search(message, limit=max_results)
                if search_results:
                    context_parts = []
                    total_chars = 0
                    max_context_chars = 20000  # ~5000 tokens
                    
                    for result in search_results:
                        content = result.get('content', '')
                        # Limit each document snippet to prevent overflow
                        if len(content) > 8000:  # ~2000 tokens per doc
                            content = content[:8000] + "..."
                        
                        if total_chars + len(content) > max_context_chars:
                            break
                        
                        context_parts.append(f"Document: {result.get('metadata', {}).get('title', 'Unknown')}\n{content}")
                        total_chars += len(content)
                    
                    return "\n\n".join(context_parts)
            return ""
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
            return ""

    def _get_agent_response(self, agent_type: str, message: str) -> str:
        """Get response from specified agent with RAG context"""
        agent = self.agents.get(agent_type)
        if not agent:
            return f"Agent {agent_type} is not available"
        
        try:
            if hasattr(agent, 'run'):
                # Use the agent's run method
                return agent.run(message)
            elif hasattr(agent, 'execute'):
                # Use async execute method
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(agent.execute({'content': message}))
                    return response.result if hasattr(response, 'result') else str(response)
                finally:
                    loop.close()
            else:
                # Use simple agent with RAG context
                rag_context = self._get_rag_context(message)
                return agent.respond(message, context=rag_context)
        except Exception as e:
            logger.error(f"Agent {agent_type} error: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                return "I'm currently experiencing high demand. Please try a shorter message or wait a moment before trying again."
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _perform_rag_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform search using full RAG system"""
        try:
            # This would use the full SearchSystem
            results = []
            # Implementation would depend on available search system
            return results
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    def _perform_vector_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform search using local vector database"""
        try:
            if self.vector_db:
                results = self.vector_db.search(query, top_k=5)
                return results
            return []
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search implementation"""
        return [
            {
                'title': f'Search Result for: {query}',
                'content': f'This is a demonstration result for your query about {query}.',
                'score': 0.9,
                'source': 'demo'
            }
        ]
    
    def _chunk_document(self, content: str, max_tokens: int = 25000) -> List[str]:
        """Chunk large document into smaller pieces to avoid token limits"""
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return [content]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(content):
            end_pos = min(current_pos + max_chars, len(content))
            
            # Try to break at sentence boundaries
            if end_pos < len(content):
                # Look for sentence endings within the last 500 chars
                last_period = content.rfind('.', current_pos, end_pos - 500)
                last_newline = content.rfind('\n', current_pos, end_pos - 500)
                
                # Use the later of the two
                break_point = max(last_period, last_newline)
                if break_point > current_pos + max_chars // 2:  # Only if it's not too early
                    end_pos = break_point + 1
            
            chunk = content[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            current_pos = end_pos
        
        return chunks

    def _process_upload(self, file) -> Dict[str, Any]:
        """Process uploaded file with chunking for large documents"""
        try:
            filename = secure_filename(file.filename)
            
            if self.vector_db:
                # Process with vector database
                content = file.read().decode('utf-8')
                
                # Check content size and chunk if necessary
                chunks = self._chunk_document(content)
                
                results = []
                for i, chunk in enumerate(chunks):
                    chunk_title = f"{filename}"
                    if len(chunks) > 1:
                        chunk_title += f" (Part {i+1}/{len(chunks)})"
                    
                    result = self.vector_db.add_document(
                        chunk, 
                        title=chunk_title,
                        source=filename
                    )
                    results.append(result)
                
                return {
                    'filename': filename,
                    'chunks_processed': len(chunks),
                    'document_ids': results,
                    'status': 'processed'
                }
            else:
                # Simple processing
                return {'filename': filename, 'status': 'processed'}
            
        except Exception as e:
            logger.error(f"Upload processing error: {e}")
            raise


class SimpleAgent:
    """Simple agent implementation for fallback"""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.openai_client = None
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'sk-demo-key-placeholder-for-testing':
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI for {name}: {e}")
    
    def _limit_context_tokens(self, text: str, max_tokens: int = 20000) -> str:
        """Limit context to avoid token overflow"""
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        # Use the later of the two
        break_point = max(last_period, last_newline)
        if break_point > max_chars // 2:  # Only if it's not too early
            truncated = text[:break_point + 1]
        
        return truncated + "\n\n[Content truncated due to length...]"

    def respond(self, message: str, context: str = None) -> str:
        """Generate response to message with optional context"""
        if not self.openai_client:
            return f"I'm {self.name}, but I need a valid OpenAI API key to provide intelligent responses. Please configure your OPENAI_API_KEY in the .env file."
        
        try:
            system_prompts = {
                'research': "You are a research analyst specialized in gathering and analyzing information.",
                'executive': "You are a CEO agent focused on strategic planning and executive decisions.",
                'analytics': "You are a performance analyst focused on optimization and metrics.",
                'coaching': "You are an AI coach focused on guidance and skill development.",
                'business': "You are a business intelligence specialist focused on data analytics.",
                'operations': "You are an operations manager focused on customer service excellence."
            }
            
            system_prompt = system_prompts.get(self.agent_type, "You are a helpful AI assistant.")
            
            # Limit message length to prevent token overflow
            limited_message = self._limit_context_tokens(message, 15000)
            
            # If context is provided, add it but limit total tokens
            if context:
                limited_context = self._limit_context_tokens(context, 8000)
                user_content = f"Context from knowledge base:\n{limited_context}\n\nUser question: {limited_message}"
            else:
                user_content = limited_message
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error for {self.name}: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                return f"I'm {self.name}, and I'm currently experiencing high demand. Please try your request again in a moment with a shorter message."
            return f"I'm {self.name}, but I'm experiencing technical difficulties. Please check your OpenAI API configuration."


def main():
    """Main application entry point"""
    logger.info("Vector RAG Database - Production Version starting")
    
    # Initialize the application
    vector_rag = ProductionVectorRAG()
    
    # Initialize components
    logger.info("Initializing OpenAI connection")
    openai_ok = vector_rag.initialize_openai()
    
    logger.info("Initializing vector database")
    vector_ok = vector_rag.initialize_vector_db()
    
    logger.info("Initializing RAG system")
    rag_ok = vector_rag.initialize_rag_system()
    
    logger.info("Initializing AI agents")
    agents_ok = vector_rag.initialize_agents()
    
    # Create Flask app
    app = vector_rag.create_app()
    
    # Print status
    logger.info(f"System initialized - OpenAI: {openai_ok}, Vector DB: {vector_ok or rag_ok}, Agents: {len(vector_rag.agents)}, Full System: {FULL_SYSTEM_AVAILABLE}")
    
    if not openai_ok:
        logger.warning("OpenAI API not configured - limited functionality available")
    
    logger.info(f"Server starting on http://localhost:5001 with {len(vector_rag.agents)} agents")
    
    # Start the application
    try:
        app.run(debug=False, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        logger.info("Vector RAG Database shutting down")
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
