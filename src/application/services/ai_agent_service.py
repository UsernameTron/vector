"""
AI Agent service for managing agent interactions
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.domain.entities import AgentResponse, AgentType, SearchQuery
from src.domain.interfaces import (
    IAIAgent, IDocumentRepository, IEventPublisher, ILoggingService, ICacheService
)
from src.presentation.responses import (
    ValidationException, NotFoundException, ExternalServiceException
)
from src.infrastructure.container import scoped

logger = logging.getLogger(__name__)


@scoped()
class AIAgentService:
    """Service for AI agent operations"""
    
    def __init__(
        self,
        document_repository: IDocumentRepository,
        event_publisher: IEventPublisher,
        logging_service: ILoggingService,
        cache_service: ICacheService,
        agents: Dict[AgentType, IAIAgent]
    ):
        self.document_repository = document_repository
        self.event_publisher = event_publisher
        self.logging_service = logging_service
        self.cache_service = cache_service
        self.agents = agents
    
    async def process_query(
        self,
        agent_type: AgentType,
        query: str,
        user_id: Optional[str] = None,
        use_context: bool = True,
        context_limit: int = 3
    ) -> AgentResponse:
        """Process query with specified AI agent"""
        
        # Validate input
        if not query or not query.strip():
            raise ValidationException("query", "Query is required")
        
        if agent_type not in self.agents:
            raise NotFoundException("Agent", agent_type.value)
        
        agent = self.agents[agent_type]
        
        try:
            # Get relevant context from documents if requested
            context = None
            if use_context:
                context = await self._get_relevant_context(query, context_limit)
            
            # Process query with agent
            start_time = datetime.now()
            response = await agent.process_query(query, context)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update response with metadata
            response.user_id = user_id
            response.processing_time = processing_time
            response.metadata.update({
                "context_used": context is not None,
                "context_documents": context_limit if context else 0
            })
            
            # Log interaction
            await self.logging_service.log_info(
                f"AI agent query processed: {agent_type.value}",
                {
                    "agent_type": agent_type.value,
                    "user_id": user_id,
                    "query_length": len(query),
                    "response_length": len(response.response),
                    "processing_time": processing_time
                }
            )
            
            # Publish event
            await self.event_publisher.publish_event(
                "agent.query_processed",
                {
                    "agent_type": agent_type.value,
                    "user_id": user_id,
                    "query": query[:100],  # Truncated for privacy
                    "processing_time": processing_time,
                    "timestamp": response.timestamp.isoformat()
                }
            )
            
            # Cache response for potential reuse
            await self._cache_response(query, response)
            
            return response
            
        except Exception as e:
            await self.logging_service.log_error(
                f"AI agent query failed: {agent_type.value}",
                e,
                {"agent_type": agent_type.value, "user_id": user_id, "query": query[:100]}
            )
            
            if "rate limit" in str(e).lower():
                raise ExternalServiceException("OpenAI", "Rate limit exceeded")
            elif "api key" in str(e).lower():
                raise ExternalServiceException("OpenAI", "API authentication failed")
            else:
                raise ExternalServiceException("AI Agent", f"Query processing failed: {e}")
    
    async def get_available_agents(self) -> Dict[str, Dict]:
        """Get information about available agents"""
        try:
            agents_info = {}
            
            for agent_type, agent in self.agents.items():
                try:
                    health_status = await agent.get_health_status()
                    capabilities = agent.get_capabilities()
                    
                    agents_info[agent_type.value] = {
                        "type": agent_type.value,
                        "capabilities": capabilities,
                        "healthy": health_status.healthy,
                        "status_message": health_status.message
                    }
                    
                except Exception as e:
                    agents_info[agent_type.value] = {
                        "type": agent_type.value,
                        "capabilities": [],
                        "healthy": False,
                        "status_message": f"Health check failed: {e}"
                    }
            
            return agents_info
            
        except Exception as e:
            await self.logging_service.log_error("Failed to get available agents", e)
            raise
    
    async def get_agent_capabilities(self, agent_type: AgentType) -> List[str]:
        """Get capabilities for specific agent"""
        if agent_type not in self.agents:
            raise NotFoundException("Agent", agent_type.value)
        
        try:
            agent = self.agents[agent_type]
            return agent.get_capabilities()
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to get agent capabilities: {agent_type.value}",
                e
            )
            raise
    
    async def check_agent_health(self, agent_type: AgentType) -> Dict:
        """Check health status of specific agent"""
        if agent_type not in self.agents:
            raise NotFoundException("Agent", agent_type.value)
        
        try:
            agent = self.agents[agent_type]
            health_status = await agent.get_health_status()
            
            return {
                "agent_type": agent_type.value,
                "healthy": health_status.healthy,
                "message": health_status.message,
                "details": health_status.details,
                "checked_at": health_status.checked_at.isoformat()
            }
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to check agent health: {agent_type.value}",
                e
            )
            raise
    
    async def get_query_history(
        self,
        user_id: str,
        agent_type: Optional[AgentType] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get query history for user (from cache/logs)"""
        
        # This would typically query a history storage system
        # For now, return empty list as this requires additional infrastructure
        try:
            # Implementation would depend on how query history is stored
            # Could use cache service, dedicated database, or event store
            
            await self.logging_service.log_info(
                f"Query history requested",
                {"user_id": user_id, "agent_type": agent_type.value if agent_type else "all"}
            )
            
            return []  # Placeholder
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to get query history for user: {user_id}",
                e
            )
            raise
    
    async def _get_relevant_context(self, query: str, limit: int) -> Optional[str]:
        """Get relevant document context for the query"""
        try:
            # Search for relevant documents
            search_query = SearchQuery(
                query=query,
                limit=limit
            )
            
            search_results = await self.document_repository.search(search_query)
            
            if not search_results:
                return None
            
            # Combine relevant content
            context_parts = []
            for result in search_results:
                if result.relevance_score > 0.5:  # Only include relevant results
                    # Truncate content to reasonable size
                    content = result.content[:500] + "..." if len(result.content) > 500 else result.content
                    context_parts.append(f"Document: {result.metadata.get('title', 'Untitled')}\nContent: {content}")
            
            return "\n\n".join(context_parts) if context_parts else None
            
        except Exception as e:
            # Don't fail the agent query if context retrieval fails
            await self.logging_service.log_warning(
                f"Failed to get context for query: {e}",
                {"query": query[:100]}
            )
            return None
    
    async def _cache_response(self, query: str, response: AgentResponse):
        """Cache response for potential reuse"""
        try:
            # Create cache key
            cache_key = f"agent_response:{hash(query + response.agent_type.value)}"
            
            # Cache for 1 hour
            await self.cache_service.set(
                cache_key,
                {
                    "query": query,
                    "response": response.response,
                    "agent_type": response.agent_type.value,
                    "timestamp": response.timestamp.isoformat()
                },
                ttl=3600
            )
            
        except Exception as e:
            # Don't fail the main operation if caching fails
            await self.logging_service.log_warning(
                f"Failed to cache response: {e}",
                {"query": query[:100]}
            )