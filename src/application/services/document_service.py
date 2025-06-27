"""
Document management service with single responsibility
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.domain.entities import Document, SearchQuery, SearchResult, DocumentStatus
from src.domain.interfaces import IDocumentRepository, IEventPublisher, ILoggingService
from src.presentation.responses import (
    ValidationException, NotFoundException, BusinessRuleException, ConflictException
)
from src.infrastructure.container import scoped

logger = logging.getLogger(__name__)


@scoped()
class DocumentService:
    """Service for document management operations"""
    
    def __init__(
        self,
        document_repository: IDocumentRepository,
        event_publisher: IEventPublisher,
        logging_service: ILoggingService
    ):
        self.document_repository = document_repository
        self.event_publisher = event_publisher
        self.logging_service = logging_service
    
    async def create_document(
        self,
        title: str,
        content: str,
        source: str = "api",
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a new document with validation"""
        
        # Validate input
        await self._validate_document_input(title, content)
        
        # Check for duplicates
        await self._check_duplicate_content(content)
        
        # Create document entity
        document = Document(
            title=title.strip(),
            content=content.strip(),
            source=source,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        try:
            # Save document
            created_document = await self.document_repository.create(document)
            
            # Log event
            await self.logging_service.log_info(
                f"Document created: {created_document.id}",
                {"document_id": created_document.id, "created_by": created_by}
            )
            
            # Publish event
            await self.event_publisher.publish_event(
                "document.created",
                {
                    "document_id": created_document.id,
                    "title": created_document.title,
                    "created_by": created_by,
                    "timestamp": created_document.created_at.isoformat()
                }
            )
            
            return created_document
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to create document: {title}",
                e,
                {"title": title, "created_by": created_by}
            )
            raise
    
    async def get_document(self, document_id: str) -> Document:
        """Get document by ID"""
        if not document_id or not document_id.strip():
            raise ValidationException("document_id", "Document ID is required")
        
        document = await self.document_repository.get_by_id(document_id.strip())
        if not document:
            raise NotFoundException("Document", document_id)
        
        return document
    
    async def update_document(
        self,
        document_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        updated_by: Optional[str] = None
    ) -> Document:
        """Update existing document"""
        
        # Get existing document
        document = await self.get_document(document_id)
        
        # Validate updates
        if title is not None:
            if not title.strip():
                raise ValidationException("title", "Title cannot be empty")
            document.title = title.strip()
        
        if content is not None:
            if not content.strip():
                raise ValidationException("content", "Content cannot be empty")
            
            # Check for duplicate content if changed
            if content.strip() != document.content:
                await self._check_duplicate_content(content.strip())
            
            document.update_content(content.strip())
        
        if metadata is not None:
            document.metadata.update(metadata)
        
        try:
            # Update document
            updated_document = await self.document_repository.update(document)
            
            # Log event
            await self.logging_service.log_info(
                f"Document updated: {document_id}",
                {"document_id": document_id, "updated_by": updated_by}
            )
            
            # Publish event
            await self.event_publisher.publish_event(
                "document.updated",
                {
                    "document_id": document_id,
                    "title": updated_document.title,
                    "updated_by": updated_by,
                    "timestamp": updated_document.updated_at.isoformat()
                }
            )
            
            return updated_document
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to update document: {document_id}",
                e,
                {"document_id": document_id, "updated_by": updated_by}
            )
            raise
    
    async def delete_document(self, document_id: str, deleted_by: Optional[str] = None) -> bool:
        """Delete document"""
        
        # Verify document exists
        document = await self.get_document(document_id)
        
        try:
            # Delete document
            success = await self.document_repository.delete(document_id)
            
            if success:
                # Log event
                await self.logging_service.log_info(
                    f"Document deleted: {document_id}",
                    {"document_id": document_id, "deleted_by": deleted_by}
                )
                
                # Publish event
                await self.event_publisher.publish_event(
                    "document.deleted",
                    {
                        "document_id": document_id,
                        "title": document.title,
                        "deleted_by": deleted_by,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return success
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to delete document: {document_id}",
                e,
                {"document_id": document_id, "deleted_by": deleted_by}
            )
            raise
    
    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None
    ) -> tuple[List[Document], int]:
        """List documents with pagination"""
        
        # Validate pagination parameters
        if page < 1:
            raise ValidationException("page", "Page must be greater than 0")
        
        if page_size < 1 or page_size > 100:
            raise ValidationException("page_size", "Page size must be between 1 and 100")
        
        try:
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Get documents
            documents = await self.document_repository.get_all(offset, page_size)
            
            # Get total count
            total_count = await self.document_repository.get_count()
            
            # Log access
            await self.logging_service.log_info(
                f"Documents listed: page {page}, size {page_size}",
                {"page": page, "page_size": page_size, "user_id": user_id}
            )
            
            return documents, total_count
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to list documents",
                e,
                {"page": page, "page_size": page_size, "user_id": user_id}
            )
            raise
    
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Search documents"""
        
        # Validate search parameters
        if not query or not query.strip():
            raise ValidationException("query", "Search query is required")
        
        if limit < 1 or limit > 50:
            raise ValidationException("limit", "Limit must be between 1 and 50")
        
        try:
            # Create search query
            search_query = SearchQuery(
                query=query.strip(),
                limit=limit,
                filters=filters or {},
                user_id=user_id
            )
            
            # Perform search
            results = await self.document_repository.search(search_query)
            
            # Log search
            await self.logging_service.log_info(
                f"Documents searched: '{query}' returned {len(results)} results",
                {"query": query, "limit": limit, "user_id": user_id, "results_count": len(results)}
            )
            
            # Publish search event
            await self.event_publisher.publish_event(
                "document.searched",
                {
                    "query": query,
                    "results_count": len(results),
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return results
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to search documents: '{query}'",
                e,
                {"query": query, "limit": limit, "user_id": user_id}
            )
            raise
    
    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get document statistics"""
        try:
            total_count = await self.document_repository.get_count()
            
            # Get recent documents for additional stats
            recent_documents = await self.document_repository.get_all(0, 100)
            
            # Calculate statistics
            status_counts = {}
            source_counts = {}
            recent_count = 0
            
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            for doc in recent_documents:
                # Count by status
                status = doc.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Count by source
                source_counts[doc.source] = source_counts.get(doc.source, 0) + 1
                
                # Count recent documents (today)
                if doc.created_at >= cutoff_date:
                    recent_count += 1
            
            statistics = {
                "total_documents": total_count,
                "documents_today": recent_count,
                "status_distribution": status_counts,
                "source_distribution": source_counts,
                "last_updated": datetime.now().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            await self.logging_service.log_error("Failed to get document statistics", e)
            raise
    
    async def _validate_document_input(self, title: str, content: str):
        """Validate document input parameters"""
        if not title or not title.strip():
            raise ValidationException("title", "Title is required")
        
        if not content or not content.strip():
            raise ValidationException("content", "Content is required")
        
        if len(title.strip()) > 500:
            raise ValidationException("title", "Title must be 500 characters or less")
        
        if len(content.strip()) > 1000000:  # 1MB limit
            raise ValidationException("content", "Content must be 1MB or less")
        
        # Check for suspicious content patterns
        suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        content_lower = content.lower()
        
        for pattern in suspicious_patterns:
            if pattern in content_lower:
                raise ValidationException("content", f"Content contains suspicious pattern: {pattern}")
    
    async def _check_duplicate_content(self, content: str):
        """Check for duplicate content"""
        try:
            # Search for similar content
            search_query = SearchQuery(
                query=content[:200],  # Use first 200 chars for similarity check
                limit=5
            )
            
            similar_docs = await self.document_repository.search(search_query)
            
            # Check for exact matches
            for result in similar_docs:
                if result.relevance_score > 0.95:  # Very high similarity
                    raise ConflictException(
                        "Document with very similar content already exists",
                        "document"
                    )
                    
        except ConflictException:
            raise
        except Exception as e:
            # Don't fail document creation due to duplicate check issues
            await self.logging_service.log_warning(
                f"Duplicate check failed: {e}",
                {"content_length": len(content)}
            )