"""
Document management API controller
"""

import logging
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify
import uuid

from src.application.services.document_service import DocumentService
from src.domain.entities import AgentType
from src.presentation.responses import (
    create_response_builder, ApiException, ValidationException
)
from src.infrastructure.container import get_container

logger = logging.getLogger(__name__)

# Create Blueprint
document_bp = Blueprint('documents', __name__, url_prefix='/api/documents')


def get_request_id() -> str:
    """Generate or extract request ID"""
    return request.headers.get('X-Request-ID', str(uuid.uuid4()))


def get_current_user_id() -> Optional[str]:
    """Get current user ID from request context"""
    # This would typically come from authentication middleware
    return getattr(request, 'current_user', {}).get('user_id')


@document_bp.route('', methods=['POST'])
async def create_document():
    """Create a new document"""
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    
    try:
        # Get service from container
        container = get_container()
        document_service = container.resolve(DocumentService)
        
        # Validate request
        if not request.is_json:
            raise ValidationException("content_type", "Request must be JSON")
        
        data = request.get_json()
        if not data:
            raise ValidationException("body", "Request body is required")
        
        # Extract parameters
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()
        source = data.get('source', 'api').strip()
        metadata = data.get('metadata', {})
        
        # Create document
        document = await document_service.create_document(
            title=title,
            content=content,
            source=source,
            created_by=get_current_user_id(),
            metadata=metadata
        )
        
        # Return success response
        return jsonify(response_builder.success(
            data={
                "id": document.id,
                "title": document.title,
                "status": document.status.value,
                "created_at": document.created_at.isoformat(),
                "metadata": document.metadata
            },
            message="Document created successfully"
        ).to_dict())
        
    except ApiException as e:
        response = e.to_response(request_id)
        return jsonify(response.to_dict()), e.status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in create_document: {e}")
        response = response_builder.system_error(str(e))
        return jsonify(response.to_dict()), 500


@document_bp.route('/<document_id>', methods=['GET'])
async def get_document(document_id: str):
    """Get document by ID"""
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    
    try:
        container = get_container()
        document_service = container.resolve(DocumentService)
        
        document = await document_service.get_document(document_id)
        
        return jsonify(response_builder.success(
            data={
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "source": document.source,
                "status": document.status.value,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "created_by": document.created_by,
                "metadata": document.metadata
            }
        ).to_dict())
        
    except ApiException as e:
        response = e.to_response(request_id)
        return jsonify(response.to_dict()), e.status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in get_document: {e}")
        response = response_builder.system_error(str(e))
        return jsonify(response.to_dict()), 500


@document_bp.route('/<document_id>', methods=['PUT'])
async def update_document(document_id: str):
    """Update document by ID"""
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    
    try:
        container = get_container()
        document_service = container.resolve(DocumentService)
        
        if not request.is_json:
            raise ValidationException("content_type", "Request must be JSON")
        
        data = request.get_json()
        if not data:
            raise ValidationException("body", "Request body is required")
        
        # Extract update parameters
        title = data.get('title')
        content = data.get('content')
        metadata = data.get('metadata')
        
        # Update document
        document = await document_service.update_document(
            document_id=document_id,
            title=title,
            content=content,
            metadata=metadata,
            updated_by=get_current_user_id()
        )
        
        return jsonify(response_builder.success(
            data={
                "id": document.id,
                "title": document.title,
                "status": document.status.value,
                "updated_at": document.updated_at.isoformat(),
                "metadata": document.metadata
            },
            message="Document updated successfully"
        ).to_dict())
        
    except ApiException as e:
        response = e.to_response(request_id)
        return jsonify(response.to_dict()), e.status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in update_document: {e}")
        response = response_builder.system_error(str(e))
        return jsonify(response.to_dict()), 500


@document_bp.route('/<document_id>', methods=['DELETE'])
async def delete_document(document_id: str):
    """Delete document by ID"""
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    
    try:
        container = get_container()
        document_service = container.resolve(DocumentService)
        
        success = await document_service.delete_document(
            document_id=document_id,
            deleted_by=get_current_user_id()
        )
        
        if success:
            return jsonify(response_builder.success(
                message="Document deleted successfully"
            ).to_dict())
        else:
            response = response_builder.system_error("Failed to delete document")
            return jsonify(response.to_dict()), 500
        
    except ApiException as e:
        response = e.to_response(request_id)
        return jsonify(response.to_dict()), e.status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in delete_document: {e}")
        response = response_builder.system_error(str(e))
        return jsonify(response.to_dict()), 500


@document_bp.route('', methods=['GET'])
async def list_documents():
    """List documents with pagination"""
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    
    try:
        container = get_container()
        document_service = container.resolve(DocumentService)
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        
        # Get documents
        documents, total_count = await document_service.list_documents(
            page=page,
            page_size=page_size,
            user_id=get_current_user_id()
        )
        
        # Format documents for response
        document_data = []
        for doc in documents:
            document_data.append({
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "status": doc.status.value,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
                "created_by": doc.created_by,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "metadata": doc.metadata
            })
        
        return jsonify(response_builder.paginated_success(
            data=document_data,
            page=page,
            page_size=page_size,
            total_count=total_count
        ).to_dict())
        
    except ApiException as e:
        response = e.to_response(request_id)
        return jsonify(response.to_dict()), e.status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in list_documents: {e}")
        response = response_builder.system_error(str(e))
        return jsonify(response.to_dict()), 500


@document_bp.route('/search', methods=['POST'])
async def search_documents():
    """Search documents"""
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    
    try:
        container = get_container()
        document_service = container.resolve(DocumentService)
        
        if not request.is_json:
            raise ValidationException("content_type", "Request must be JSON")
        
        data = request.get_json()
        if not data:
            raise ValidationException("body", "Request body is required")
        
        # Extract search parameters
        query = data.get('query', '').strip()
        limit = data.get('limit', 10)
        filters = data.get('filters', {})
        
        # Perform search
        results = await document_service.search_documents(
            query=query,
            limit=limit,
            filters=filters,
            user_id=get_current_user_id()
        )
        
        # Format results
        search_data = []
        for result in results:
            search_data.append({
                "document_id": result.document_id,
                "content": result.content,
                "metadata": result.metadata,
                "relevance_score": result.relevance_score,
                "distance": result.distance
            })
        
        return jsonify(response_builder.success(
            data={
                "query": query,
                "results": search_data,
                "count": len(search_data)
            },
            message=f"Found {len(search_data)} matching documents"
        ).to_dict())
        
    except ApiException as e:
        response = e.to_response(request_id)
        return jsonify(response.to_dict()), e.status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in search_documents: {e}")
        response = response_builder.system_error(str(e))
        return jsonify(response.to_dict()), 500


@document_bp.route('/statistics', methods=['GET'])
async def get_document_statistics():
    """Get document statistics"""
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    
    try:
        container = get_container()
        document_service = container.resolve(DocumentService)
        
        statistics = await document_service.get_document_statistics()
        
        return jsonify(response_builder.success(
            data=statistics,
            message="Document statistics retrieved successfully"
        ).to_dict())
        
    except ApiException as e:
        response = e.to_response(request_id)
        return jsonify(response.to_dict()), e.status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in get_document_statistics: {e}")
        response = response_builder.system_error(str(e))
        return jsonify(response.to_dict()), 500


# Error handlers for the blueprint
@document_bp.errorhandler(ApiException)
def handle_api_exception(e: ApiException):
    """Handle API exceptions"""
    request_id = get_request_id()
    response = e.to_response(request_id)
    return jsonify(response.to_dict()), e.status_code


@document_bp.errorhandler(Exception)
def handle_unexpected_exception(e: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error in documents API: {e}")
    request_id = get_request_id()
    response_builder = create_response_builder(request_id)
    response = response_builder.system_error("An unexpected error occurred")
    return jsonify(response.to_dict()), 500