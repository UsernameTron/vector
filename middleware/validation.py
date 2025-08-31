"""Request validation middleware and schemas"""

import logging
from functools import wraps
from flask import request, jsonify
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import re
import bleach

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error for field '{field}': {message}")


class InputSanitizer:
    """Sanitize and validate user inputs"""
    
    # Allow basic HTML tags for rich text content
    ALLOWED_HTML_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
    ]
    
    ALLOWED_HTML_ATTRIBUTES = {
        '*': ['class', 'id'],
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'width', 'height']
    }
    
    @staticmethod
    def sanitize_html(content: str, allow_html: bool = True) -> str:
        """Sanitize HTML content to prevent XSS attacks"""
        if not isinstance(content, str):
            return str(content)
        
        if allow_html:
            return bleach.clean(
                content,
                tags=InputSanitizer.ALLOWED_HTML_TAGS,
                attributes=InputSanitizer.ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
        else:
            return bleach.clean(content, tags=[], attributes={}, strip=True)
    
    @staticmethod
    def validate_string(value: Any, field_name: str, min_length: int = 0, 
                       max_length: int = None, pattern: str = None,
                       allow_empty: bool = False) -> str:
        """Validate and sanitize string input"""
        if value is None:
            if allow_empty:
                return ""
            raise ValidationError(field_name, "Field is required")
        
        if not isinstance(value, str):
            value = str(value)
        
        # Trim whitespace
        value = value.strip()
        
        if not value and not allow_empty:
            raise ValidationError(field_name, "Field cannot be empty")
        
        if len(value) < min_length:
            raise ValidationError(field_name, f"Must be at least {min_length} characters long")
        
        if max_length and len(value) > max_length:
            raise ValidationError(field_name, f"Must be no more than {max_length} characters long")
        
        if pattern and not re.match(pattern, value):
            raise ValidationError(field_name, "Invalid format")
        
        return value
    
    @staticmethod
    def validate_integer(value: Any, field_name: str, min_value: int = None,
                        max_value: int = None) -> int:
        """Validate integer input"""
        if value is None:
            raise ValidationError(field_name, "Field is required")
        
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(field_name, "Must be a valid integer")
        
        if min_value is not None and value < min_value:
            raise ValidationError(field_name, f"Must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(field_name, f"Must be no more than {max_value}")
        
        return value
    
    @staticmethod
    def validate_email(email: str, field_name: str) -> str:
        """Validate email format"""
        email = InputSanitizer.validate_string(email, field_name, min_length=1)
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValidationError(field_name, "Invalid email format")
        
        return email.lower()
    
    @staticmethod
    def validate_agent_type(agent_type: str, field_name: str) -> str:
        """Validate AI agent type"""
        agent_type = InputSanitizer.validate_string(agent_type, field_name, min_length=1)
        
        valid_agents = [
            'research', 'ceo', 'performance', 'coaching', 
            'business_intelligence', 'contact_center'
        ]
        
        if agent_type not in valid_agents:
            raise ValidationError(field_name, f"Invalid agent type. Must be one of: {', '.join(valid_agents)}")
        
        return agent_type


class RequestValidator:
    """Validate incoming requests"""
    
    @staticmethod
    def validate_chat_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chat API request"""
        validated = {}
        
        # Validate message
        validated['message'] = InputSanitizer.validate_string(
            data.get('message'), 
            'message', 
            min_length=1, 
            max_length=10000
        )
        
        # Validate agent type
        validated['agent_type'] = InputSanitizer.validate_agent_type(
            data.get('agent_type'), 
            'agent_type'
        )
        
        # Optional context limit
        if 'context_limit' in data:
            validated['context_limit'] = InputSanitizer.validate_integer(
                data['context_limit'], 
                'context_limit', 
                min_value=1, 
                max_value=20
            )
        else:
            validated['context_limit'] = 3
        
        return validated
    
    @staticmethod
    def validate_search_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate search API request"""
        validated = {}
        
        # Validate query
        validated['query'] = InputSanitizer.validate_string(
            data.get('query'), 
            'query', 
            min_length=1, 
            max_length=1000
        )
        
        # Optional limit
        if 'limit' in data:
            validated['limit'] = InputSanitizer.validate_integer(
                data['limit'], 
                'limit', 
                min_value=1, 
                max_value=50
            )
        else:
            validated['limit'] = 10
        
        return validated
    
    @staticmethod
    def validate_upload_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document upload request"""
        validated = {}
        
        # Validate title
        validated['title'] = InputSanitizer.validate_string(
            data.get('title'), 
            'title', 
            min_length=1, 
            max_length=200
        )
        
        # Validate content
        validated['content'] = InputSanitizer.validate_string(
            data.get('content'), 
            'content', 
            min_length=1, 
            max_length=1000000  # 1MB of text
        )
        
        # Sanitize content to prevent XSS
        validated['content'] = InputSanitizer.sanitize_html(validated['content'])
        
        # Optional source
        if 'source' in data:
            validated['source'] = InputSanitizer.validate_string(
                data['source'], 
                'source', 
                max_length=100, 
                allow_empty=True
            )
        else:
            validated['source'] = 'api'
        
        return validated


def validate_json_request(validator_func):
    """Decorator to validate JSON requests"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            request_id = getattr(request, 'request_id', str(uuid.uuid4()))
            
            # Check if request is JSON
            if not request.is_json:
                error_response = {
                    "error": True,
                    "status": "error",
                    "message": "Request must be JSON",
                    "error_code": "INVALID_CONTENT_TYPE",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                }
                return jsonify(error_response), 400
            
            # Get JSON data
            try:
                data = request.get_json()
                if data is None:
                    raise ValueError("No JSON data provided")
            except Exception as e:
                error_response = {
                    "error": True,
                    "status": "error", 
                    "message": "Invalid JSON format",
                    "error_code": "INVALID_JSON",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                }
                logger.warning(f"Invalid JSON in request to {request.path}: {e}")
                return jsonify(error_response), 400
            
            # Validate using provided validator function
            try:
                validated_data = validator_func(data)
                
                # Add validated data to request object for use in view
                request.validated_data = validated_data
                
                return f(*args, **kwargs)
                
            except ValidationError as e:
                error_response = {
                    "error": True,
                    "status": "error",
                    "message": f"Validation failed: {e.message}",
                    "error_code": "VALIDATION_ERROR",
                    "field": e.field,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                }
                logger.warning(f"Validation error for {request.path}: {e}")
                return jsonify(error_response), 400
            
            except Exception as e:
                error_response = {
                    "error": True,
                    "status": "error",
                    "message": "Request validation failed",
                    "error_code": "VALIDATION_ERROR",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                }
                logger.error(f"Unexpected validation error for {request.path}: {e}")
                return jsonify(error_response), 500
        
        return decorated_function
    return decorator


def add_request_id_middleware(app):
    """Add request ID to all requests for tracking"""
    
    @app.before_request
    def add_request_id():
        request.request_id = str(uuid.uuid4())
        logger.info(f"Request {request.request_id}: {request.method} {request.path}")
    
    @app.after_request
    def add_request_id_header(response):
        if hasattr(request, 'request_id'):
            response.headers['X-Request-ID'] = request.request_id
        return response


def setup_validation_middleware(app):
    """Setup all validation middleware"""
    add_request_id_middleware(app)
    logger.info("Validation middleware setup complete")