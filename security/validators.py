"""
Security validation middleware for Flask application
Handles input sanitization and validation
"""

import re
import html
import json
from typing import Any, Dict, List, Optional, Union
from flask import request, jsonify
from functools import wraps
import bleach
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"('|(\\'))|(;|(\\);)|(--|(\\)--)|(\/\*(.|\n)*\*\/)",
        r"\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE)?|INSERT|MASTER|SELECT|UNION|UPDATE|WHERE)\b",
        r"\b(script|javascript|vbscript|onload|onerror|onclick)\b",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    # Allowed HTML tags for content sanitization
    ALLOWED_HTML_TAGS = ['b', 'i', 'u', 'em', 'strong', 'p', 'br', 'ul', 'ol', 'li']
    ALLOWED_HTML_ATTRIBUTES = {}
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML content to prevent XSS"""
        if not isinstance(text, str):
            return str(text)
        
        # First escape HTML entities
        sanitized = html.escape(text)
        
        # Use bleach for more sophisticated cleaning if needed
        try:
            sanitized = bleach.clean(
                text, 
                tags=InputValidator.ALLOWED_HTML_TAGS, 
                attributes=InputValidator.ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
        except Exception as e:
            logger.warning(f"Bleach sanitization failed, using html.escape: {e}")
            sanitized = html.escape(text)
        
        return sanitized
    
    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        """Detect potential SQL injection attempts"""
        if not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def detect_xss(text: str) -> bool:
        """Detect potential XSS attempts"""
        if not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def validate_string(text: str, max_length: int = 10000, min_length: int = 0) -> tuple[bool, str]:
        """Validate string input with length and content checks"""
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if len(text) < min_length:
            return False, f"Input too short (minimum {min_length} characters)"
        
        if len(text) > max_length:
            return False, f"Input too long (maximum {max_length} characters)"
        
        if InputValidator.detect_sql_injection(text):
            return False, "Potential SQL injection detected"
        
        if InputValidator.detect_xss(text):
            return False, "Potential XSS attack detected"
        
        return True, "Valid"
    
    @staticmethod
    def validate_json_payload(data: Dict[str, Any], required_fields: List[str] = None, max_size: int = 1024*1024) -> tuple[bool, str, Dict[str, Any]]:
        """Validate JSON payload"""
        if not isinstance(data, dict):
            return False, "Invalid JSON format", {}
        
        # Check payload size
        try:
            json_str = json.dumps(data)
            if len(json_str.encode('utf-8')) > max_size:
                return False, f"Payload too large (maximum {max_size} bytes)", {}
        except Exception:
            return False, "Invalid JSON serialization", {}
        
        # Check required fields
        if required_fields:
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}", {}
        
        # Sanitize string values
        sanitized_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                is_valid, error = InputValidator.validate_string(value)
                if not is_valid:
                    return False, f"Invalid value for field '{key}': {error}", {}
                sanitized_data[key] = InputValidator.sanitize_html(value)
            else:
                sanitized_data[key] = value
        
        return True, "Valid", sanitized_data
    
    @staticmethod
    def validate_query_params(params: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Validate query parameters"""
        sanitized_params = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                is_valid, error = InputValidator.validate_string(value, max_length=1000)
                if not is_valid:
                    return False, f"Invalid query parameter '{key}': {error}", {}
                sanitized_params[key] = InputValidator.sanitize_html(value)
            elif isinstance(value, (int, float, bool)):
                sanitized_params[key] = value
            else:
                # Convert to string and validate
                str_value = str(value)
                is_valid, error = InputValidator.validate_string(str_value, max_length=1000)
                if not is_valid:
                    return False, f"Invalid query parameter '{key}': {error}", {}
                sanitized_params[key] = InputValidator.sanitize_html(str_value)
        
        return True, "Valid", sanitized_params


def validate_input(required_json_fields: List[str] = None, validate_query: bool = True):
    """Decorator for input validation middleware"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Validate JSON payload if present
                if request.is_json and request.get_json():
                    is_valid, error, sanitized_data = InputValidator.validate_json_payload(
                        request.get_json(), 
                        required_json_fields
                    )
                    if not is_valid:
                        logger.warning(f"Invalid JSON payload: {error}")
                        return jsonify({'error': f'Invalid input: {error}'}), 400
                    
                    # Replace request data with sanitized version
                    request._cached_json = sanitized_data
                
                # Validate query parameters
                if validate_query and request.args:
                    is_valid, error, sanitized_params = InputValidator.validate_query_params(
                        request.args.to_dict()
                    )
                    if not is_valid:
                        logger.warning(f"Invalid query parameters: {error}")
                        return jsonify({'error': f'Invalid query parameters: {error}'}), 400
                
                return f(*args, **kwargs)
            
            except Exception as e:
                logger.error(f"Input validation error: {str(e)}")
                return jsonify({'error': 'Input validation failed'}), 400
        
        return decorated_function
    return decorator


def sanitize_response(response_data: Union[Dict, List, str]) -> Union[Dict, List, str]:
    """Sanitize response data before sending to client"""
    if isinstance(response_data, dict):
        return {key: sanitize_response(value) for key, value in response_data.items()}
    elif isinstance(response_data, list):
        return [sanitize_response(item) for item in response_data]
    elif isinstance(response_data, str):
        return InputValidator.sanitize_html(response_data)
    else:
        return response_data