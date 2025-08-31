"""Global error handlers for Flask application"""

import logging
import traceback
from flask import jsonify, request, current_app
from werkzeug.exceptions import HTTPException
import uuid
from datetime import datetime

from utils.error_handler import ErrorRecovery, ErrorCategories

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    """Register all global error handlers with the Flask app"""
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle 400 Bad Request errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": "Bad request. Please check your input and try again.",
            "error_code": "BAD_REQUEST",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        if hasattr(error, 'description') and error.description:
            error_response["details"] = error.description
        
        logger.warning(f"Bad request (400) for {request.path}: {error}")
        return jsonify(error_response), 400

    @app.errorhandler(401)
    def handle_unauthorized(error):
        """Handle 401 Unauthorized errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": "Authentication required. Please provide valid credentials.",
            "error_code": "UNAUTHORIZED",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        logger.warning(f"Unauthorized access (401) for {request.path}")
        return jsonify(error_response), 401

    @app.errorhandler(403)
    def handle_forbidden(error):
        """Handle 403 Forbidden errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error", 
            "message": "Access denied. You don't have permission to access this resource.",
            "error_code": "FORBIDDEN",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        logger.warning(f"Forbidden access (403) for {request.path}")
        return jsonify(error_response), 403

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 Not Found errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": "The requested resource was not found.",
            "error_code": "NOT_FOUND", 
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        logger.info(f"Resource not found (404): {request.path}")
        return jsonify(error_response), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 Method Not Allowed errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": f"Method {request.method} not allowed for this endpoint.",
            "error_code": "METHOD_NOT_ALLOWED",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        if hasattr(error, 'valid_methods'):
            error_response["allowed_methods"] = list(error.valid_methods)
        
        logger.warning(f"Method not allowed (405): {request.method} {request.path}")
        return jsonify(error_response), 405

    @app.errorhandler(413)
    def handle_payload_too_large(error):
        """Handle 413 Payload Too Large errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": "The file or request is too large. Please try with a smaller size.",
            "error_code": "PAYLOAD_TOO_LARGE",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "suggestions": [
                "Try uploading a smaller file",
                "Compress your file before uploading",
                "Split large requests into smaller chunks"
            ]
        }
        
        logger.warning(f"Payload too large (413) for {request.path}")
        return jsonify(error_response), 413

    @app.errorhandler(429)
    def handle_rate_limit_exceeded(error):
        """Handle 429 Too Many Requests errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": "Too many requests. Please wait a moment before trying again.",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "suggestions": [
                "Wait a few seconds before making another request",
                "Consider implementing request batching",
                "Contact support if you need higher rate limits"
            ]
        }
        
        logger.warning(f"Rate limit exceeded (429) for {request.path}")
        return jsonify(error_response), 429

    @app.errorhandler(500)
    def handle_internal_server_error(error):
        """Handle 500 Internal Server Error"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": "An internal server error occurred. Please try again later.",
            "error_code": "INTERNAL_SERVER_ERROR",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "suggestions": [
                "Try your request again in a few moments",
                "Contact support if the problem persists"
            ]
        }
        
        # Log the full traceback for debugging
        logger.error(f"Internal server error (500) for {request.path}: {error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify(error_response), 500

    @app.errorhandler(503)
    def handle_service_unavailable(error):
        """Handle 503 Service Unavailable errors"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response = {
            "error": True,
            "status": "error",
            "message": "Service temporarily unavailable. Please try again later.",
            "error_code": "SERVICE_UNAVAILABLE",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "suggestions": [
                "Try again in a few minutes",
                "Check system status page",
                "Contact support if service remains unavailable"
            ]
        }
        
        logger.error(f"Service unavailable (503) for {request.path}")
        return jsonify(error_response), 503

    @app.errorhandler(ValueError)
    def handle_value_error(error):
        """Handle ValueError exceptions"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response, status_code = ErrorRecovery.create_error_response(
            ErrorCategories.VALIDATION,
            "invalid_value",
            str(error),
            400,
            ["Check your input format", "Ensure all required fields are provided"]
        )
        error_response["request_id"] = request_id
        
        logger.warning(f"ValueError for {request.path}: {error}")
        return jsonify(error_response), status_code

    @app.errorhandler(KeyError)
    def handle_key_error(error):
        """Handle KeyError exceptions (missing required fields)"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        missing_field = str(error).strip("'\"")
        error_response = {
            "error": True,
            "status": "error",
            "message": f"Missing required field: {missing_field}",
            "error_code": "MISSING_FIELD",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "missing_field": missing_field,
            "suggestions": [
                f"Include the '{missing_field}' field in your request",
                "Check the API documentation for required fields"
            ]
        }
        
        logger.warning(f"KeyError for {request.path}: {error}")
        return jsonify(error_response), 400

    @app.errorhandler(ConnectionError)
    def handle_connection_error(error):
        """Handle connection errors to external services"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        error_response, status_code = ErrorRecovery.create_error_response(
            ErrorCategories.NETWORK,
            "connection_failed",
            str(error),
            503,
            ["Try again in a few moments", "Check your internet connection"]
        )
        error_response["request_id"] = request_id
        
        logger.error(f"Connection error for {request.path}: {error}")
        return jsonify(error_response), status_code

    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle any uncaught exceptions"""
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        
        # Log the full exception for debugging
        logger.error(f"Unhandled exception for {request.path}: {error}")
        logger.error(f"Exception type: {type(error).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Don't expose internal error details in production
        if current_app.config.get('FLASK_ENV') == 'production':
            message = "An unexpected error occurred. Please try again later."
            technical_details = None
        else:
            message = f"An unexpected error occurred: {str(error)}"
            technical_details = {
                "exception_type": type(error).__name__,
                "traceback": traceback.format_exc() if current_app.debug else None
            }
        
        error_response = {
            "error": True,
            "status": "error", 
            "message": message,
            "error_code": "UNEXPECTED_ERROR",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "suggestions": [
                "Try your request again",
                "Contact support if the problem persists",
                "Check that all required fields are included"
            ]
        }
        
        if technical_details and current_app.debug:
            error_response["technical_details"] = technical_details
        
        return jsonify(error_response), 500

    logger.info("Global error handlers registered successfully")