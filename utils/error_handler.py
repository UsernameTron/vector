"""
Comprehensive error handling and recovery mechanisms
Provides robust error handling, user-friendly messages, and recovery strategies
"""

import logging
import traceback
import sys
import os
from typing import Dict, Any, Optional, Tuple, List
from functools import wraps
from flask import jsonify, request
from datetime import datetime
import tempfile
import shutil

logger = logging.getLogger(__name__)


class ErrorCategories:
    """Error category constants for consistent error handling"""
    VALIDATION = "validation_error"
    AUTHENTICATION = "authentication_error"
    AUTHORIZATION = "authorization_error"
    DATABASE = "database_error"
    FILE_PROCESSING = "file_processing_error"
    NETWORK = "network_error"
    DEPENDENCY = "dependency_error"
    SYSTEM = "system_error"
    USER_INPUT = "user_input_error"


class ErrorRecovery:
    """Error recovery strategies and mechanisms"""
    
    @staticmethod
    def get_user_friendly_message(error_category: str, error_code: str = None) -> str:
        """Get user-friendly error messages"""
        messages = {
            ErrorCategories.VALIDATION: {
                "default": "The information you provided is not valid. Please check your input and try again.",
                "file_too_large": "The file you're trying to upload is too large. Please use a smaller file.",
                "invalid_format": "The file format is not supported. Please use a supported file type.",
                "malicious_content": "The file contains potentially harmful content and cannot be processed."
            },
            ErrorCategories.AUTHENTICATION: {
                "default": "Authentication failed. Please check your credentials and try again.",
                "token_expired": "Your session has expired. Please log in again.",
                "invalid_token": "Your session is invalid. Please log in again."
            },
            ErrorCategories.AUTHORIZATION: {
                "default": "You don't have permission to perform this action.",
                "insufficient_permissions": "You need additional permissions to access this resource."
            },
            ErrorCategories.DATABASE: {
                "default": "There was a problem accessing the database. Please try again later.",
                "connection_failed": "Unable to connect to the database. Please try again in a few moments.",
                "search_failed": "Search is temporarily unavailable. Please try again later.",
                "storage_full": "Storage is full. Please contact support or try again later."
            },
            ErrorCategories.FILE_PROCESSING: {
                "default": "There was a problem processing your file. Please try again.",
                "corrupted_file": "The file appears to be corrupted. Please try uploading a different file.",
                "unsupported_encoding": "The file encoding is not supported. Please save the file as UTF-8 and try again."
            },
            ErrorCategories.NETWORK: {
                "default": "Network error occurred. Please check your connection and try again.",
                "timeout": "The request timed out. Please try again.",
                "service_unavailable": "The service is temporarily unavailable. Please try again later."
            },
            ErrorCategories.DEPENDENCY: {
                "default": "A required service is unavailable. Please try again later.",
                "ai_service_down": "AI services are temporarily unavailable. Please try again later.",
                "missing_dependency": "A required component is missing. Please contact support."
            },
            ErrorCategories.SYSTEM: {
                "default": "A system error occurred. Please try again later.",
                "disk_full": "System storage is full. Please try again later or contact support.",
                "memory_error": "System resources are temporarily unavailable. Please try again later."
            },
            ErrorCategories.USER_INPUT: {
                "default": "Please check your input and try again.",
                "empty_input": "Please provide the required information.",
                "invalid_characters": "Your input contains invalid characters. Please use only allowed characters."
            }
        }
        
        category_messages = messages.get(error_category, {"default": "An unexpected error occurred. Please try again."})
        return category_messages.get(error_code, category_messages["default"])
    
    @staticmethod
    def create_error_response(
        error_category: str,
        error_code: str = None,
        technical_message: str = None,
        status_code: int = 500,
        recovery_suggestions: List[str] = None
    ) -> Tuple[Dict[str, Any], int]:
        """Create standardized error response"""
        user_message = ErrorRecovery.get_user_friendly_message(error_category, error_code)
        
        error_response = {
            "error": True,
            "category": error_category,
            "message": user_message,
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request, 'request_id', 'unknown') if request else 'unknown'
        }
        
        if error_code:
            error_response["error_code"] = error_code
        
        if recovery_suggestions:
            error_response["suggestions"] = recovery_suggestions
        
        # Add technical details for debugging (not exposed to users in production)
        if technical_message and os.getenv('FLASK_ENV') == 'development':
            error_response["technical_details"] = technical_message
        
        return error_response, status_code
    
    @staticmethod
    def log_error(
        error: Exception,
        context: str,
        user_id: str = None,
        additional_info: Dict[str, Any] = None
    ):
        """Comprehensive error logging"""
        error_info = {
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "request_path": request.path if request else None,
            "request_method": request.method if request else None,
            "traceback": traceback.format_exc()
        }
        
        if additional_info:
            error_info.update(additional_info)
        
        logger.error(f"Error in {context}: {error_info}")


class DependencyManager:
    """Manages optional dependencies and provides fallbacks"""
    
    _dependency_status = {}
    
    @classmethod
    def check_dependency(cls, module_name: str, fallback_available: bool = False) -> bool:
        """Check if a dependency is available"""
        if module_name in cls._dependency_status:
            return cls._dependency_status[module_name]
        
        try:
            __import__(module_name)
            cls._dependency_status[module_name] = True
            logger.info(f"Dependency {module_name} is available")
            return True
        except ImportError as e:
            cls._dependency_status[module_name] = False
            if fallback_available:
                logger.warning(f"Dependency {module_name} not available, using fallback: {e}")
            else:
                logger.error(f"Required dependency {module_name} not available: {e}")
            return False
    
    @classmethod
    def safe_import(cls, module_name: str, fallback_module: str = None):
        """Safely import a module with optional fallback"""
        try:
            return __import__(module_name, fromlist=[''])
        except ImportError as e:
            if fallback_module:
                try:
                    logger.warning(f"Failed to import {module_name}, trying fallback {fallback_module}")
                    return __import__(fallback_module, fromlist=[''])
                except ImportError:
                    logger.error(f"Both {module_name} and fallback {fallback_module} failed to import")
                    raise
            else:
                logger.error(f"Failed to import required module {module_name}: {e}")
                raise
    
    @classmethod
    def get_dependency_status(cls) -> Dict[str, bool]:
        """Get status of all checked dependencies"""
        return cls._dependency_status.copy()


class FileCleanupManager:
    """Manages temporary file cleanup and resource management"""
    
    _temp_files = set()
    _temp_dirs = set()
    
    @classmethod
    def register_temp_file(cls, file_path: str):
        """Register a temporary file for cleanup"""
        cls._temp_files.add(file_path)
        logger.debug(f"Registered temp file: {file_path}")
    
    @classmethod
    def register_temp_dir(cls, dir_path: str):
        """Register a temporary directory for cleanup"""
        cls._temp_dirs.add(dir_path)
        logger.debug(f"Registered temp directory: {dir_path}")
    
    @classmethod
    def cleanup_file(cls, file_path: str) -> bool:
        """Clean up a specific file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
            cls._temp_files.discard(file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
            return False
    
    @classmethod
    def cleanup_directory(cls, dir_path: str) -> bool:
        """Clean up a specific directory"""
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.debug(f"Cleaned up directory: {dir_path}")
            cls._temp_dirs.discard(dir_path)
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup directory {dir_path}: {e}")
            return False
    
    @classmethod
    def cleanup_all(cls):
        """Clean up all registered temporary files and directories"""
        cleaned_files = 0
        cleaned_dirs = 0
        
        # Clean up files
        for file_path in cls._temp_files.copy():
            if cls.cleanup_file(file_path):
                cleaned_files += 1
        
        # Clean up directories
        for dir_path in cls._temp_dirs.copy():
            if cls.cleanup_directory(dir_path):
                cleaned_dirs += 1
        
        logger.info(f"Cleanup completed: {cleaned_files} files, {cleaned_dirs} directories")
    
    @classmethod
    def create_temp_file(cls, suffix: str = None, prefix: str = "temp_", dir: str = None) -> str:
        """Create a temporary file and register it for cleanup"""
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
        os.close(fd)  # Close the file descriptor
        cls.register_temp_file(temp_path)
        return temp_path
    
    @classmethod
    def create_temp_dir(cls, suffix: str = None, prefix: str = "temp_", dir: str = None) -> str:
        """Create a temporary directory and register it for cleanup"""
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        cls.register_temp_dir(temp_dir)
        return temp_dir


def handle_errors(error_category: str = ErrorCategories.SYSTEM):
    """Decorator for comprehensive error handling"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                # Determine error category and code
                error_code = None
                status_code = 500
                recovery_suggestions = []
                
                if isinstance(e, ValueError):
                    error_category_actual = ErrorCategories.VALIDATION
                    status_code = 400
                    recovery_suggestions = ["Check your input format", "Ensure all required fields are provided"]
                elif isinstance(e, PermissionError):
                    error_category_actual = ErrorCategories.AUTHORIZATION
                    status_code = 403
                    recovery_suggestions = ["Contact an administrator for access"]
                elif isinstance(e, FileNotFoundError):
                    error_category_actual = ErrorCategories.FILE_PROCESSING
                    status_code = 404
                    recovery_suggestions = ["Check if the file exists", "Try uploading the file again"]
                elif isinstance(e, MemoryError):
                    error_category_actual = ErrorCategories.SYSTEM
                    error_code = "memory_error"
                    recovery_suggestions = ["Try with a smaller file", "Try again later"]
                else:
                    error_category_actual = error_category
                
                # Log the error
                context = f"{f.__module__}.{f.__name__}"
                user_id = getattr(request, 'current_user', {}).get('user_id') if request else None
                ErrorRecovery.log_error(e, context, user_id)
                
                # Create user-friendly response
                error_response, status = ErrorRecovery.create_error_response(
                    error_category_actual,
                    error_code,
                    str(e),
                    status_code,
                    recovery_suggestions
                )
                
                return jsonify(error_response), status
        
        return decorated_function
    return decorator


def safe_operation(operation_name: str, cleanup_on_error: bool = True):
    """Decorator for safe operations with automatic cleanup"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            operation_id = f"op_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            logger.info(f"Starting operation {operation_name} (ID: {operation_id})")
            
            try:
                result = f(*args, **kwargs)
                logger.info(f"Operation {operation_name} completed successfully (ID: {operation_id})")
                return result
            except Exception as e:
                logger.error(f"Operation {operation_name} failed (ID: {operation_id}): {e}")
                
                if cleanup_on_error:
                    logger.info(f"Performing cleanup for failed operation {operation_id}")
                    FileCleanupManager.cleanup_all()
                
                raise
        
        return decorated_function
    return decorator