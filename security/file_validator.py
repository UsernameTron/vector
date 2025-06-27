"""
File upload validation and security scanning
Handles file type checking, size limits, and basic virus scanning
"""

import os
import magic
import hashlib
import mimetypes
from typing import Dict, List, Tuple, Optional, Any
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from flask import request, jsonify
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """Comprehensive file upload validation and security scanning"""
    
    # Allowed file types and their MIME types
    ALLOWED_EXTENSIONS = {
        'txt': ['text/plain'],
        'pdf': ['application/pdf'],
        'doc': ['application/msword'],
        'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        'rtf': ['application/rtf', 'text/rtf'],
        'odt': ['application/vnd.oasis.opendocument.text'],
        'md': ['text/markdown', 'text/x-markdown'],
        'json': ['application/json'],
        'csv': ['text/csv'],
        'xml': ['application/xml', 'text/xml']
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB default
    MAX_FILENAME_LENGTH = 255
    
    # Dangerous file signatures (magic bytes)
    DANGEROUS_SIGNATURES = [
        b'\x4D\x5A',  # Windows executable (MZ)
        b'\x50\x4B\x03\x04',  # ZIP archive (could contain executables)
        b'\x7F\x45\x4C\x46',  # Linux executable (ELF)
        b'\xCA\xFE\xBA\xBE',  # Java class file
        b'\xFE\xED\xFA\xCE',  # Mach-O binary (macOS)
        b'\xFE\xED\xFA\xCF',  # Mach-O binary (macOS 64-bit)
    ]
    
    # Suspicious content patterns
    SUSPICIOUS_PATTERNS = [
        b'<script',
        b'javascript:',
        b'vbscript:',
        b'data:text/html',
        b'<?php',
        b'<%',
        b'eval(',
        b'exec(',
        b'system(',
        b'shell_exec(',
    ]
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Get file type using python-magic"""
        try:
            file_type = magic.from_file(file_path, mime=True)
            return file_type
        except Exception as e:
            logger.warning(f"Could not determine file type for {file_path}: {e}")
            # Fallback to mimetypes
            file_type, _ = mimetypes.guess_type(file_path)
            return file_type or 'application/octet-stream'
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """Validate filename for security"""
        if not filename:
            return False, "Filename cannot be empty"
        
        if len(filename) > FileValidator.MAX_FILENAME_LENGTH:
            return False, f"Filename too long (max {FileValidator.MAX_FILENAME_LENGTH} characters)"
        
        # Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            if char in filename:
                return False, f"Filename contains dangerous character: {char}"
        
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        
        base_name = filename.split('.')[0].upper()
        if base_name in reserved_names:
            return False, f"Filename uses reserved name: {base_name}"
        
        return True, "Valid filename"
    
    @staticmethod
    def validate_file_extension(filename: str) -> Tuple[bool, str, Optional[str]]:
        """Validate file extension against allowed types"""
        if '.' not in filename:
            return False, "File must have an extension", None
        
        extension = filename.rsplit('.', 1)[1].lower()
        
        if extension not in FileValidator.ALLOWED_EXTENSIONS:
            allowed = ', '.join(FileValidator.ALLOWED_EXTENSIONS.keys())
            return False, f"File type not allowed. Allowed types: {allowed}", extension
        
        return True, "Valid extension", extension
    
    @staticmethod
    def validate_file_size(file: FileStorage, max_size: int = None) -> Tuple[bool, str]:
        """Validate file size"""
        if max_size is None:
            max_size = FileValidator.MAX_FILE_SIZE
        
        # Get file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        if file_size > max_size:
            size_mb = max_size / (1024 * 1024)
            return False, f"File too large (max {size_mb:.1f}MB)"
        
        if file_size == 0:
            return False, "File is empty"
        
        return True, "Valid file size"
    
    @staticmethod
    def validate_mime_type(file_path: str, expected_extension: str) -> Tuple[bool, str]:
        """Validate MIME type matches file extension"""
        try:
            detected_mime = FileValidator.get_file_type(file_path)
            allowed_mimes = FileValidator.ALLOWED_EXTENSIONS.get(expected_extension, [])
            
            if detected_mime not in allowed_mimes:
                return False, f"File content doesn't match extension. Expected: {allowed_mimes}, Got: {detected_mime}"
            
            return True, "Valid MIME type"
        
        except Exception as e:
            logger.error(f"MIME type validation error: {e}")
            return False, "Could not validate file type"
    
    @staticmethod
    def scan_file_content(file_path: str) -> Tuple[bool, List[str]]:
        """Basic content scanning for malicious patterns"""
        threats = []
        
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes for signature checking
                header = f.read(1024)
                
                # Check for dangerous file signatures
                for signature in FileValidator.DANGEROUS_SIGNATURES:
                    if header.startswith(signature):
                        threats.append(f"Dangerous file signature detected: {signature.hex()}")
                
                # Reset and read more content for pattern matching
                f.seek(0)
                content = f.read(10240)  # Read first 10KB
                
                # Check for suspicious patterns
                for pattern in FileValidator.SUSPICIOUS_PATTERNS:
                    if pattern in content.lower():
                        threats.append(f"Suspicious pattern detected: {pattern.decode('utf-8', errors='ignore')}")
        
        except Exception as e:
            logger.error(f"File content scanning error: {e}")
            threats.append("Could not scan file content")
        
        is_safe = len(threats) == 0
        return is_safe, threats
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation error: {e}")
            return ""
    
    @staticmethod
    def comprehensive_validation(file: FileStorage, save_path: str, max_size: int = None) -> Tuple[bool, Dict[str, Any]]:
        """Perform comprehensive file validation"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Validate filename
            is_valid, message = FileValidator.validate_filename(file.filename)
            if not is_valid:
                result['valid'] = False
                result['errors'].append(message)
                return result['valid'], result
            
            # Secure filename
            secure_name = secure_filename(file.filename)
            result['file_info']['secure_filename'] = secure_name
            
            # Validate file extension
            is_valid, message, extension = FileValidator.validate_file_extension(secure_name)
            if not is_valid:
                result['valid'] = False
                result['errors'].append(message)
                return result['valid'], result
            
            result['file_info']['extension'] = extension
            
            # Validate file size
            is_valid, message = FileValidator.validate_file_size(file, max_size)
            if not is_valid:
                result['valid'] = False
                result['errors'].append(message)
                return result['valid'], result
            
            # Save file temporarily for content validation
            temp_path = os.path.join(save_path, secure_name)
            file.save(temp_path)
            
            try:
                # Validate MIME type
                is_valid, message = FileValidator.validate_mime_type(temp_path, extension)
                if not is_valid:
                    result['valid'] = False
                    result['errors'].append(message)
                
                # Scan file content
                is_safe, threats = FileValidator.scan_file_content(temp_path)
                if not is_safe:
                    result['valid'] = False
                    result['errors'].extend(threats)
                
                # Calculate file hash
                file_hash = FileValidator.calculate_file_hash(temp_path)
                result['file_info']['sha256'] = file_hash
                
                # Get file size
                file_size = os.path.getsize(temp_path)
                result['file_info']['size_bytes'] = file_size
                
                # Get MIME type
                mime_type = FileValidator.get_file_type(temp_path)
                result['file_info']['mime_type'] = mime_type
            
            finally:
                # Clean up temporary file if validation failed
                if not result['valid'] and os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"File validation error: {str(e)}")
            logger.error(f"Comprehensive file validation error: {e}")
        
        return result['valid'], result


def validate_file_upload(max_size: int = None, allowed_extensions: List[str] = None):
    """Decorator for file upload validation"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Create uploads directory if it doesn't exist
            upload_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Perform comprehensive validation
            is_valid, result = FileValidator.comprehensive_validation(file, upload_dir, max_size)
            
            if not is_valid:
                return jsonify({
                    'error': 'File validation failed',
                    'details': result['errors']
                }), 400
            
            # Add validation result to request context
            request.file_validation = result
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator