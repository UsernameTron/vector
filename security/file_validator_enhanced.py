"""
Enhanced file upload validation and security scanning
Comprehensive file validation with content analysis, format verification, and security scanning
"""

import os
import hashlib
import mimetypes
import tempfile
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from flask import request, jsonify
from functools import wraps
from datetime import datetime
import json
import re

from utils.error_handler import (
    ErrorCategories,
    ErrorRecovery,
    DependencyManager,
    FileCleanupManager,
    handle_errors,
    safe_operation
)

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass


class FileFormatAnalyzer:
    """Advanced file format analysis and validation"""
    
    # Extended file signatures for better detection
    FILE_SIGNATURES = {
        # Documents
        b'\x25\x50\x44\x46': 'application/pdf',
        b'\x50\x4B\x03\x04': 'application/zip',  # Also docx, xlsx
        b'\xD0\xCF\x11\xE0': 'application/msword',  # Old Office format
        b'\x7B\x5C\x72\x74': 'application/rtf',
        
        # Text files
        b'\xEF\xBB\xBF': 'text/plain',  # UTF-8 BOM
        b'\xFF\xFE': 'text/plain',      # UTF-16 LE BOM
        b'\xFE\xFF': 'text/plain',      # UTF-16 BE BOM
        
        # Images (for future expansion)
        b'\xFF\xD8\xFF': 'image/jpeg',
        b'\x89\x50\x4E\x47': 'image/png',
        
        # Executables (dangerous)
        b'\x4D\x5A': 'application/x-executable',
        b'\x7F\x45\x4C\x46': 'application/x-executable',
        b'\xCA\xFE\xBA\xBE': 'application/x-executable',
        
        # Scripts (potentially dangerous)
        b'#!/bin/sh': 'text/x-shellscript',
        b'#!/bin/bash': 'text/x-shellscript',
        b'<?php': 'application/x-php',
        b'<script': 'text/html',
    }
    
    @staticmethod
    def detect_file_type(file_path: str) -> Tuple[str, float]:
        """Detect file type with confidence score"""
        confidence = 0.0
        detected_type = 'application/octet-stream'
        
        try:
            # Check file signature
            with open(file_path, 'rb') as f:
                header = f.read(32)
                
                for signature, mime_type in FileFormatAnalyzer.FILE_SIGNATURES.items():
                    if header.startswith(signature):
                        detected_type = mime_type
                        confidence = 0.9
                        break
            
            # Fallback to python-magic if available
            if confidence < 0.5:
                try:
                    if DependencyManager.check_dependency('magic'):
                        import magic
                        detected_type = magic.from_file(file_path, mime=True)
                        confidence = 0.7
                except Exception as e:
                    logger.warning(f"python-magic detection failed: {e}")
            
            # Final fallback to mimetypes
            if confidence < 0.3:
                guessed_type, _ = mimetypes.guess_type(file_path)
                if guessed_type:
                    detected_type = guessed_type
                    confidence = 0.3
            
        except Exception as e:
            logger.error(f"File type detection failed: {e}")
        
        return detected_type, confidence
    
    @staticmethod
    def analyze_text_encoding(file_path: str) -> Dict[str, Any]:
        """Analyze text file encoding"""
        analysis = {
            'encoding': 'unknown',
            'confidence': 0.0,
            'bom_detected': False,
            'errors': []
        }
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(8192)  # Read first 8KB
            
            # Check for BOM
            if raw_data.startswith(b'\xEF\xBB\xBF'):
                analysis['encoding'] = 'utf-8'
                analysis['bom_detected'] = True
                analysis['confidence'] = 0.9
            elif raw_data.startswith(b'\xFF\xFE'):
                analysis['encoding'] = 'utf-16-le'
                analysis['bom_detected'] = True
                analysis['confidence'] = 0.9
            elif raw_data.startswith(b'\xFE\xFF'):
                analysis['encoding'] = 'utf-16-be'
                analysis['bom_detected'] = True
                analysis['confidence'] = 0.9
            else:
                # Try common encodings
                encodings_to_try = ['utf-8', 'ascii', 'latin-1', 'cp1252']
                for encoding in encodings_to_try:
                    try:
                        raw_data.decode(encoding)
                        analysis['encoding'] = encoding
                        analysis['confidence'] = 0.7 if encoding == 'utf-8' else 0.5
                        break
                    except UnicodeDecodeError:
                        continue
        
        except Exception as e:
            analysis['errors'].append(str(e))
        
        return analysis


class ContentSecurityScanner:
    """Advanced content security scanning"""
    
    # Malicious patterns for different file types
    MALICIOUS_PATTERNS = {
        'general': [
            rb'<script[^>]*>.*?</script>',
            rb'javascript:',
            rb'vbscript:',
            rb'data:text/html',
            rb'eval\s*\(',
            rb'exec\s*\(',
            rb'system\s*\(',
            rb'shell_exec\s*\(',
            rb'passthru\s*\(',
            rb'base64_decode\s*\(',
        ],
        'office': [
            rb'ActiveXObject',
            rb'WScript\.Shell',
            rb'Shell\.Application',
            rb'CreateObject',
            rb'GetObject',
            rb'<%.*?%>',  # ASP tags
            rb'<\?.*?\?>',  # PHP tags
        ],
        'pdf': [
            rb'/JavaScript',
            rb'/JS',
            rb'/OpenAction',
            rb'/Launch',
            rb'/EmbeddedFile',
            rb'/URI\s*\(',
        ],
        'text': [
            rb'rm\s+-rf\s+/',
            rb'format\s+c:',
            rb'del\s+/s\s+/q',
            rb'DROP\s+DATABASE',
            rb'DELETE\s+FROM.*WHERE.*1=1',
        ]
    }
    
    # Suspicious keywords that might indicate malicious content
    SUSPICIOUS_KEYWORDS = [
        'password', 'secret', 'private_key', 'api_key', 'token',
        'malware', 'virus', 'trojan', 'backdoor', 'exploit',
        'inject', 'payload', 'shellcode'
    ]
    
    @staticmethod
    def scan_content(file_path: str, file_type: str) -> Dict[str, Any]:
        """Comprehensive content security scan"""
        scan_result = {
            'safe': True,
            'threats': [],
            'warnings': [],
            'confidence': 1.0,
            'scan_details': {}
        }
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024 * 1024)  # Read first 1MB
            
            # General pattern scanning
            threats = ContentSecurityScanner._scan_patterns(
                content, ContentSecurityScanner.MALICIOUS_PATTERNS['general']
            )
            scan_result['threats'].extend(threats)
            
            # File-type specific scanning
            if 'office' in file_type or 'word' in file_type or 'excel' in file_type:
                office_threats = ContentSecurityScanner._scan_patterns(
                    content, ContentSecurityScanner.MALICIOUS_PATTERNS['office']
                )
                scan_result['threats'].extend(office_threats)
            
            elif 'pdf' in file_type:
                pdf_threats = ContentSecurityScanner._scan_patterns(
                    content, ContentSecurityScanner.MALICIOUS_PATTERNS['pdf']
                )
                scan_result['threats'].extend(pdf_threats)
            
            elif 'text' in file_type:
                text_threats = ContentSecurityScanner._scan_patterns(
                    content, ContentSecurityScanner.MALICIOUS_PATTERNS['text']
                )
                scan_result['threats'].extend(text_threats)
            
            # Keyword analysis
            warnings = ContentSecurityScanner._scan_keywords(content)
            scan_result['warnings'].extend(warnings)
            
            # Entropy analysis (detect encrypted/compressed content)
            entropy = ContentSecurityScanner._calculate_entropy(content)
            scan_result['scan_details']['entropy'] = entropy
            
            if entropy > 7.5:  # High entropy might indicate encryption/compression
                scan_result['warnings'].append("High entropy content detected (possibly encrypted/compressed)")
            
            # URL/Domain analysis
            urls = ContentSecurityScanner._extract_urls(content)
            if urls:
                scan_result['scan_details']['urls_found'] = len(urls)
                suspicious_urls = ContentSecurityScanner._analyze_urls(urls)
                if suspicious_urls:
                    scan_result['threats'].extend([f"Suspicious URL: {url}" for url in suspicious_urls])
            
            # Overall safety assessment
            scan_result['safe'] = len(scan_result['threats']) == 0
            if scan_result['threats']:
                scan_result['confidence'] = max(0.0, 1.0 - len(scan_result['threats']) * 0.2)
            
        except Exception as e:
            logger.error(f"Content scanning failed: {e}")
            scan_result['safe'] = False
            scan_result['threats'].append(f"Scan error: {str(e)}")
            scan_result['confidence'] = 0.0
        
        return scan_result
    
    @staticmethod
    def _scan_patterns(content: bytes, patterns: List[bytes]) -> List[str]:
        """Scan content for malicious patterns"""
        threats = []
        content_lower = content.lower()
        
        for pattern in patterns:
            try:
                if re.search(pattern, content_lower, re.IGNORECASE | re.DOTALL):
                    threats.append(f"Malicious pattern detected: {pattern.decode('utf-8', errors='ignore')}")
            except Exception:
                continue  # Skip invalid patterns
        
        return threats
    
    @staticmethod
    def _scan_keywords(content: bytes) -> List[str]:
        """Scan for suspicious keywords"""
        warnings = []
        try:
            content_str = content.decode('utf-8', errors='ignore').lower()
            
            for keyword in ContentSecurityScanner.SUSPICIOUS_KEYWORDS:
                if keyword in content_str:
                    warnings.append(f"Suspicious keyword found: {keyword}")
        
        except Exception:
            pass
        
        return warnings
    
    @staticmethod
    def _calculate_entropy(data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        frequencies = {}
        for byte in data:
            frequencies[byte] = frequencies.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in frequencies.values():
            if count > 0:
                probability = count / data_len
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    @staticmethod
    def _extract_urls(content: bytes) -> List[str]:
        """Extract URLs from content"""
        urls = []
        try:
            content_str = content.decode('utf-8', errors='ignore')
            url_pattern = r'https?://[^\s<>"\']*'
            urls = re.findall(url_pattern, content_str, re.IGNORECASE)
        except Exception:
            pass
        
        return urls
    
    @staticmethod
    def _analyze_urls(urls: List[str]) -> List[str]:
        """Analyze URLs for suspicious patterns"""
        suspicious = []
        suspicious_domains = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co',  # URL shorteners
            'localhost', '127.0.0.1', '0.0.0.0',        # Local addresses
        ]
        
        suspicious_patterns = [
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
            r'[a-z0-9]{20,}\.com',  # Very long random domains
            r'[^a-zA-Z0-9\-\.]',    # Special characters in domain
        ]
        
        for url in urls:
            try:
                # Check suspicious domains
                for domain in suspicious_domains:
                    if domain in url.lower():
                        suspicious.append(url)
                        break
                
                # Check suspicious patterns
                for pattern in suspicious_patterns:
                    if re.search(pattern, url, re.IGNORECASE):
                        suspicious.append(url)
                        break
                        
            except Exception:
                continue
        
        return list(set(suspicious))  # Remove duplicates


class EnhancedFileValidator:
    """Enhanced file validator with comprehensive security scanning"""
    
    # Extended allowed file types with security considerations
    ALLOWED_FILE_TYPES = {
        # Text files
        'txt': {
            'mime_types': ['text/plain'],
            'max_size': 10 * 1024 * 1024,  # 10MB
            'security_level': 'low'
        },
        'md': {
            'mime_types': ['text/markdown', 'text/x-markdown', 'text/plain'],
            'max_size': 5 * 1024 * 1024,   # 5MB
            'security_level': 'low'
        },
        'csv': {
            'mime_types': ['text/csv', 'application/csv'],
            'max_size': 50 * 1024 * 1024,  # 50MB
            'security_level': 'medium'
        },
        'json': {
            'mime_types': ['application/json', 'text/json'],
            'max_size': 10 * 1024 * 1024,  # 10MB
            'security_level': 'medium'
        },
        'xml': {
            'mime_types': ['application/xml', 'text/xml'],
            'max_size': 10 * 1024 * 1024,  # 10MB
            'security_level': 'high'  # XXE attacks possible
        },
        
        # Document files
        'pdf': {
            'mime_types': ['application/pdf'],
            'max_size': 100 * 1024 * 1024,  # 100MB
            'security_level': 'high'  # Can contain JavaScript
        },
        'doc': {
            'mime_types': ['application/msword'],
            'max_size': 50 * 1024 * 1024,   # 50MB
            'security_level': 'high'  # Can contain macros
        },
        'docx': {
            'mime_types': [
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ],
            'max_size': 50 * 1024 * 1024,   # 50MB
            'security_level': 'high'  # Can contain macros
        },
        'rtf': {
            'mime_types': ['application/rtf', 'text/rtf'],
            'max_size': 10 * 1024 * 1024,   # 10MB
            'security_level': 'medium'
        },
        'odt': {
            'mime_types': ['application/vnd.oasis.opendocument.text'],
            'max_size': 50 * 1024 * 1024,   # 50MB
            'security_level': 'medium'
        }
    }
    
    @staticmethod
    @safe_operation("comprehensive_file_validation")
    def comprehensive_validation(
        file: FileStorage, 
        temp_dir: str = None,
        max_size: int = None,
        allowed_extensions: List[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform comprehensive file validation with security scanning"""
        
        validation_result = {
            'valid': False,
            'file_info': {},
            'security_scan': {},
            'errors': [],
            'warnings': [],
            'validation_details': {}
        }
        
        temp_file_path = None
        
        try:
            # Basic validation
            if not file or not file.filename:
                validation_result['errors'].append("No file provided")
                return False, validation_result
            
            # Secure filename
            secure_name = secure_filename(file.filename)
            if not secure_name:
                validation_result['errors'].append("Invalid filename")
                return False, validation_result
            
            validation_result['file_info']['original_filename'] = file.filename
            validation_result['file_info']['secure_filename'] = secure_name
            
            # Extract and validate extension
            if '.' not in secure_name:
                validation_result['errors'].append("File must have an extension")
                return False, validation_result
            
            extension = secure_name.rsplit('.', 1)[1].lower()
            validation_result['file_info']['extension'] = extension
            
            # Check allowed extensions
            allowed_exts = allowed_extensions or list(EnhancedFileValidator.ALLOWED_FILE_TYPES.keys())
            if extension not in allowed_exts:
                validation_result['errors'].append(
                    f"File type '{extension}' not allowed. Allowed: {', '.join(allowed_exts)}"
                )
                return False, validation_result
            
            file_config = EnhancedFileValidator.ALLOWED_FILE_TYPES[extension]
            validation_result['validation_details']['security_level'] = file_config['security_level']
            
            # Validate file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            validation_result['file_info']['size_bytes'] = file_size
            
            max_allowed_size = max_size or file_config['max_size']
            if file_size > max_allowed_size:
                validation_result['errors'].append(
                    f"File too large: {file_size} bytes (max: {max_allowed_size} bytes)"
                )
                return False, validation_result
            
            if file_size == 0:
                validation_result['errors'].append("File is empty")
                return False, validation_result
            
            # Create temporary file for analysis
            if not temp_dir:
                temp_dir = FileCleanupManager.create_temp_dir()
            
            temp_file_path = os.path.join(temp_dir, secure_name)
            FileCleanupManager.register_temp_file(temp_file_path)
            
            # Save file for analysis
            file.save(temp_file_path)
            
            # File format analysis
            detected_type, confidence = FileFormatAnalyzer.detect_file_type(temp_file_path)
            validation_result['file_info']['detected_mime_type'] = detected_type
            validation_result['file_info']['detection_confidence'] = confidence
            
            # Verify MIME type matches extension
            allowed_mimes = file_config['mime_types']
            if detected_type not in allowed_mimes and confidence > 0.5:
                validation_result['errors'].append(
                    f"File content ({detected_type}) doesn't match extension ({extension})"
                )
                return False, validation_result
            
            # Text encoding analysis for text files
            if 'text' in detected_type:
                encoding_analysis = FileFormatAnalyzer.analyze_text_encoding(temp_file_path)
                validation_result['file_info']['encoding_analysis'] = encoding_analysis
                
                if encoding_analysis['confidence'] < 0.3:
                    validation_result['warnings'].append("Could not reliably detect text encoding")
            
            # Security content scanning
            security_scan = ContentSecurityScanner.scan_content(temp_file_path, detected_type)
            validation_result['security_scan'] = security_scan
            
            if not security_scan['safe']:
                validation_result['errors'].extend(security_scan['threats'])
                return False, validation_result
            
            if security_scan['warnings']:
                validation_result['warnings'].extend(security_scan['warnings'])
            
            # Calculate file hash
            file_hash = EnhancedFileValidator._calculate_file_hash(temp_file_path)
            validation_result['file_info']['sha256_hash'] = file_hash
            
            # Additional checks based on security level
            if file_config['security_level'] == 'high':
                additional_checks = EnhancedFileValidator._perform_high_security_checks(
                    temp_file_path, detected_type
                )
                validation_result['validation_details']['additional_checks'] = additional_checks
                
                if not additional_checks['passed']:
                    validation_result['errors'].extend(additional_checks['errors'])
                    return False, validation_result
            
            # Final validation
            validation_result['valid'] = len(validation_result['errors']) == 0
            validation_result['validation_details']['timestamp'] = datetime.now().isoformat()
            
            return validation_result['valid'], validation_result
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            return False, validation_result
        
        finally:
            # Cleanup temporary file if validation failed
            if temp_file_path and not validation_result.get('valid', False):
                FileCleanupManager.cleanup_file(temp_file_path)
    
    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return ""
    
    @staticmethod
    def _perform_high_security_checks(file_path: str, file_type: str) -> Dict[str, Any]:
        """Perform additional security checks for high-risk files"""
        checks = {
            'passed': True,
            'errors': [],
            'checks_performed': []
        }
        
        try:
            # PDF-specific checks
            if 'pdf' in file_type:
                checks['checks_performed'].append('pdf_security_check')
                pdf_check = EnhancedFileValidator._check_pdf_security(file_path)
                if not pdf_check['safe']:
                    checks['passed'] = False
                    checks['errors'].extend(pdf_check['threats'])
            
            # Office document checks
            elif any(office_type in file_type for office_type in ['word', 'excel', 'powerpoint', 'office']):
                checks['checks_performed'].append('office_security_check')
                office_check = EnhancedFileValidator._check_office_security(file_path)
                if not office_check['safe']:
                    checks['passed'] = False
                    checks['errors'].extend(office_check['threats'])
            
            # XML-specific checks (XXE prevention)
            elif 'xml' in file_type:
                checks['checks_performed'].append('xml_security_check')
                xml_check = EnhancedFileValidator._check_xml_security(file_path)
                if not xml_check['safe']:
                    checks['passed'] = False
                    checks['errors'].extend(xml_check['threats'])
        
        except Exception as e:
            logger.error(f"High security checks failed: {e}")
            checks['passed'] = False
            checks['errors'].append(f"Security check error: {str(e)}")
        
        return checks
    
    @staticmethod
    def _check_pdf_security(file_path: str) -> Dict[str, Any]:
        """Check PDF for security issues"""
        result = {'safe': True, 'threats': []}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Check for dangerous PDF features
            dangerous_features = [
                b'/JavaScript', b'/JS', b'/OpenAction', b'/Launch',
                b'/EmbeddedFile', b'/URI', b'/GoToR', b'/ImportData'
            ]
            
            for feature in dangerous_features:
                if feature in content:
                    result['safe'] = False
                    result['threats'].append(f"Dangerous PDF feature detected: {feature.decode('utf-8', errors='ignore')}")
        
        except Exception as e:
            result['safe'] = False
            result['threats'].append(f"PDF security check failed: {str(e)}")
        
        return result
    
    @staticmethod
    def _check_office_security(file_path: str) -> Dict[str, Any]:
        """Check Office documents for security issues"""
        result = {'safe': True, 'threats': []}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Check for macro indicators
            macro_indicators = [
                b'vbaProject', b'macros/', b'VBA', b'Microsoft Office Word',
                b'ActiveXObject', b'WScript', b'Shell.Application'
            ]
            
            for indicator in macro_indicators:
                if indicator in content:
                    result['safe'] = False
                    result['threats'].append("Document may contain macros or active content")
                    break
        
        except Exception as e:
            result['safe'] = False
            result['threats'].append(f"Office security check failed: {str(e)}")
        
        return result
    
    @staticmethod
    def _check_xml_security(file_path: str) -> Dict[str, Any]:
        """Check XML for security issues (XXE, etc.)"""
        result = {'safe': True, 'threats': []}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read(8192)  # Read first 8KB
            
            # Check for XXE indicators
            xxe_patterns = [
                b'<!ENTITY', b'SYSTEM', b'file://', b'http://', b'https://',
                b'<!DOCTYPE', b'ENTITY'
            ]
            
            content_upper = content.upper()
            for pattern in xxe_patterns:
                if pattern.upper() in content_upper:
                    result['safe'] = False
                    result['threats'].append("XML file contains potentially dangerous external entity references")
                    break
        
        except Exception as e:
            result['safe'] = False
            result['threats'].append(f"XML security check failed: {str(e)}")
        
        return result


@handle_errors(ErrorCategories.FILE_PROCESSING)
def enhanced_validate_file_upload(
    max_size: int = None,
    allowed_extensions: List[str] = None,
    security_level: str = 'medium'
):
    """Enhanced decorator for file upload validation"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'file' not in request.files:
                error_response, status = ErrorRecovery.create_error_response(
                    ErrorCategories.VALIDATION,
                    'no_file',
                    'No file provided',
                    400,
                    ['Please select a file to upload']
                )
                return jsonify(error_response), status
            
            file = request.files['file']
            if not file.filename:
                error_response, status = ErrorRecovery.create_error_response(
                    ErrorCategories.VALIDATION,
                    'empty_filename',
                    'No file selected',
                    400,
                    ['Please select a valid file']
                )
                return jsonify(error_response), status
            
            # Create temporary directory for validation
            temp_dir = FileCleanupManager.create_temp_dir()
            
            try:
                # Perform comprehensive validation
                is_valid, validation_result = EnhancedFileValidator.comprehensive_validation(
                    file, temp_dir, max_size, allowed_extensions
                )
                
                if not is_valid:
                    # Create user-friendly error response
                    primary_error = validation_result['errors'][0] if validation_result['errors'] else "File validation failed"
                    
                    recovery_suggestions = [
                        "Check that your file is not corrupted",
                        "Ensure the file is in a supported format",
                        "Try reducing the file size if it's too large",
                        "Scan the file for malware before uploading"
                    ]
                    
                    error_response, status = ErrorRecovery.create_error_response(
                        ErrorCategories.FILE_PROCESSING,
                        'validation_failed',
                        primary_error,
                        400,
                        recovery_suggestions
                    )
                    
                    # Add validation details for debugging
                    error_response['validation_details'] = {
                        'errors': validation_result['errors'],
                        'warnings': validation_result['warnings'],
                        'file_info': validation_result.get('file_info', {})
                    }
                    
                    return jsonify(error_response), status
                
                # Add validation result to request context
                request.file_validation = validation_result
                request.temp_file_path = os.path.join(temp_dir, validation_result['file_info']['secure_filename'])
                
                return f(*args, **kwargs)
            
            except Exception as e:
                logger.error(f"File validation decorator error: {e}")
                error_response, status = ErrorRecovery.create_error_response(
                    ErrorCategories.FILE_PROCESSING,
                    'processing_error',
                    str(e),
                    500,
                    ['Try uploading the file again', 'Contact support if the problem persists']
                )
                return jsonify(error_response), status
            
            finally:
                # Cleanup temporary directory
                FileCleanupManager.cleanup_directory(temp_dir)
        
        return decorated_function
    return decorator