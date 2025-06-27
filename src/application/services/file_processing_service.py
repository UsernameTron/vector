"""
File processing service for handling file uploads and validation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

from src.domain.entities import FileUpload, Document, DocumentStatus
from src.domain.interfaces import (
    IFileRepository, IFileValidationService, IDocumentRepository,
    IEventPublisher, ILoggingService
)
from src.presentation.responses import (
    ValidationException, BusinessRuleException, ExternalServiceException
)
from src.infrastructure.container import scoped

logger = logging.getLogger(__name__)


@scoped()
class FileProcessingService:
    """Service for file processing operations"""
    
    def __init__(
        self,
        file_repository: IFileRepository,
        file_validation_service: IFileValidationService,
        document_repository: IDocumentRepository,
        event_publisher: IEventPublisher,
        logging_service: ILoggingService
    ):
        self.file_repository = file_repository
        self.file_validation_service = file_validation_service
        self.document_repository = document_repository
        self.event_publisher = event_publisher
        self.logging_service = logging_service
    
    async def process_file_upload(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        uploaded_by: Optional[str] = None
    ) -> tuple[FileUpload, Document]:
        """Process uploaded file and create document"""
        
        # Validate file data
        if not file_data:
            raise ValidationException("file", "File data is required")
        
        if not filename or not filename.strip():
            raise ValidationException("filename", "Filename is required")
        
        try:
            # Calculate file hash
            file_hash = hashlib.sha256(file_data).hexdigest()
            
            # Create file upload entity
            file_upload = FileUpload(
                filename=self._sanitize_filename(filename),
                original_filename=filename,
                content_type=content_type,
                size_bytes=len(file_data),
                hash_sha256=file_hash,
                uploaded_by=uploaded_by
            )
            
            # Validate file
            validation_result = await self.file_validation_service.validate_file(
                file_data, file_upload.filename
            )
            file_upload.validation_result = validation_result
            
            if not validation_result.get('valid', False):
                errors = validation_result.get('errors', ['File validation failed'])
                raise BusinessRuleException(
                    "file_validation",
                    f"File validation failed: {'; '.join(errors)}"
                )
            
            # Scan for security threats
            threat_scan = await self.file_validation_service.scan_for_threats(file_data)
            if not threat_scan.get('safe', True):
                threats = threat_scan.get('threats', ['Security threats detected'])
                raise BusinessRuleException(
                    "security_scan",
                    f"Security scan failed: {'; '.join(threats)}"
                )
            
            # Save file
            saved_file = await self.file_repository.save_file(file_upload, file_data)
            
            # Extract text content from file
            text_content = await self._extract_text_content(file_data, content_type)
            
            if not text_content:
                raise BusinessRuleException(
                    "content_extraction",
                    "Could not extract text content from file"
                )
            
            # Create document from file content
            document = Document(
                title=self._extract_title_from_filename(saved_file.filename),
                content=text_content,
                source="file_upload",
                created_by=uploaded_by,
                metadata={
                    "file_id": saved_file.id,
                    "original_filename": saved_file.original_filename,
                    "file_size": saved_file.size_bytes,
                    "file_hash": saved_file.hash_sha256,
                    "content_type": content_type
                }
            )
            
            # Save document
            created_document = await self.document_repository.create(document)
            
            # Update file upload status
            saved_file.mark_as_processed(created_document.id)
            
            # Log success
            await self.logging_service.log_info(
                f"File processed successfully: {saved_file.filename}",
                {
                    "file_id": saved_file.id,
                    "document_id": created_document.id,
                    "uploaded_by": uploaded_by,
                    "file_size": saved_file.size_bytes
                }
            )
            
            # Publish events
            await self.event_publisher.publish_event(
                "file.uploaded",
                {
                    "file_id": saved_file.id,
                    "filename": saved_file.filename,
                    "uploaded_by": uploaded_by,
                    "file_size": saved_file.size_bytes,
                    "timestamp": saved_file.uploaded_at.isoformat()
                }
            )
            
            await self.event_publisher.publish_event(
                "file.processed",
                {
                    "file_id": saved_file.id,
                    "document_id": created_document.id,
                    "filename": saved_file.filename,
                    "uploaded_by": uploaded_by,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return saved_file, created_document
            
        except (ValidationException, BusinessRuleException):
            raise
        except Exception as e:
            await self.logging_service.log_error(
                f"File processing failed: {filename}",
                e,
                {"filename": filename, "uploaded_by": uploaded_by, "file_size": len(file_data)}
            )
            raise ExternalServiceException("FileProcessing", f"File processing failed: {e}")
    
    async def get_file_metadata(self, file_id: str) -> FileUpload:
        """Get file metadata by ID"""
        if not file_id or not file_id.strip():
            raise ValidationException("file_id", "File ID is required")
        
        file_metadata = await self.file_repository.get_file_metadata(file_id)
        if not file_metadata:
            raise ValidationException("file_id", f"File not found: {file_id}")
        
        return file_metadata
    
    async def get_file_content(self, file_id: str) -> bytes:
        """Get file content by ID"""
        # Verify file exists
        await self.get_file_metadata(file_id)
        
        file_content = await self.file_repository.get_file_content(file_id)
        if file_content is None:
            raise ExternalServiceException("FileStorage", "File content not available")
        
        return file_content
    
    async def delete_file(self, file_id: str, deleted_by: Optional[str] = None) -> bool:
        """Delete file and associated document"""
        # Get file metadata
        file_metadata = await self.get_file_metadata(file_id)
        
        try:
            # Find associated document
            document_id = file_metadata.validation_result.get('document_id')
            if document_id:
                try:
                    await self.document_repository.delete(document_id)
                except Exception as e:
                    await self.logging_service.log_warning(
                        f"Failed to delete associated document: {document_id}",
                        {"file_id": file_id, "document_id": document_id}
                    )
            
            # Delete file
            success = await self.file_repository.delete_file(file_id)
            
            if success:
                # Log deletion
                await self.logging_service.log_info(
                    f"File deleted: {file_metadata.filename}",
                    {"file_id": file_id, "deleted_by": deleted_by}
                )
                
                # Publish event
                await self.event_publisher.publish_event(
                    "file.deleted",
                    {
                        "file_id": file_id,
                        "filename": file_metadata.filename,
                        "deleted_by": deleted_by,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return success
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to delete file: {file_id}",
                e,
                {"file_id": file_id, "deleted_by": deleted_by}
            )
            raise
    
    async def list_files(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[list[FileUpload], int]:
        """List files with pagination"""
        
        # Validate pagination
        if page < 1:
            raise ValidationException("page", "Page must be greater than 0")
        
        if page_size < 1 or page_size > 100:
            raise ValidationException("page_size", "Page size must be between 1 and 100")
        
        try:
            # Get files (implementation depends on repository)
            files = await self.file_repository.list_files(user_id)
            
            # Simple pagination (in a real implementation, this would be done in the repository)
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            paginated_files = files[start_index:end_index]
            
            total_count = len(files)
            
            return paginated_files, total_count
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to list files",
                e,
                {"user_id": user_id, "page": page, "page_size": page_size}
            )
            raise
    
    async def get_file_statistics(self) -> Dict[str, Any]:
        """Get file upload statistics"""
        try:
            # Get all files
            all_files = await self.file_repository.list_files()
            
            # Calculate statistics
            total_files = len(all_files)
            total_size = sum(f.size_bytes for f in all_files)
            
            # Group by content type
            content_type_stats = {}
            processing_status_stats = {}
            upload_by_user = {}
            
            recent_count = 0
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            for file_upload in all_files:
                # Content type stats
                ct = file_upload.content_type
                content_type_stats[ct] = content_type_stats.get(ct, 0) + 1
                
                # Processing status stats
                status = file_upload.processing_status.value
                processing_status_stats[status] = processing_status_stats.get(status, 0) + 1
                
                # Upload by user
                user = file_upload.uploaded_by or "anonymous"
                upload_by_user[user] = upload_by_user.get(user, 0) + 1
                
                # Recent uploads
                if file_upload.uploaded_at >= cutoff_date:
                    recent_count += 1
            
            statistics = {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files_today": recent_count,
                "content_type_distribution": content_type_stats,
                "processing_status_distribution": processing_status_stats,
                "uploads_by_user": upload_by_user,
                "last_updated": datetime.now().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            await self.logging_service.log_error("Failed to get file statistics", e)
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        dangerous_chars = '<>:"|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
        
        return filename
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract document title from filename"""
        # Remove extension
        title = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        # Replace underscores and hyphens with spaces
        title = title.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title or "Untitled Document"
    
    async def _extract_text_content(self, file_data: bytes, content_type: str) -> str:
        """Extract text content from file data"""
        try:
            # Simple text extraction based on content type
            if content_type.startswith('text/'):
                # Text files
                return file_data.decode('utf-8', errors='ignore')
            
            elif content_type == 'application/json':
                # JSON files
                import json
                try:
                    data = json.loads(file_data.decode('utf-8'))
                    return json.dumps(data, indent=2)
                except:
                    return file_data.decode('utf-8', errors='ignore')
            
            elif content_type in ['application/csv', 'text/csv']:
                # CSV files
                return file_data.decode('utf-8', errors='ignore')
            
            else:
                # For other types, try to decode as text
                # In a real implementation, you would use appropriate libraries
                # for PDF, DOCX, etc.
                try:
                    return file_data.decode('utf-8', errors='ignore')
                except:
                    raise BusinessRuleException(
                        "unsupported_format",
                        f"Unsupported file format: {content_type}"
                    )
            
        except Exception as e:
            raise BusinessRuleException(
                "content_extraction",
                f"Failed to extract content: {e}"
            )