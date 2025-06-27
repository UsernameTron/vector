"""
Standardized API response formats and error handling
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from enum import Enum
import json


class ResponseStatus(Enum):
    """Standard response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class ErrorCategory(Enum):
    """Error categories for consistent error handling"""
    VALIDATION = "validation_error"
    AUTHENTICATION = "authentication_error"
    AUTHORIZATION = "authorization_error"
    NOT_FOUND = "not_found_error"
    CONFLICT = "conflict_error"
    BUSINESS_RULE = "business_rule_error"
    EXTERNAL_SERVICE = "external_service_error"
    SYSTEM = "system_error"
    RATE_LIMIT = "rate_limit_error"


@dataclass
class ErrorDetail:
    """Detailed error information"""
    code: str
    message: str
    field: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class PaginationInfo:
    """Pagination information for list responses"""
    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool


@dataclass
class ApiResponse:
    """Standard API response format"""
    status: ResponseStatus
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: List[ErrorDetail] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None
    pagination: Optional[PaginationInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        result = {
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
        }
        
        if self.data is not None:
            result['data'] = self._serialize_data(self.data)
        
        if self.message:
            result['message'] = self.message
        
        if self.errors:
            result['errors'] = [asdict(error) for error in self.errors]
        
        if self.warnings:
            result['warnings'] = self.warnings
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        if self.request_id:
            result['request_id'] = self.request_id
        
        if self.pagination:
            result['pagination'] = asdict(self.pagination)
        
        return result
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON response"""
        if hasattr(data, '__dict__'):
            # Handle dataclass or object with attributes
            if hasattr(data, '__dataclass_fields__'):
                return asdict(data)
            else:
                return {key: self._serialize_data(value) for key, value in data.__dict__.items() 
                       if not key.startswith('_')}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, Enum):
            return data.value
        else:
            return data


class ResponseBuilder:
    """Builder for creating standardized API responses"""
    
    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id
    
    def success(
        self,
        data: Any = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ApiResponse:
        """Create a success response"""
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=data,
            message=message,
            metadata=metadata or {},
            request_id=self.request_id
        )
    
    def error(
        self,
        errors: Union[ErrorDetail, List[ErrorDetail]],
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ApiResponse:
        """Create an error response"""
        if isinstance(errors, ErrorDetail):
            errors = [errors]
        
        return ApiResponse(
            status=ResponseStatus.ERROR,
            errors=errors,
            message=message,
            metadata=metadata or {},
            request_id=self.request_id
        )
    
    def validation_error(
        self,
        field: str,
        message: str,
        code: str = "VALIDATION_FAILED"
    ) -> ApiResponse:
        """Create a validation error response"""
        error = ErrorDetail(
            code=code,
            message=message,
            field=field
        )
        return self.error(error, "Validation failed")
    
    def not_found(
        self,
        resource: str,
        identifier: str = None
    ) -> ApiResponse:
        """Create a not found error response"""
        message = f"{resource} not found"
        if identifier:
            message += f" with identifier: {identifier}"
        
        error = ErrorDetail(
            code="NOT_FOUND",
            message=message,
            context={"resource": resource, "identifier": identifier}
        )
        return self.error(error, message)
    
    def unauthorized(self, message: str = "Authentication required") -> ApiResponse:
        """Create an unauthorized error response"""
        error = ErrorDetail(
            code="UNAUTHORIZED",
            message=message
        )
        return self.error(error, message)
    
    def forbidden(self, message: str = "Access denied") -> ApiResponse:
        """Create a forbidden error response"""
        error = ErrorDetail(
            code="FORBIDDEN",
            message=message
        )
        return self.error(error, message)
    
    def conflict(self, message: str, resource: str = None) -> ApiResponse:
        """Create a conflict error response"""
        error = ErrorDetail(
            code="CONFLICT",
            message=message,
            context={"resource": resource} if resource else None
        )
        return self.error(error, message)
    
    def business_rule_violation(
        self,
        rule: str,
        message: str
    ) -> ApiResponse:
        """Create a business rule violation error response"""
        error = ErrorDetail(
            code="BUSINESS_RULE_VIOLATION",
            message=message,
            context={"rule": rule}
        )
        return self.error(error, "Business rule violation")
    
    def external_service_error(
        self,
        service: str,
        message: str = "External service unavailable"
    ) -> ApiResponse:
        """Create an external service error response"""
        error = ErrorDetail(
            code="EXTERNAL_SERVICE_ERROR",
            message=message,
            context={"service": service}
        )
        return self.error(error, f"{service} is currently unavailable")
    
    def system_error(
        self,
        message: str = "An unexpected error occurred"
    ) -> ApiResponse:
        """Create a system error response"""
        error = ErrorDetail(
            code="SYSTEM_ERROR",
            message=message
        )
        return self.error(error, "System error occurred")
    
    def rate_limit_exceeded(
        self,
        limit: int,
        window: str,
        retry_after: Optional[int] = None
    ) -> ApiResponse:
        """Create a rate limit exceeded error response"""
        message = f"Rate limit exceeded: {limit} requests per {window}"
        error = ErrorDetail(
            code="RATE_LIMIT_EXCEEDED",
            message=message,
            context={
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            }
        )
        return self.error(error, message)
    
    def paginated_success(
        self,
        data: List[Any],
        page: int,
        page_size: int,
        total_count: int,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ApiResponse:
        """Create a paginated success response"""
        total_pages = (total_count + page_size - 1) // page_size
        
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )
        
        response = self.success(data, message, metadata)
        response.pagination = pagination
        return response


class ApiException(Exception):
    """Base exception for API errors that should be converted to API responses"""
    
    def __init__(
        self,
        error_detail: ErrorDetail,
        status_code: int = 500,
        message: Optional[str] = None
    ):
        self.error_detail = error_detail
        self.status_code = status_code
        self.message = message
        super().__init__(message or error_detail.message)
    
    def to_response(self, request_id: Optional[str] = None) -> ApiResponse:
        """Convert exception to API response"""
        builder = ResponseBuilder(request_id)
        return builder.error(self.error_detail, self.message)


class ValidationException(ApiException):
    """Exception for validation errors"""
    
    def __init__(self, field: str, message: str, code: str = "VALIDATION_FAILED"):
        error_detail = ErrorDetail(code=code, message=message, field=field)
        super().__init__(error_detail, 400, "Validation failed")


class NotFoundException(ApiException):
    """Exception for not found errors"""
    
    def __init__(self, resource: str, identifier: str = None):
        message = f"{resource} not found"
        if identifier:
            message += f" with identifier: {identifier}"
        
        error_detail = ErrorDetail(
            code="NOT_FOUND",
            message=message,
            context={"resource": resource, "identifier": identifier}
        )
        super().__init__(error_detail, 404, message)


class UnauthorizedException(ApiException):
    """Exception for authentication errors"""
    
    def __init__(self, message: str = "Authentication required"):
        error_detail = ErrorDetail(code="UNAUTHORIZED", message=message)
        super().__init__(error_detail, 401, message)


class ForbiddenException(ApiException):
    """Exception for authorization errors"""
    
    def __init__(self, message: str = "Access denied"):
        error_detail = ErrorDetail(code="FORBIDDEN", message=message)
        super().__init__(error_detail, 403, message)


class ConflictException(ApiException):
    """Exception for conflict errors"""
    
    def __init__(self, message: str, resource: str = None):
        error_detail = ErrorDetail(
            code="CONFLICT",
            message=message,
            context={"resource": resource} if resource else None
        )
        super().__init__(error_detail, 409, message)


class BusinessRuleException(ApiException):
    """Exception for business rule violations"""
    
    def __init__(self, rule: str, message: str):
        error_detail = ErrorDetail(
            code="BUSINESS_RULE_VIOLATION",
            message=message,
            context={"rule": rule}
        )
        super().__init__(error_detail, 422, "Business rule violation")


class ExternalServiceException(ApiException):
    """Exception for external service errors"""
    
    def __init__(self, service: str, message: str = "External service unavailable"):
        error_detail = ErrorDetail(
            code="EXTERNAL_SERVICE_ERROR",
            message=message,
            context={"service": service}
        )
        super().__init__(error_detail, 502, f"{service} is currently unavailable")


class RateLimitException(ApiException):
    """Exception for rate limit errors"""
    
    def __init__(self, limit: int, window: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        error_detail = ErrorDetail(
            code="RATE_LIMIT_EXCEEDED",
            message=message,
            context={
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            }
        )
        super().__init__(error_detail, 429, message)


def create_response_builder(request_id: Optional[str] = None) -> ResponseBuilder:
    """Factory function to create response builder"""
    return ResponseBuilder(request_id)