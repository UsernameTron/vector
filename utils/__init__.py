"""
Utilities package for Vector RAG Database
"""

from .error_handler import (
    ErrorCategories,
    ErrorRecovery,
    DependencyManager,
    FileCleanupManager,
    handle_errors,
    safe_operation
)

__all__ = [
    'ErrorCategories',
    'ErrorRecovery',
    'DependencyManager',
    'FileCleanupManager',
    'handle_errors',
    'safe_operation'
]