"""
Dependency Injection Container
Manages object creation and dependency resolution
"""

import inspect
from typing import Dict, Type, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime management"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceDescriptor:
    """Describes how a service should be created and managed"""
    service_type: Type
    implementation_type: Optional[Type] = None
    instance: Optional[Any] = None
    factory: Optional[Callable] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    initialized: bool = False


class DependencyInjectionError(Exception):
    """Exception raised when dependency injection fails"""
    pass


class ServiceContainer:
    """
    Dependency injection container for managing service registration and resolution
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._building: set = set()  # Track circular dependencies
    
    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> 'ServiceContainer':
        """Register a service as singleton"""
        impl_type = implementation_type or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.SINGLETON
        )
        logger.debug(f"Registered singleton: {service_type.__name__} -> {impl_type.__name__}")
        return self
    
    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> 'ServiceContainer':
        """Register a service as transient (new instance each time)"""
        impl_type = implementation_type or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.TRANSIENT
        )
        logger.debug(f"Registered transient: {service_type.__name__} -> {impl_type.__name__}")
        return self
    
    def register_scoped(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> 'ServiceContainer':
        """Register a service as scoped (one instance per scope)"""
        impl_type = implementation_type or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.SCOPED
        )
        logger.debug(f"Registered scoped: {service_type.__name__} -> {impl_type.__name__}")
        return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'ServiceContainer':
        """Register a specific instance"""
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
            initialized=True
        )
        self._singletons[service_type] = instance
        logger.debug(f"Registered instance: {service_type.__name__}")
        return self
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T], lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'ServiceContainer':
        """Register a factory function for creating instances"""
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            lifetime=lifetime
        )
        logger.debug(f"Registered factory: {service_type.__name__} ({lifetime.value})")
        return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance"""
        if service_type in self._building:
            cycle = " -> ".join(cls.__name__ for cls in self._building) + f" -> {service_type.__name__}"
            raise DependencyInjectionError(f"Circular dependency detected: {cycle}")
        
        if service_type not in self._services:
            raise DependencyInjectionError(f"Service {service_type.__name__} is not registered")
        
        descriptor = self._services[service_type]
        
        # Handle different lifetimes
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = self._create_instance(service_type, descriptor)
            self._singletons[service_type] = instance
            return instance
        
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            instance = self._create_instance(service_type, descriptor)
            self._scoped_instances[service_type] = instance
            return instance
        
        else:  # TRANSIENT
            return self._create_instance(service_type, descriptor)
    
    def _create_instance(self, service_type: Type[T], descriptor: ServiceDescriptor) -> T:
        """Create an instance based on the service descriptor"""
        if descriptor.instance is not None:
            return descriptor.instance
        
        if descriptor.factory is not None:
            try:
                return descriptor.factory()
            except Exception as e:
                raise DependencyInjectionError(f"Factory for {service_type.__name__} failed: {e}")
        
        if descriptor.implementation_type is None:
            raise DependencyInjectionError(f"No implementation specified for {service_type.__name__}")
        
        return self._build_instance(descriptor.implementation_type)
    
    def _build_instance(self, implementation_type: Type[T]) -> T:
        """Build an instance by resolving its dependencies"""
        self._building.add(implementation_type)
        
        try:
            # Get constructor signature
            signature = inspect.signature(implementation_type.__init__)
            parameters = signature.parameters
            
            # Skip 'self' parameter
            param_names = [name for name in parameters.keys() if name != 'self']
            
            # Resolve dependencies
            dependencies = {}
            for param_name in param_names:
                param = parameters[param_name]
                
                if param.annotation == inspect.Parameter.empty:
                    raise DependencyInjectionError(
                        f"Parameter '{param_name}' in {implementation_type.__name__} has no type annotation"
                    )
                
                # Handle optional parameters
                if param.default != inspect.Parameter.empty:
                    if param.annotation in self._services:
                        dependencies[param_name] = self.resolve(param.annotation)
                    else:
                        dependencies[param_name] = param.default
                else:
                    dependencies[param_name] = self.resolve(param.annotation)
            
            # Create instance
            instance = implementation_type(**dependencies)
            logger.debug(f"Created instance: {implementation_type.__name__}")
            return instance
        
        except Exception as e:
            raise DependencyInjectionError(f"Failed to build {implementation_type.__name__}: {e}")
        
        finally:
            self._building.discard(implementation_type)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service is registered"""
        return service_type in self._services
    
    def clear_scoped(self):
        """Clear scoped instances (useful for request/response cycle)"""
        self._scoped_instances.clear()
        logger.debug("Cleared scoped instances")
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services (for debugging)"""
        return self._services.copy()
    
    def validate_registrations(self) -> List[str]:
        """Validate all service registrations and return any issues"""
        issues = []
        
        for service_type, descriptor in self._services.items():
            try:
                if descriptor.lifetime != ServiceLifetime.SINGLETON or service_type not in self._singletons:
                    # Test resolution without creating singletons
                    temp_building = self._building.copy()
                    self._building.clear()
                    try:
                        self._validate_dependencies(descriptor.implementation_type or service_type)
                    finally:
                        self._building = temp_building
            except Exception as e:
                issues.append(f"{service_type.__name__}: {e}")
        
        return issues
    
    def _validate_dependencies(self, implementation_type: Type):
        """Validate that all dependencies can be resolved"""
        if implementation_type in self._building:
            return  # Circular dependency will be caught during resolution
        
        self._building.add(implementation_type)
        
        try:
            signature = inspect.signature(implementation_type.__init__)
            parameters = signature.parameters
            
            for param_name, param in parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation == inspect.Parameter.empty:
                    raise DependencyInjectionError(f"Parameter '{param_name}' has no type annotation")
                
                if param.default == inspect.Parameter.empty:
                    if not self.is_registered(param.annotation):
                        raise DependencyInjectionError(f"Dependency '{param.annotation.__name__}' is not registered")
                    
                    # Recursively validate
                    dep_descriptor = self._services[param.annotation]
                    if dep_descriptor.implementation_type:
                        self._validate_dependencies(dep_descriptor.implementation_type)
        
        finally:
            self._building.discard(implementation_type)


class ServiceScope:
    """Context manager for service scopes"""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
    
    def __enter__(self) -> ServiceContainer:
        return self.container
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.clear_scoped()


# Global container instance
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get the global service container"""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def set_container(container: ServiceContainer):
    """Set the global service container"""
    global _container
    _container = container


def create_scope(container: Optional[ServiceContainer] = None) -> ServiceScope:
    """Create a service scope"""
    if container is None:
        container = get_container()
    return ServiceScope(container)


# Decorator for automatic service registration
def service(lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT, service_type: Optional[Type] = None):
    """Decorator to automatically register a service"""
    def decorator(cls):
        container = get_container()
        target_type = service_type or cls
        
        if lifetime == ServiceLifetime.SINGLETON:
            container.register_singleton(target_type, cls)
        elif lifetime == ServiceLifetime.SCOPED:
            container.register_scoped(target_type, cls)
        else:
            container.register_transient(target_type, cls)
        
        return cls
    return decorator


# Convenience decorators
def singleton(service_type: Optional[Type] = None):
    """Decorator to register as singleton"""
    return service(ServiceLifetime.SINGLETON, service_type)


def transient(service_type: Optional[Type] = None):
    """Decorator to register as transient"""
    return service(ServiceLifetime.TRANSIENT, service_type)


def scoped(service_type: Optional[Type] = None):
    """Decorator to register as scoped"""
    return service(ServiceLifetime.SCOPED, service_type)