"""
Comprehensive health monitoring system
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from src.domain.interfaces import IHealthMonitor, ILoggingService
from src.domain.entities import HealthStatus, SystemStatus
from src.infrastructure.container import singleton

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    healthy: bool
    message: str
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


@singleton(IHealthMonitor)
class HealthMonitor(IHealthMonitor):
    """Comprehensive health monitoring system"""
    
    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self._health_checks: Dict[str, Callable] = {}
        self._health_history: Dict[str, List[HealthCheckResult]] = {}
        self._system_metrics_history: List[SystemMetrics] = []
        self._start_time = time.time()
        self._lock = threading.RLock()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks"""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("memory_usage", self._check_memory_usage)
    
    async def register_health_check(self, component_name: str, check_function: Callable) -> bool:
        """Register custom health check function"""
        try:
            with self._lock:
                self._health_checks[component_name] = check_function
                self._health_history[component_name] = []
            
            await self.logging_service.log_info(
                f"Registered health check: {component_name}",
                {"component": component_name}
            )
            return True
            
        except Exception as e:
            await self.logging_service.log_error(
                f"Failed to register health check: {component_name}",
                e,
                {"component": component_name}
            )
            return False
    
    async def check_component_health(self, component_name: str) -> HealthStatus:
        """Check health of specific component"""
        try:
            if component_name not in self._health_checks:
                return HealthStatus(
                    component=component_name,
                    healthy=False,
                    message=f"Health check not registered for component: {component_name}"
                )
            
            check_function = self._health_checks[component_name]
            
            # Measure response time
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(check_function):
                    result = await check_function()
                else:
                    result = check_function()
                
                response_time_ms = (time.time() - start_time) * 1000
                
                # Parse result
                if isinstance(result, HealthStatus):
                    health_status = result
                elif isinstance(result, dict):
                    health_status = HealthStatus(
                        component=component_name,
                        healthy=result.get('healthy', False),
                        message=result.get('message', ''),
                        details=result.get('details', {})
                    )
                elif isinstance(result, bool):
                    health_status = HealthStatus(
                        component=component_name,
                        healthy=result,
                        message="OK" if result else "Component unhealthy"
                    )
                else:
                    health_status = HealthStatus(
                        component=component_name,
                        healthy=False,
                        message=f"Invalid health check result type: {type(result)}"
                    )
                
                # Store result in history
                check_result = HealthCheckResult(
                    component=component_name,
                    healthy=health_status.healthy,
                    message=health_status.message,
                    response_time_ms=response_time_ms,
                    details=health_status.details
                )
                
                with self._lock:
                    self._health_history[component_name].append(check_result)
                    # Keep only last 100 results
                    if len(self._health_history[component_name]) > 100:
                        self._health_history[component_name] = self._health_history[component_name][-100:]
                
                return health_status
                
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                
                health_status = HealthStatus(
                    component=component_name,
                    healthy=False,
                    message=f"Health check failed: {str(e)}"
                )
                
                # Store failed result
                check_result = HealthCheckResult(
                    component=component_name,
                    healthy=False,
                    message=str(e),
                    response_time_ms=response_time_ms
                )
                
                with self._lock:
                    self._health_history[component_name].append(check_result)
                
                await self.logging_service.log_error(
                    f"Health check failed for {component_name}",
                    e,
                    {"component": component_name, "response_time_ms": response_time_ms}
                )
                
                return health_status
                
        except Exception as e:
            await self.logging_service.log_error(
                f"Error in health check system for {component_name}",
                e,
                {"component": component_name}
            )
            
            return HealthStatus(
                component=component_name,
                healthy=False,
                message=f"Health check system error: {str(e)}"
            )
    
    async def get_system_status(self) -> SystemStatus:
        """Get overall system status"""
        try:
            # Collect system metrics
            system_metrics = self._collect_system_metrics()
            
            # Run all health checks
            component_statuses = []
            for component_name in self._health_checks.keys():
                health_status = await self.check_component_health(component_name)
                component_statuses.append(health_status)
            
            # Calculate overall status
            all_healthy = all(status.healthy for status in component_statuses)
            overall_status = "healthy" if all_healthy else "unhealthy"
            
            # Calculate uptime
            uptime_seconds = time.time() - self._start_time
            uptime_str = str(timedelta(seconds=int(uptime_seconds)))
            
            # Gather statistics
            statistics = {
                "uptime_seconds": uptime_seconds,
                "total_components": len(component_statuses),
                "healthy_components": sum(1 for status in component_statuses if status.healthy),
                "system_metrics": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "memory_used_mb": system_metrics.memory_used_mb,
                    "disk_usage_percent": system_metrics.disk_usage_percent,
                    "load_average": system_metrics.load_average
                }
            }
            
            system_status = SystemStatus(
                status=overall_status,
                uptime=uptime_str,
                components=component_statuses,
                statistics=statistics
            )
            
            return system_status
            
        except Exception as e:
            await self.logging_service.log_error("Failed to get system status", e)
            
            return SystemStatus(
                status="error",
                uptime="unknown",
                components=[],
                statistics={"error": str(e)}
            )
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # Load average
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have getloadavg
                load_average = [0.0, 0.0, 0.0]
            
            # Uptime
            uptime_seconds = time.time() - self._start_time
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                load_average=load_average,
                uptime_seconds=uptime_seconds
            )
            
            # Store in history
            with self._lock:
                self._system_metrics_history.append(metrics)
                # Keep only last 100 metrics
                if len(self._system_metrics_history) > 100:
                    self._system_metrics_history = self._system_metrics_history[-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                load_average=[0.0, 0.0, 0.0],
                uptime_seconds=0.0
            )
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            metrics = self._collect_system_metrics()
            
            # Define thresholds
            cpu_threshold = 80.0
            memory_threshold = 85.0
            disk_threshold = 90.0
            
            issues = []
            
            if metrics.cpu_percent > cpu_threshold:
                issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
            if metrics.memory_percent > memory_threshold:
                issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
            if metrics.disk_usage_percent > disk_threshold:
                issues.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
            
            healthy = len(issues) == 0
            message = "System resources OK" if healthy else "; ".join(issues)
            
            return {
                "healthy": healthy,
                "message": message,
                "details": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "disk_usage_percent": metrics.disk_usage_percent,
                    "load_average": metrics.load_average
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Failed to check system resources: {e}",
                "details": {}
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024 * 1024 * 1024)
            usage_percent = disk.percent
            
            # Thresholds
            min_free_gb = 1.0  # 1GB minimum
            max_usage_percent = 95.0
            
            issues = []
            
            if free_gb < min_free_gb:
                issues.append(f"Low disk space: {free_gb:.1f}GB free")
            
            if usage_percent > max_usage_percent:
                issues.append(f"Disk almost full: {usage_percent:.1f}% used")
            
            healthy = len(issues) == 0
            message = "Disk space OK" if healthy else "; ".join(issues)
            
            return {
                "healthy": healthy,
                "message": message,
                "details": {
                    "free_gb": free_gb,
                    "usage_percent": usage_percent,
                    "total_gb": disk.total / (1024 * 1024 * 1024)
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Failed to check disk space: {e}",
                "details": {}
            }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 * 1024 * 1024)
            usage_percent = memory.percent
            
            # Thresholds
            min_available_gb = 0.5  # 500MB minimum
            max_usage_percent = 90.0
            
            issues = []
            
            if available_gb < min_available_gb:
                issues.append(f"Low memory: {available_gb:.1f}GB available")
            
            if usage_percent > max_usage_percent:
                issues.append(f"High memory usage: {usage_percent:.1f}%")
            
            healthy = len(issues) == 0
            message = "Memory usage OK" if healthy else "; ".join(issues)
            
            return {
                "healthy": healthy,
                "message": message,
                "details": {
                    "available_gb": available_gb,
                    "usage_percent": usage_percent,
                    "total_gb": memory.total / (1024 * 1024 * 1024)
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Failed to check memory usage: {e}",
                "details": {}
            }
    
    def get_health_history(self, component_name: str, limit: int = 50) -> List[HealthCheckResult]:
        """Get health check history for a component"""
        with self._lock:
            history = self._health_history.get(component_name, [])
            return history[-limit:] if history else []
    
    def get_system_metrics_history(self, limit: int = 50) -> List[SystemMetrics]:
        """Get system metrics history"""
        with self._lock:
            return self._system_metrics_history[-limit:] if self._system_metrics_history else []
    
    def get_registered_components(self) -> List[str]:
        """Get list of registered health check components"""
        with self._lock:
            return list(self._health_checks.keys())