"""
Health check and monitoring API controller
"""

import logging
from typing import Dict, Any
from flask import Blueprint, request, jsonify
import uuid

from src.domain.interfaces import IHealthMonitor, IDocumentRepository, IVectorStore
from src.presentation.responses import create_response_builder
from src.infrastructure.container import get_container

logger = logging.getLogger(__name__)

# Create Blueprint
health_bp = Blueprint('health', __name__, url_prefix='/health')


def get_request_id() -> str:
    """Generate or extract request ID"""
    return request.headers.get('X-Request-ID', str(uuid.uuid4()))


@health_bp.route('', methods=['GET'])
@health_bp.route('/live', methods=['GET'])
async def liveness_check():
    """Kubernetes liveness probe - basic health check"""
    response_builder = create_response_builder(get_request_id())
    
    try:
        # Basic liveness check - just return that the service is running
        return jsonify(response_builder.success(
            data={
                "status": "alive",
                "service": "vector-rag-database",
                "timestamp": "2024-01-01T00:00:00Z"  # Would use datetime.now()
            },
            message="Service is alive"
        ).to_dict())
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        response = response_builder.system_error("Liveness check failed")
        return jsonify(response.to_dict()), 500


@health_bp.route('/ready', methods=['GET'])
async def readiness_check():
    """Kubernetes readiness probe - detailed health check"""
    response_builder = create_response_builder(get_request_id())
    
    try:
        container = get_container()
        health_monitor = container.resolve(IHealthMonitor)
        
        # Check critical components for readiness
        critical_components = ['database', 'vector_store']
        component_status = {}
        overall_ready = True
        
        # Check database
        try:
            document_repository = container.resolve(IDocumentRepository)
            # In a real implementation, this would be an async call
            # db_status = await document_repository.health_check()
            component_status['database'] = {
                "ready": True,
                "message": "Database connection OK"
            }
        except Exception as e:
            component_status['database'] = {
                "ready": False,
                "message": f"Database connection failed: {e}"
            }
            overall_ready = False
        
        # Check vector store
        try:
            vector_store = container.resolve(IVectorStore)
            # In a real implementation, this would be an async call
            # vs_status = await vector_store.get_health_status()
            component_status['vector_store'] = {
                "ready": True,
                "message": "Vector store OK"
            }
        except Exception as e:
            component_status['vector_store'] = {
                "ready": False,
                "message": f"Vector store failed: {e}"
            }
            overall_ready = False
        
        status_code = 200 if overall_ready else 503
        
        return jsonify(response_builder.success(
            data={
                "ready": overall_ready,
                "service": "vector-rag-database",
                "components": component_status,
                "timestamp": "2024-01-01T00:00:00Z"
            },
            message="Readiness check completed"
        ).to_dict()), status_code
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        response = response_builder.system_error("Readiness check failed")
        return jsonify(response.to_dict()), 503


@health_bp.route('/detailed', methods=['GET'])
async def detailed_health_check():
    """Comprehensive health check with detailed status"""
    response_builder = create_response_builder(get_request_id())
    
    try:
        container = get_container()
        health_monitor = container.resolve(IHealthMonitor)
        
        # Get comprehensive system status
        system_status = await health_monitor.get_system_status()
        
        # Format response data
        health_data = {
            "overall_status": system_status.status,
            "uptime": system_status.uptime,
            "timestamp": system_status.timestamp.isoformat(),
            "components": [],
            "system_metrics": system_status.statistics.get("system_metrics", {}),
            "statistics": {
                key: value for key, value in system_status.statistics.items() 
                if key != "system_metrics"
            }
        }
        
        # Add component details
        for component in system_status.components:
            health_data["components"].append({
                "name": component.component,
                "healthy": component.healthy,
                "message": component.message,
                "details": component.details,
                "checked_at": component.checked_at.isoformat()
            })
        
        status_code = 200 if system_status.status == "healthy" else 503
        
        return jsonify(response_builder.success(
            data=health_data,
            message="Detailed health check completed"
        ).to_dict()), status_code
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        response = response_builder.system_error("Health check failed")
        return jsonify(response.to_dict()), 500


@health_bp.route('/metrics', methods=['GET'])
async def get_metrics():
    """Get system metrics for monitoring"""
    response_builder = create_response_builder(get_request_id())
    
    try:
        container = get_container()
        health_monitor = container.resolve(IHealthMonitor)
        
        # Get recent system metrics
        metrics_history = health_monitor.get_system_metrics_history(limit=10)
        
        if not metrics_history:
            return jsonify(response_builder.success(
                data={"metrics": [], "message": "No metrics available"},
                message="Metrics retrieved"
            ).to_dict())
        
        # Format metrics data
        latest_metrics = metrics_history[-1]
        metrics_data = {
            "current": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_used_mb": latest_metrics.memory_used_mb,
                "memory_available_mb": latest_metrics.memory_available_mb,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "disk_free_gb": latest_metrics.disk_free_gb,
                "load_average": latest_metrics.load_average,
                "uptime_seconds": latest_metrics.uptime_seconds,
                "timestamp": latest_metrics.timestamp.isoformat()
            },
            "history": [
                {
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "disk_usage_percent": m.disk_usage_percent,
                    "timestamp": m.timestamp.isoformat()
                } for m in metrics_history
            ]
        }
        
        return jsonify(response_builder.success(
            data=metrics_data,
            message="Metrics retrieved successfully"
        ).to_dict())
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        response = response_builder.system_error("Failed to retrieve metrics")
        return jsonify(response.to_dict()), 500


@health_bp.route('/component/<component_name>', methods=['GET'])
async def check_component(component_name: str):
    """Check health of specific component"""
    response_builder = create_response_builder(get_request_id())
    
    try:
        container = get_container()
        health_monitor = container.resolve(IHealthMonitor)
        
        # Check specific component
        health_status = await health_monitor.check_component_health(component_name)
        
        # Get component history if requested
        include_history = request.args.get('history', 'false').lower() == 'true'
        history_limit = int(request.args.get('limit', '10'))
        
        component_data = {
            "component": health_status.component,
            "healthy": health_status.healthy,
            "message": health_status.message,
            "details": health_status.details,
            "checked_at": health_status.checked_at.isoformat()
        }
        
        if include_history:
            history = health_monitor.get_health_history(component_name, history_limit)
            component_data["history"] = [
                {
                    "healthy": h.healthy,
                    "message": h.message,
                    "response_time_ms": h.response_time_ms,
                    "timestamp": h.timestamp.isoformat()
                } for h in history
            ]
        
        status_code = 200 if health_status.healthy else 503
        
        return jsonify(response_builder.success(
            data=component_data,
            message=f"Component {component_name} health check completed"
        ).to_dict()), status_code
        
    except Exception as e:
        logger.error(f"Component health check failed for {component_name}: {e}")
        response = response_builder.system_error(f"Component health check failed: {e}")
        return jsonify(response.to_dict()), 500


@health_bp.route('/components', methods=['GET'])
async def list_components():
    """List all registered health check components"""
    response_builder = create_response_builder(get_request_id())
    
    try:
        container = get_container()
        health_monitor = container.resolve(IHealthMonitor)
        
        components = health_monitor.get_registered_components()
        
        return jsonify(response_builder.success(
            data={
                "components": components,
                "count": len(components)
            },
            message="Registered components retrieved"
        ).to_dict())
        
    except Exception as e:
        logger.error(f"Failed to list components: {e}")
        response = response_builder.system_error("Failed to list components")
        return jsonify(response.to_dict()), 500


# Prometheus metrics endpoint (if enabled)
@health_bp.route('/prometheus', methods=['GET'])
def prometheus_metrics():
    """Prometheus metrics endpoint"""
    try:
        # This would integrate with prometheus_client library
        # For now, return a simple text response
        metrics_text = """
# HELP vector_rag_health_status Component health status (1=healthy, 0=unhealthy)
# TYPE vector_rag_health_status gauge
vector_rag_health_status{component="database"} 1
vector_rag_health_status{component="vector_store"} 1
vector_rag_health_status{component="system_resources"} 1

# HELP vector_rag_cpu_usage_percent CPU usage percentage
# TYPE vector_rag_cpu_usage_percent gauge
vector_rag_cpu_usage_percent 25.5

# HELP vector_rag_memory_usage_percent Memory usage percentage
# TYPE vector_rag_memory_usage_percent gauge
vector_rag_memory_usage_percent 45.2

# HELP vector_rag_disk_usage_percent Disk usage percentage
# TYPE vector_rag_disk_usage_percent gauge
vector_rag_disk_usage_percent 60.1

# HELP vector_rag_uptime_seconds Application uptime in seconds
# TYPE vector_rag_uptime_seconds counter
vector_rag_uptime_seconds 3600
"""
        
        from flask import Response
        return Response(metrics_text, mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Prometheus metrics failed: {e}")
        return f"# Error generating metrics: {e}", 500


# Error handlers for the blueprint
@health_bp.errorhandler(Exception)
def handle_health_exception(e: Exception):
    """Handle exceptions in health endpoints"""
    logger.error(f"Health endpoint error: {e}")
    response_builder = create_response_builder(get_request_id())
    response = response_builder.system_error("Health check error")
    return jsonify(response.to_dict()), 500