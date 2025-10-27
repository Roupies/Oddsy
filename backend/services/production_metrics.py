#!/usr/bin/env python3
"""
Production Metrics Service pour API v5.3
========================================

Service de métriques production avec Prometheus et OpenTelemetry
pour monitoring temps-réel et observabilité.

Métriques clés:
- Latence et disponibilité API
- Cache hits/misses et performance  
- Rate limiting et security events
- Data validation et integrity checks
- Pipeline reliability et coverage
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import contextmanager
from functools import wraps
import asyncio

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Prometheus imports
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available - metrics disabled")


class ProductionMetricsService:
    """Service de métriques production complet"""
    
    def __init__(self, 
                 service_name: str = "oddsy-api",
                 enable_prometheus: bool = True,
                 enable_otlp: bool = True,
                 otlp_endpoint: str = None):
        """
        Initialise le service de métriques
        
        Args:
            service_name: Nom du service pour les métriques
            enable_prometheus: Activer export Prometheus
            enable_otlp: Activer export OTLP (Jaeger/Grafana)
            otlp_endpoint: Endpoint OTLP (défaut: OTEL_EXPORTER_OTLP_ENDPOINT)
        """
        self.service_name = service_name
        self.logger = logging.getLogger('ProductionMetrics')
        self.enabled = PROMETHEUS_AVAILABLE and OPENTELEMETRY_AVAILABLE and (enable_prometheus or enable_otlp)
        
        if not self.enabled:
            self.logger.warning("⚠️ Production metrics disabled - missing dependencies")
            return
        
        # Configuration OpenTelemetry
        self._setup_opentelemetry(enable_otlp, otlp_endpoint)
        
        # Configuration Prometheus
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        
        self.logger.info(f"✅ Production metrics enabled for {service_name}")
    
    def _setup_opentelemetry(self, enable_otlp: bool, otlp_endpoint: str):
        """Configure OpenTelemetry tracing et metrics"""
        
        if not OPENTELEMETRY_AVAILABLE:
            self.tracer = None
            return
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(self.service_name)
        
        # Setup OTLP export si activé
        if enable_otlp:
            otlp_endpoint = otlp_endpoint or os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
            if otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
                self.logger.info(f"🔗 OTLP export enabled: {otlp_endpoint}")
        
        # Setup metrics avec Prometheus reader
        if PROMETHEUS_AVAILABLE:
            prometheus_reader = PrometheusMetricReader()
            metrics.set_meter_provider(MeterProvider(metric_readers=[prometheus_reader]))
            self.meter = metrics.get_meter(self.service_name)
    
    def _setup_prometheus_metrics(self):
        """Initialise les métriques Prometheus spécialisées"""
        
        # === API Performance Metrics ===
        self.api_requests_total = Counter(
            'oddsy_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code', 'api_version']
        )
        
        self.api_request_duration = Histogram(
            'oddsy_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint', 'api_version'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.api_active_requests = Gauge(
            'oddsy_api_active_requests',
            'Currently active API requests',
            ['endpoint']
        )
        
        # === Cache Metrics ===
        self.cache_operations_total = Counter(
            'oddsy_cache_operations_total',
            'Total cache operations',
            ['operation', 'cache_type', 'result']  # operation: get/set/delete, result: hit/miss/error
        )
        
        self.cache_hit_ratio = Gauge(
            'oddsy_cache_hit_ratio',
            'Cache hit ratio by type',
            ['cache_type']
        )
        
        self.cache_size_bytes = Gauge(
            'oddsy_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type']
        )
        
        # === Rate Limiting Metrics ===
        self.rate_limit_events_total = Counter(
            'oddsy_rate_limit_events_total',
            'Rate limiting events',
            ['client_id', 'endpoint', 'action']  # action: allowed/blocked/warning
        )
        
        self.rate_limit_current_usage = Gauge(
            'oddsy_rate_limit_current_usage',
            'Current rate limit usage',
            ['client_id', 'endpoint']
        )
        
        # === Data Validation Metrics ===
        self.data_validation_total = Counter(
            'oddsy_data_validation_total',
            'Data validation operations',
            ['validator_type', 'gameweek', 'result']  # result: success/error/warning
        )
        
        self.probability_validation_errors = Counter(
            'oddsy_probability_validation_errors_total',
            'Probability validation errors',
            ['error_type', 'normalization_method']
        )
        
        self.coverage_validation_status = Gauge(
            'oddsy_coverage_validation_status',
            'Coverage validation status (1=valid, 0=invalid)',
            ['gameweek']
        )
        
        # === Pipeline Metrics ===
        self.pipeline_operations_total = Counter(
            'oddsy_pipeline_operations_total',
            'Pipeline operations',
            ['operation_type', 'gameweek', 'status']  # status: success/error/timeout
        )
        
        self.pipeline_duration = Histogram(
            'oddsy_pipeline_duration_seconds',
            'Pipeline operation duration',
            ['operation_type', 'gameweek'],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600]
        )
        
        self.revalidation_operations_total = Counter(
            'oddsy_revalidation_operations_total',
            'ISR revalidation operations',
            ['operation_type', 'status', 'frontend_url']
        )
        
        # === System Health Metrics ===
        self.system_info = Info(
            'oddsy_system_info',
            'System information'
        )
        
        self.degraded_mode_active = Gauge(
            'oddsy_degraded_mode_active',
            'Degraded mode status (1=active, 0=normal)',
            ['mode_type']
        )
        
        self.last_successful_operation = Gauge(
            'oddsy_last_successful_operation_timestamp',
            'Timestamp of last successful operation',
            ['operation_type']
        )
        
        # Initialize system info
        self.system_info.info({
            'service': self.service_name,
            'version': os.getenv('APP_VERSION', 'unknown'),
            'environment': os.getenv('ENVIRONMENT', 'unknown'),
            'git_sha': os.getenv('GIT_SHA', 'unknown')
        })
    
    def record_api_request(self, 
                          method: str, 
                          endpoint: str, 
                          status_code: int,
                          duration_seconds: float,
                          api_version: str = "v5"):
        """Enregistre une requête API"""
        if not self.enabled:
            return
            
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint, 
            status_code=str(status_code),
            api_version=api_version
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint,
            api_version=api_version
        ).observe(duration_seconds)
    
    @contextmanager
    def track_api_request(self, method: str, endpoint: str, api_version: str = "v5"):
        """Context manager pour tracker automatiquement une requête API"""
        if not self.enabled:
            yield
            return
            
        self.api_active_requests.labels(endpoint=endpoint).inc()
        start_time = time.time()
        status_code = 500  # Default to error
        
        try:
            yield
            status_code = 200  # Success
        except Exception as e:
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            raise
        finally:
            duration = time.time() - start_time
            self.api_active_requests.labels(endpoint=endpoint).dec()
            self.record_api_request(method, endpoint, status_code, duration, api_version)
    
    def record_cache_operation(self, 
                              operation: str, 
                              cache_type: str, 
                              result: str):
        """Enregistre une opération de cache"""
        if not self.enabled:
            return
            
        self.cache_operations_total.labels(
            operation=operation,
            cache_type=cache_type,
            result=result
        ).inc()
    
    def update_cache_hit_ratio(self, cache_type: str, hit_ratio: float):
        """Met à jour le ratio de hit du cache"""
        if not self.enabled:
            return
            
        self.cache_hit_ratio.labels(cache_type=cache_type).set(hit_ratio)
    
    def record_rate_limit_event(self, 
                               client_id: str, 
                               endpoint: str, 
                               action: str):
        """Enregistre un événement de rate limiting"""
        if not self.enabled:
            return
            
        self.rate_limit_events_total.labels(
            client_id=client_id,
            endpoint=endpoint,
            action=action
        ).inc()
    
    def update_rate_limit_usage(self, 
                               client_id: str, 
                               endpoint: str, 
                               current_usage: int):
        """Met à jour l'usage actuel du rate limiting"""
        if not self.enabled:
            return
            
        self.rate_limit_current_usage.labels(
            client_id=client_id,
            endpoint=endpoint
        ).set(current_usage)
    
    def record_data_validation(self, 
                              validator_type: str, 
                              gameweek: int, 
                              result: str):
        """Enregistre une validation de données"""
        if not self.enabled:
            return
            
        self.data_validation_total.labels(
            validator_type=validator_type,
            gameweek=str(gameweek),
            result=result
        ).inc()
    
    def record_probability_validation_error(self, 
                                          error_type: str, 
                                          normalization_method: str):
        """Enregistre une erreur de validation des probabilités"""
        if not self.enabled:
            return
            
        self.probability_validation_errors.labels(
            error_type=error_type,
            normalization_method=normalization_method
        ).inc()
    
    def update_coverage_validation_status(self, gameweek: int, is_valid: bool):
        """Met à jour le statut de validation de couverture"""
        if not self.enabled:
            return
            
        self.coverage_validation_status.labels(
            gameweek=str(gameweek)
        ).set(1 if is_valid else 0)
    
    def record_pipeline_operation(self, 
                                 operation_type: str, 
                                 gameweek: int, 
                                 status: str, 
                                 duration_seconds: float):
        """Enregistre une opération de pipeline"""
        if not self.enabled:
            return
            
        self.pipeline_operations_total.labels(
            operation_type=operation_type,
            gameweek=str(gameweek),
            status=status
        ).inc()
        
        self.pipeline_duration.labels(
            operation_type=operation_type,
            gameweek=str(gameweek)
        ).observe(duration_seconds)
        
        if status == "success":
            self.last_successful_operation.labels(
                operation_type=operation_type
            ).set(time.time())
    
    def record_revalidation_operation(self, 
                                    operation_type: str, 
                                    status: str, 
                                    frontend_url: str):
        """Enregistre une opération de revalidation"""
        if not self.enabled:
            return
            
        self.revalidation_operations_total.labels(
            operation_type=operation_type,
            status=status,
            frontend_url=frontend_url
        ).inc()
    
    def set_degraded_mode(self, mode_type: str, is_active: bool):
        """Met à jour le statut du mode dégradé"""
        if not self.enabled:
            return
            
        self.degraded_mode_active.labels(mode_type=mode_type).set(1 if is_active else 0)
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager pour tracer une opération avec OpenTelemetry"""
        if not self.enabled:
            yield
            return
            
        with self.tracer.start_as_current_span(operation_name) as span:
            # Ajouter les attributs custom
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            try:
                yield span
                span.set_status(trace.Status(trace.StatusCode.OK))
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques pour debugging"""
        if not self.enabled:
            return {"enabled": False, "reason": "dependencies_missing"}
        
        try:
            # Récupérer quelques métriques clés via le registre Prometheus
            registry = prometheus_client.REGISTRY
            
            summary = {
                "enabled": True,
                "service": self.service_name,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics_available": []
            }
            
            # Lister les métriques disponibles
            for collector in registry._collector_to_names.keys():
                if hasattr(collector, '_name') and collector._name.startswith('oddsy_'):
                    summary["metrics_available"].append(collector._name)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating metrics summary: {e}")
            return {"enabled": True, "error": str(e)}
    
    def instrument_fastapi(self, app):
        """Instrumente automatiquement une application FastAPI"""
        if not self.enabled:
            return
            
        try:
            # Instrumenter FastAPI avec OpenTelemetry
            FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
            
            # Instrumenter requests pour les calls externes
            RequestsInstrumentor().instrument()
            
            self.logger.info("✅ FastAPI instrumentation enabled")
            
        except Exception as e:
            self.logger.error(f"Error instrumenting FastAPI: {e}")


def api_metrics_middleware(metrics_service: ProductionMetricsService):
    """Middleware FastAPI pour métriques automatiques"""
    
    def metrics_decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extraire les infos de la requête FastAPI
            request = args[0] if args else kwargs.get('request')
            if not request:
                return await func(*args, **kwargs)
            
            method = request.method
            path = request.url.path
            
            with metrics_service.track_api_request(method, path):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Version synchrone
            request = args[0] if args else kwargs.get('request')
            if not request:
                return func(*args, **kwargs)
            
            method = request.method
            path = request.url.path
            
            with metrics_service.track_api_request(method, path):
                return func(*args, **kwargs)
        
        # Retourner la version appropriée selon si la fonction est async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return metrics_decorator


# Instance globale du service
_metrics_service: Optional[ProductionMetricsService] = None

def get_metrics_service() -> ProductionMetricsService:
    """Récupère l'instance singleton du service de métriques"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = ProductionMetricsService()
    return _metrics_service

def init_metrics_service(**kwargs) -> ProductionMetricsService:
    """Initialise le service de métriques avec configuration custom"""
    global _metrics_service
    _metrics_service = ProductionMetricsService(**kwargs)
    return _metrics_service