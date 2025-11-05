#!/usr/bin/env python3
"""
Structured Logging Middleware pour API v5.3
===========================================

Middleware pour logs JSON structurés avec correlation-ID,
métriques de performance et observabilité production.
"""

import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse


class StructuredLogger:
    """Logger structuré pour production avec correlation tracking"""
    
    def __init__(self, service_name: str = "oddsy-api"):
        self.service_name = service_name
        self.logger = logging.getLogger(f'StructuredLogger.{service_name}')
        
        # Configuration du handler JSON si pas déjà fait
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # Pas de formatter ici, on le fait manuellement pour plus de contrôle
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _generate_correlation_id(self) -> str:
        """Génère un correlation ID unique"""
        return f"req_{uuid.uuid4().hex[:12]}"
    
    def _extract_client_info(self, request: Request) -> Dict[str, Any]:
        """Extrait les informations client de la requête"""
        # IP client avec gestion des proxies
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
            request.headers.get("X-Real-IP") or
            getattr(request.client, "host", "unknown")
        )
        
        return {
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "referer": request.headers.get("Referer", ""),
            "origin": request.headers.get("Origin", ""),
            "accept": request.headers.get("Accept", ""),
            "accept_encoding": request.headers.get("Accept-Encoding", ""),
            "accept_language": request.headers.get("Accept-Language", "")
        }
    
    def _extract_cache_info(self, request: Request, response: Response) -> Dict[str, Any]:
        """Extrait les informations de cache"""
        cache_info = {
            "cache_hit": False,
            "etag_present": False,
            "if_none_match": request.headers.get("If-None-Match"),
            "if_modified_since": request.headers.get("If-Modified-Since")
        }
        
        if hasattr(response, 'headers'):
            cache_info.update({
                "etag_present": "ETag" in response.headers,
                "cache_control": response.headers.get("Cache-Control"),
                "cache_hit": response.status_code == 304
            })
        
        return cache_info
    
    def _categorize_endpoint(self, path: str) -> Dict[str, str]:
        """Catégorise l'endpoint pour les métriques"""
        if path.startswith("/api/v5/gameweeks"):
            if "/latest" in path:
                return {"endpoint_type": "gameweek_latest", "api_version": "v5"}
            elif "/predictions" in path:
                return {"endpoint_type": "gameweek_predictions", "api_version": "v5"}
            elif "/status" in path:
                return {"endpoint_type": "gameweek_status", "api_version": "v5"}
            else:
                return {"endpoint_type": "gameweek_other", "api_version": "v5"}
        elif path.startswith("/api/v1/health"):
            return {"endpoint_type": "health", "api_version": "v1"}
        elif path.startswith("/api/v1/ops"):
            return {"endpoint_type": "operations", "api_version": "v1"}
        elif path.startswith("/api/v1"):
            return {"endpoint_type": "legacy_v1", "api_version": "v1"}
        else:
            return {"endpoint_type": "other", "api_version": "unknown"}
    
    def log_request_start(self, request: Request, correlation_id: str) -> None:
        """Log le début d'une requête"""
        client_info = self._extract_client_info(request)
        endpoint_info = self._categorize_endpoint(str(request.url.path))
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            "service": self.service_name,
            "correlation_id": correlation_id,
            "event_type": "request_start",
            "http": {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": {
                    "content_type": request.headers.get("Content-Type"),
                    "content_length": request.headers.get("Content-Length"),
                    "authorization_present": "Authorization" in request.headers
                }
            },
            "client": client_info,
            "endpoint": endpoint_info
        }
        
        self.logger.info(json.dumps(log_entry, separators=(',', ':')))
    
    def log_request_end(self, 
                       request: Request, 
                       response: Response, 
                       correlation_id: str, 
                       duration_ms: float,
                       error: Optional[Exception] = None) -> None:
        """Log la fin d'une requête avec métriques"""
        
        client_info = self._extract_client_info(request)
        endpoint_info = self._categorize_endpoint(str(request.url.path))
        cache_info = self._extract_cache_info(request, response)
        
        # Classification des status codes
        status_category = "unknown"
        if 200 <= response.status_code < 300:
            status_category = "success"
        elif 300 <= response.status_code < 400:
            status_category = "redirect"
        elif 400 <= response.status_code < 500:
            status_category = "client_error"
        elif 500 <= response.status_code:
            status_category = "server_error"
        
        # Performance classification
        perf_category = "fast"
        if duration_ms > 5000:
            perf_category = "very_slow"
        elif duration_ms > 1000:
            perf_category = "slow"
        elif duration_ms > 500:
            perf_category = "medium"
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "ERROR" if error else "INFO",
            "service": self.service_name,
            "correlation_id": correlation_id,
            "event_type": "request_end",
            "http": {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "status_code": response.status_code,
                "status_category": status_category,
                "response_size_bytes": response.headers.get("Content-Length")
            },
            "performance": {
                "duration_ms": round(duration_ms, 2),
                "category": perf_category
            },
            "client": client_info,
            "endpoint": endpoint_info,
            "cache": cache_info
        }
        
        # Ajouter les détails d'erreur si présente
        if error:
            log_entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "occurred_at": datetime.utcnow().isoformat() + "Z"
            }
        
        # Ajouter les headers de rate limiting si présents
        if hasattr(request.state, 'rate_limit_headers'):
            log_entry["rate_limiting"] = {
                "limit": request.state.rate_limit_headers.get("X-RateLimit-Limit"),
                "remaining": request.state.rate_limit_headers.get("X-RateLimit-Remaining"),
                "reset": request.state.rate_limit_headers.get("X-RateLimit-Reset")
            }
        
        self.logger.info(json.dumps(log_entry, separators=(',', ':')))
    
    def log_rate_limit_exceeded(self, 
                               request: Request, 
                               correlation_id: str,
                               client_key: str,
                               limit_info: Dict[str, Any]) -> None:
        """Log spécialisé pour les dépassements de rate limit"""
        
        client_info = self._extract_client_info(request)
        endpoint_info = self._categorize_endpoint(str(request.url.path))
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "WARN",
            "service": self.service_name,
            "correlation_id": correlation_id,
            "event_type": "rate_limit_exceeded",
            "client": client_info,
            "endpoint": endpoint_info,
            "rate_limiting": {
                "client_key": client_key,
                "endpoint_pattern": limit_info.get("endpoint_pattern"),
                "current_requests": limit_info.get("current_requests"),
                "limit": limit_info.get("limit"),
                "window_seconds": limit_info.get("window_seconds"),
                "retry_after_seconds": limit_info.get("retry_after_seconds")
            },
            "http": {
                "method": request.method,
                "path": request.url.path,
                "status_code": 429
            }
        }
        
        self.logger.warning(json.dumps(log_entry, separators=(',', ':')))
    
    def log_cache_event(self, 
                       correlation_id: str,
                       event_type: str,  # "hit", "miss", "invalidation"
                       cache_key: str,
                       ttl_seconds: Optional[int] = None) -> None:
        """Log les événements de cache"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "DEBUG",
            "service": self.service_name,
            "correlation_id": correlation_id,
            "event_type": f"cache_{event_type}",
            "cache": {
                "key": cache_key,
                "operation": event_type,
                "ttl_seconds": ttl_seconds
            }
        }
        
        self.logger.debug(json.dumps(log_entry, separators=(',', ':')))


# Instance globale du logger structuré
_structured_logger: Optional[StructuredLogger] = None

def get_structured_logger() -> StructuredLogger:
    """Récupère l'instance singleton du logger structuré"""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger


async def structured_logging_middleware(request: Request, call_next):
    """Middleware FastAPI pour logging structuré avec correlation ID"""
    
    logger = get_structured_logger()
    
    # Générer correlation ID
    correlation_id = logger._generate_correlation_id()
    
    # Ajouter à l'état de la requête pour usage dans les handlers
    request.state.correlation_id = correlation_id
    
    # Log début de requête
    logger.log_request_start(request, correlation_id)
    
    start_time = time.time()
    error = None
    response = None
    
    try:
        # Exécuter la requête
        response = await call_next(request)
        
        # Ajouter correlation ID aux headers de réponse
        response.headers["X-Correlation-ID"] = correlation_id
        
    except Exception as e:
        error = e
        # Créer une réponse d'erreur
        response = JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        response.headers["X-Correlation-ID"] = correlation_id
        
    finally:
        # Calculer la durée
        duration_ms = (time.time() - start_time) * 1000
        
        # Log fin de requête
        logger.log_request_end(request, response, correlation_id, duration_ms, error)
    
    return response