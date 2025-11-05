#!/usr/bin/env python3
"""
Production Rate Limiter for API v5.3
====================================

Rate limiting middleware avec Redis backend pour production.
Protège les endpoints publics contre les abus et les pics de trafic.
"""

import time
import logging
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
import hashlib
from services.production_metrics import get_metrics_service


class ProductionRateLimiter:
    """Rate limiter production avec Redis backend"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 default_requests: int = 100,
                 default_window: int = 3600):
        """
        Initialise le rate limiter
        
        Args:
            redis_url: URL de connexion Redis
            default_requests: Nombre de requêtes par défaut
            default_window: Fenêtre temporelle en secondes
        """
        self.redis_client = self._setup_redis(redis_url) if REDIS_AVAILABLE else None
        self.default_requests = default_requests
        self.default_window = default_window
        self.logger = logging.getLogger('RateLimiter')
        
        if not REDIS_AVAILABLE:
            self.logger.warning("⚠️ Redis not available - rate limiting disabled")
        
        # Configuration par endpoint
        self.endpoint_limits = {
            "/api/v5/gameweeks/latest": {"requests": 300, "window": 3600},  # Plus demandé
            "/api/v5/gameweeks/{gw}/predictions": {"requests": 200, "window": 3600},
            "/api/v5/gameweeks/{gw}/status": {"requests": 150, "window": 3600},
            "/api/v5/gameweeks/available": {"requests": 100, "window": 3600},
            "/api/v1/health": {"requests": 500, "window": 3600},  # Health checks fréquents
            "/api/v1/ops": {"requests": 20, "window": 3600},  # Ops limitées
        }
        
    def _setup_redis(self, redis_url: str):
        """Configure la connexion Redis avec fallback"""
        if not REDIS_AVAILABLE:
            return None
            
        try:
            client = redis.from_url(redis_url, decode_responses=True)
            # Test de connexion
            client.ping()
            self.logger.info(f"✅ Redis connected: {redis_url}")
            return client
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available ({e}), using in-memory fallback")
            return None
    
    def _get_client_key(self, request: Request) -> str:
        """Génère une clé unique pour le client"""
        # Priorité: X-Forwarded-For → X-Real-IP → client IP
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
            request.headers.get("X-Real-IP") or
            getattr(request.client, "host", "unknown")
        )
        
        # User-Agent pour détecter les bots
        user_agent = request.headers.get("User-Agent", "")
        
        # Clé composite : IP + hash du User-Agent
        ua_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        return f"rate_limit:{client_ip}:{ua_hash}"
    
    def _get_endpoint_pattern(self, path: str) -> str:
        """Normalise le path vers un pattern d'endpoint"""
        # Remplacer les IDs numériques par {gw}
        import re
        normalized = re.sub(r'/\d+/', '/{gw}/', path)
        normalized = re.sub(r'/\d+$', '/{gw}', normalized)
        
        # Trouver la correspondance la plus spécifique
        for pattern in self.endpoint_limits:
            if pattern in normalized:
                return pattern
        
        # Fallback par préfixe
        if path.startswith("/api/v5/"):
            return "/api/v5/*"
        elif path.startswith("/api/v1/health"):
            return "/api/v1/health"
        elif path.startswith("/api/v1/ops"):
            return "/api/v1/ops"
        else:
            return "/*"
    
    def _get_limits(self, endpoint_pattern: str) -> Tuple[int, int]:
        """Récupère les limites pour un endpoint"""
        limits = self.endpoint_limits.get(endpoint_pattern, {
            "requests": self.default_requests,
            "window": self.default_window
        })
        return limits["requests"], limits["window"]
    
    async def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        """
        Vérifie les limites de taux pour une requête
        
        Returns:
            JSONResponse avec 429 si limite dépassée, None sinon
        """
        
        if not self.redis_client:
            # Mode dégradé sans Redis - on laisse passer
            return None
        
        client_key = self._get_client_key(request)
        endpoint_pattern = self._get_endpoint_pattern(str(request.url.path))
        max_requests, window_seconds = self._get_limits(endpoint_pattern)
        
        # Clé Redis unique par client/endpoint/fenêtre
        current_window = int(time.time()) // window_seconds
        redis_key = f"{client_key}:{endpoint_pattern}:{current_window}"
        
        try:
            # Atomic increment avec TTL
            pipe = self.redis_client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds)
            results = pipe.execute()
            
            current_requests = results[0]
            
            # Enregistrer les métriques
            metrics_service = get_metrics_service()
            metrics_service.update_rate_limit_usage(client_key, endpoint_pattern, current_requests)
            
            # Calculer les headers de réponse
            remaining = max(0, max_requests - current_requests)
            reset_time = (current_window + 1) * window_seconds
            
            # Headers rate limit standard
            headers = {
                "X-RateLimit-Limit": str(max_requests),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time),
                "X-RateLimit-Window": str(window_seconds)
            }
            
            # Vérifier si limite dépassée
            if current_requests > max_requests:
                retry_after = reset_time - int(time.time())
                
                # Enregistrer l'événement de blocage
                metrics_service.record_rate_limit_event(client_key, endpoint_pattern, "blocked")
                
                # Log de l'incident
                self.logger.warning(
                    f"Rate limit exceeded: {client_key} on {endpoint_pattern} "
                    f"({current_requests}/{max_requests})"
                )
                
                return JSONResponse(
                    status_code=429,
                    headers={
                        **headers,
                        "Retry-After": str(max(1, retry_after))
                    },
                    content={
                        "error": "Rate limit exceeded",
                        "error_type": "rate_limit_exceeded",
                        "message": f"Too many requests. Maximum {max_requests} requests per {window_seconds} seconds.",
                        "retry_after_seconds": max(1, retry_after),
                        "endpoint_pattern": endpoint_pattern,
                        "current_requests": current_requests,
                        "limit": max_requests,
                        "window_seconds": window_seconds,
                        "reset_time": datetime.fromtimestamp(reset_time).isoformat()
                    }
                )
            
            # Ajouter les headers à la requête pour que les middlewares suivants puissent les voir
            request.state.rate_limit_headers = headers
            
            # Enregistrer l'événement d'autorisation
            if current_requests > max_requests * 0.8:
                # Warning quand proche de la limite
                metrics_service.record_rate_limit_event(client_key, endpoint_pattern, "warning")
                self.logger.info(
                    f"Rate limit warning: {client_key} on {endpoint_pattern} "
                    f"({current_requests}/{max_requests}) - {remaining} remaining"
                )
            else:
                # Requête normale autorisée
                metrics_service.record_rate_limit_event(client_key, endpoint_pattern, "allowed")
            
            return None  # Pas de limite atteinte
            
        except Exception as e:
            self.logger.error(f"Rate limiter error: {e}")
            # En cas d'erreur Redis, on laisse passer (fail-open)
            return None
    
    def get_client_stats(self, request: Request) -> Dict[str, Any]:
        """Récupère les statistiques du client (pour debugging)"""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        client_key = self._get_client_key(request)
        endpoint_pattern = self._get_endpoint_pattern(str(request.url.path))
        max_requests, window_seconds = self._get_limits(endpoint_pattern)
        
        current_window = int(time.time()) // window_seconds
        redis_key = f"{client_key}:{endpoint_pattern}:{current_window}"
        
        try:
            current_requests = int(self.redis_client.get(redis_key) or 0)
            remaining = max(0, max_requests - current_requests)
            reset_time = (current_window + 1) * window_seconds
            
            return {
                "client_key": client_key,
                "endpoint_pattern": endpoint_pattern,
                "current_requests": current_requests,
                "limit": max_requests,
                "remaining": remaining,
                "window_seconds": window_seconds,
                "reset_time": datetime.fromtimestamp(reset_time).isoformat(),
                "rate_limited": current_requests >= max_requests
            }
        except Exception as e:
            return {"error": str(e)}


# Instance globale du rate limiter
_rate_limiter: Optional[ProductionRateLimiter] = None

def get_rate_limiter() -> ProductionRateLimiter:
    """Récupère l'instance singleton du rate limiter"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = ProductionRateLimiter()
    return _rate_limiter


# Middleware FastAPI
async def rate_limit_middleware(request: Request, call_next):
    """Middleware FastAPI pour rate limiting"""
    
    # Ignorer les fichiers statiques
    if request.url.path.startswith("/static/") or request.url.path.startswith("/_next/"):
        return await call_next(request)
    
    rate_limiter = get_rate_limiter()
    
    # Vérifier les limites
    rate_limit_response = await rate_limiter.check_rate_limit(request)
    if rate_limit_response:
        return rate_limit_response
    
    # Continuer vers le handler
    response = await call_next(request)
    
    # Ajouter les headers de rate limit à la réponse
    if hasattr(request.state, 'rate_limit_headers'):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value
    
    return response