#!/usr/bin/env python3
"""
Health checks et métriques avec auto-diagnostic
==============================================
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from datetime import datetime
import psutil
import os
import logging

from core.config import settings
from services.pipeline_interface import PipelineDurciInterface
from services.production_metrics import get_metrics_service
from schemas.common import APIResponse, APIMetadata

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
async def health_check() -> JSONResponse:
    """Simple health check - système opérationnel"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "5.0.0"
    })

@router.get("/live")
async def liveness_check() -> JSONResponse:
    """Liveness probe - service répond"""
    return JSONResponse({
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "oddsy-pipeline-api"
    })

@router.get("/ready") 
async def readiness_check() -> JSONResponse:
    """
    Readiness probe - service prêt à servir
    
    Recommandations appliquées:
    - Propager erreur si schémas ne valident pas (auto-diagnostic)
    """
    
    checks = {}
    overall_ready = True
    
    # Check 1: Répertoires Pipeline accessibles
    try:
        predictions_accessible = settings.PIPELINE_PREDICTIONS_DIR.exists()
        reports_accessible = settings.PIPELINE_REPORTS_DIR.exists()
        
        checks["filesystem"] = {
            "status": "healthy" if (predictions_accessible and reports_accessible) else "unhealthy",
            "predictions_dir": str(settings.PIPELINE_PREDICTIONS_DIR),
            "reports_dir": str(settings.PIPELINE_REPORTS_DIR),
            "predictions_accessible": predictions_accessible,
            "reports_accessible": reports_accessible
        }
        
        if not (predictions_accessible and reports_accessible):
            overall_ready = False
            
    except Exception as e:
        checks["filesystem"] = {"status": "error", "error": str(e)}
        overall_ready = False
    
    # Check 2: Pipeline Interface fonctionne
    try:
        pipeline = PipelineDurciInterface()
        status = pipeline.get_pipeline_status()
        
        checks["pipeline_interface"] = {
            "status": "healthy",
            "pipeline_version": status.pipeline_version
        }
        
    except Exception as e:
        checks["pipeline_interface"] = {"status": "error", "error": str(e)}
        overall_ready = False
    
    # Check 3: Validation schémas (auto-diagnostic)
    try:
        # Test validation schéma de base
        from schemas.predictions import PredictionProbabilities
        
        # Test valide
        PredictionProbabilities(home=0.5, draw=0.3, away=0.2)
        
        # Test invalide (doit lever exception)
        try:
            PredictionProbabilities(home=0.8, draw=0.3, away=0.2)  # Somme > 1
            checks["schema_validation"] = {"status": "error", "error": "Validation failed"}
            overall_ready = False
        except ValueError:
            # Attendu - validation fonctionne
            checks["schema_validation"] = {"status": "healthy"}
            
    except Exception as e:
        checks["schema_validation"] = {"status": "error", "error": str(e)}
        overall_ready = False
    
    response_data = {
        "status": "ready" if overall_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }
    
    status_code = 200 if overall_ready else 503
    return JSONResponse(response_data, status_code=status_code)

@router.get("/metrics")
async def get_metrics() -> JSONResponse:
    """
    Métriques système et applicatives
    
    Recommandations appliquées:
    - Log structuré minimal
    - Git SHA via env
    - Metrics de base système
    """
    
    try:
        # Métriques système
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Métriques Pipeline
        predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
        prediction_files = list(predictions_dir.glob("j*_dual_champions_*.json"))
        
        total_predictions_size = sum(f.stat().st_size for f in prediction_files)
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": {
                "api_version": settings.API_VERSION,
                "pipeline_version": "Pipeline_Durci_v1.0",
                "git_sha": settings.GIT_SHA
            },
            "system": {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "pipeline": {
                "total_prediction_files": len(prediction_files),
                "total_predictions_size_mb": round(total_predictions_size / (1024**2), 2),
                "prediction_files_dir": str(predictions_dir)
            },
            "settings": {
                "enable_pipeline_triggers": settings.ENABLE_PIPELINE_TRIGGERS,
                "max_concurrent_jobs": settings.MAX_CONCURRENT_JOBS,
                "cache_ttl_past_rounds": settings.CACHE_TTL_PAST_ROUNDS,
                "cache_ttl_current_round": settings.CACHE_TTL_CURRENT_ROUND
            }
        }
        
        return JSONResponse(metrics)
        
    except Exception as e:
        logger.error(f"Erreur récupération métriques: {e}")
        return JSONResponse(
            {"error": "Erreur récupération métriques", "detail": str(e)},
            status_code=500
        )

@router.get("/db")
async def database_check() -> JSONResponse:
    """Check rapide des dépendances critiques"""
    checks = {}
    overall_healthy = True
    
    # Check filesystem accès
    try:
        predictions_readable = settings.PIPELINE_PREDICTIONS_DIR.exists() and os.access(settings.PIPELINE_PREDICTIONS_DIR, os.R_OK)
        predictions_writable = os.access(settings.PIPELINE_PREDICTIONS_DIR, os.W_OK)
        
        checks["filesystem"] = {
            "predictions_readable": predictions_readable,
            "predictions_writable": predictions_writable,
            "status": "healthy" if (predictions_readable and predictions_writable) else "degraded"
        }
        
        if not (predictions_readable and predictions_writable):
            overall_healthy = False
            
    except Exception as e:
        checks["filesystem"] = {"status": "error", "error": str(e)}
        overall_healthy = False
    
    # Check latest.json exists
    try:
        latest_files = list(settings.PIPELINE_PREDICTIONS_DIR.glob("versioned/*_latest.json"))
        
        checks["pipeline_data"] = {
            "latest_files_count": len(latest_files),
            "status": "healthy" if len(latest_files) > 0 else "no_data"
        }
        
        if len(latest_files) == 0:
            overall_healthy = False
            
    except Exception as e:
        checks["pipeline_data"] = {"status": "error", "error": str(e)}
        overall_healthy = False
    
    response_data = {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }
    
    status_code = 200 if overall_healthy else 503
    return JSONResponse(response_data, status_code=status_code)

@router.get("/metrics/prometheus")
async def prometheus_metrics():
    """
    Endpoint Prometheus metrics
    
    Expose les métriques au format Prometheus pour scraping
    """
    try:
        metrics_service = get_metrics_service()
        
        if not metrics_service.enabled:
            raise HTTPException(
                status_code=503,
                detail="Metrics service disabled - missing dependencies"
            )
        
        # Import Prometheus client pour générer les métriques
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            # Générer les métriques au format Prometheus
            metrics_output = generate_latest()
            
            return JSONResponse(
                content=metrics_output.decode('utf-8'),
                headers={'Content-Type': CONTENT_TYPE_LATEST}
            )
            
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Prometheus client not available"
            )
            
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating metrics: {str(e)}"
        )

@router.get("/metrics/summary")
async def metrics_summary():
    """
    Résumé des métriques disponibles
    
    Endpoint de debugging pour vérifier l'état du système de métriques
    """
    try:
        metrics_service = get_metrics_service()
        summary = metrics_service.get_metrics_summary()
        
        return JSONResponse(summary)
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        return JSONResponse(
            {
                "enabled": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=500
        )