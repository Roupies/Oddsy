#!/usr/bin/env python3
"""
Application FastAPI principale avec middleware sécurisé
=====================================================
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from datetime import datetime

from core.config import settings
from core.exceptions import PipelineError, ValidationError
from api.v1 import predictions, pipeline, health, operations, results
from api.v5 import gameweeks
from middleware.production_rate_limiter import rate_limit_middleware
from middleware.structured_logging import structured_logging_middleware
from services.production_metrics import get_metrics_service, api_metrics_middleware

# Configuration logging structuré
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(name)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Application FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="API Pipeline Oddsy - Interface vers Pipeline Durci v1.0",
    openapi_url="/api/system/openapi.json",
    docs_url="/api/system/docs",
    redoc_url="/api/system/redoc"
)

# Middleware sécurité et performance
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware Rate Limiting (en premier)
app.middleware("http")(rate_limit_middleware)

# Middleware Structured Logging (après rate limiting)
app.middleware("http")(structured_logging_middleware)

# Exception handlers
@app.exception_handler(PipelineError)
async def pipeline_error_handler(request: Request, exc: PipelineError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Pipeline Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error", 
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Routes API v1 (legacy aliases)
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(pipeline.router, prefix="/api/v1") 
app.include_router(health.router, prefix="/api/v1")
app.include_router(operations.router, prefix="/api/v1")
app.include_router(results.router, prefix="")

# Routes API v5 (legacy aliases)
app.include_router(gameweeks.router, prefix="/api/v5")

# Routes descriptives pour le jury
app.include_router(predictions.router, prefix="/api/system")
app.include_router(pipeline.router, prefix="/api/system") 
app.include_router(health.router, prefix="/api/system")
app.include_router(operations.router, prefix="/api/system")
app.include_router(gameweeks.router, prefix="/api/gameweeks")

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Oddsy Pipeline API",
        "version": settings.API_VERSION,
        "pipeline_version": "Pipeline_Durci_v1.0",
        "docs": "/api/system/docs",
        "health": "/api/system/health/live",
        "api_endpoints": {
            "system": "/api/system (health, metrics, pipeline ops)",
            "gameweeks": "/api/gameweeks (fixtures, predictions)",
            "legacy_v1": "/api/v1 (alias for system)",
            "legacy_v5": "/api/v5 (alias for gameweeks)"
        }
    }

# Initialize metrics service
metrics_service = get_metrics_service()

# Instrument FastAPI app with metrics
if metrics_service.enabled:
    metrics_service.instrument_fastapi(app)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Oddsy Pipeline API v{settings.API_VERSION} démarrée")
    logger.info(f"Pipeline Durci integration: {settings.PIPELINE_PREDICTIONS_DIR}")
    logger.info(f"Triggers enabled: {settings.ENABLE_PIPELINE_TRIGGERS}")
    
    # Log metrics service status
    metrics_summary = metrics_service.get_metrics_summary()
    if metrics_summary.get("enabled"):
        logger.info(f"Production metrics enabled: {len(metrics_summary.get('metrics_available', []))} metrics")
    else:
        logger.warning(f"Production metrics disabled: {metrics_summary.get('reason', 'unknown')}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.DEBUG,
        log_config=None  # Use our custom logging
    )