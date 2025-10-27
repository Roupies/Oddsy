#!/usr/bin/env python3
"""
Endpoints Pipeline status et triggers
====================================
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from core.config import settings
from services.pipeline_interface import PipelineDurciInterface  
from services.job_manager import job_manager, JobStatus
from schemas.common import APIResponse, APIMetadata
from schemas.pipeline import PipelineStatus, JobStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])

def get_pipeline_interface() -> PipelineDurciInterface:
    """Dependency injection pour interface pipeline"""
    return PipelineDurciInterface()

@router.get("/status")
async def get_pipeline_status(
    pipeline: PipelineDurciInterface = Depends(get_pipeline_interface)
) -> APIResponse[PipelineStatus]:
    """Status complet Pipeline Durci v1.0"""
    
    logger.info("Request pipeline status")
    
    status = pipeline.get_pipeline_status()
    
    meta = APIMetadata(
        api_version=settings.API_VERSION,
        pipeline_version="Pipeline_Durci_v1.0",
        generated_at=datetime.utcnow(),
        git_sha=settings.GIT_SHA
    )
    
    return APIResponse[PipelineStatus](
        meta=meta,
        data=status
    )

@router.post("/trigger/j{round}")
async def trigger_round_generation(
    round: int,
    background_tasks: BackgroundTasks
) -> APIResponse[dict]:
    """
    Déclenche génération Jx via Pipeline Durci
    
    Recommandations appliquées:
    - Contrôler settings.ENABLE_PIPELINE_TRIGGERS
    - Si False renvoyer 403
    - Utiliser job_manager.submit_pipeline_job
    - Renvoyer job_id
    """
    
    logger.info(f"Request trigger J{round}")
    
    # Vérification feature flag
    if not settings.ENABLE_PIPELINE_TRIGGERS:
        raise HTTPException(
            status_code=403,
            detail="Pipeline triggers désactivés. "
                   "Configurez ENABLE_PIPELINE_TRIGGERS=true pour activer."
        )
    
    # Validation round
    if not (8 <= round <= 38):  # J7 déjà générée
        raise HTTPException(
            status_code=400,
            detail=f"Round invalide: {round}. Seules J8-J38 peuvent être générées."
        )
    
    try:
        # Soumettre job asynchrone
        job_id = await job_manager.submit_pipeline_job(round)
        
        meta = APIMetadata(
            api_version=settings.API_VERSION,
            pipeline_version="Pipeline_Durci_v1.0",
            generated_at=datetime.utcnow(),
            git_sha=settings.GIT_SHA
        )
        
        return APIResponse[dict](
            meta=meta,
            data={
                "job_id": job_id,
                "round": round,
                "status": "submitted",
                "message": f"Génération J{round} soumise en arrière-plan",
                "check_status_url": f"/api/v1/pipeline/jobs/{job_id}"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur trigger J{round}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne pipeline")

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> APIResponse[dict]:
    """Status d'un job Pipeline par ID"""
    
    job_status = job_manager.get_job_status(job_id)
    
    if job_status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} introuvable"
        )
    
    meta = APIMetadata(
        api_version=settings.API_VERSION,
        pipeline_version="Pipeline_Durci_v1.0",
        generated_at=datetime.utcnow(),
        git_sha=settings.GIT_SHA
    )
    
    return APIResponse[dict](
        meta=meta,
        data=job_status
    )

@router.get("/jobs")
async def list_jobs(limit: int = 10) -> APIResponse[dict]:
    """Liste derniers jobs Pipeline"""
    
    if not (1 <= limit <= 50):
        raise HTTPException(status_code=400, detail="Limit doit être entre 1 et 50")
    
    jobs = job_manager.list_jobs(limit)
    
    meta = APIMetadata(
        api_version=settings.API_VERSION,
        pipeline_version="Pipeline_Durci_v1.0", 
        generated_at=datetime.utcnow(),
        git_sha=settings.GIT_SHA
    )
    
    return APIResponse[dict](
        meta=meta,
        data={
            "jobs": jobs,
            "total_displayed": len(jobs),
            "limit": limit
        }
    )

@router.post("/trigger/v2/j{round}")
async def trigger_pipeline_v2(
    round: int,
    background_tasks: BackgroundTasks,
    odds_api_key: Optional[str] = None
) -> APIResponse[dict]:
    """
    Déclenche Pipeline Durci v2.0 avec intégration odds API
    """
    
    logger.info(f"Request trigger Pipeline v2.0 J{round}")
    
    # Vérification feature flag
    if not settings.ENABLE_PIPELINE_TRIGGERS:
        raise HTTPException(
            status_code=403,
            detail="Pipeline triggers désactivés"
        )
    
    # Validation round
    if not (8 <= round <= 38):
        raise HTTPException(
            status_code=400,
            detail=f"Round invalide: {round}. J8-J38 supportées."
        )
    
    try:
        # Soumettre job pipeline v2
        job_id = await job_manager.submit_pipeline_v2_job(round, odds_api_key)
        
        meta = APIMetadata(
            api_version=settings.API_VERSION,
            pipeline_version="Pipeline_Durci_v2.0",
            generated_at=datetime.utcnow(),
            git_sha=settings.GIT_SHA
        )
        
        return APIResponse[dict](
            meta=meta,
            data={
                "job_id": job_id,
                "round": round,
                "status": "submitted",
                "pipeline_version": "v2.0",
                "message": f"Pipeline v2.0 J{round} avec odds API soumis",
                "check_status_url": f"/api/v1/pipeline/jobs/{job_id}"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur trigger Pipeline v2 J{round}: {e}")
        raise HTTPException(status_code=500, detail="Erreur pipeline v2")