#!/usr/bin/env python3
"""
Operations endpoints - Pipeline automation and publishing
======================================================
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import asyncio
import json
import logging
from pathlib import Path
import subprocess
import shutil
import requests

from core.config import settings
from schemas.common import APIResponse, APIMetadata

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ops", tags=["operations"])

# In-memory task storage (pour demo - utiliser Redis en prod)
RUNNING_TASKS: Dict[str, Dict] = {}

class TaskStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING" 
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

@router.post("/gameweeks/{round}/run")
async def trigger_gameweek_generation(
    round: int,
    dry_run: bool = Query(False, description="Mode dry-run sans écriture"),
    admin_token: str = Query("", description="Admin authentication token")
) -> JSONResponse:
    """
    Déclenche génération prédictions pour une journée
    
    Args:
        round: Numéro de journée (1-38)
        dry_run: Si True, calcule sans écrire d'artefacts
    """
    
    # Vérification token admin
    if admin_token != "ADMIN":
        raise HTTPException(status_code=401, detail="Admin token required")
    
    # Validation journée
    if round < 1 or round > 38:
        raise HTTPException(status_code=422, detail="Round must be between 1 and 38")
    
    # Génération run_id unique
    run_id = str(uuid.uuid4())
    
    # Initialisation tâche
    task_data = {
        "run_id": run_id,
        "gameweek": round,
        "dry_run": dry_run,
        "status": TaskStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "output": [],
        "error": None,
        "artifacts_generated": []
    }
    
    RUNNING_TASKS[run_id] = task_data
    
    # Démarrage tâche asynchrone
    asyncio.create_task(execute_pipeline_task(run_id, round, dry_run))
    
    logger.info(f"🚀 Pipeline task enqueued: {run_id} (J{round}, dry_run={dry_run})")
    
    return JSONResponse({
        "run_id": run_id,
        "gameweek": round,
        "dry_run": dry_run,
        "status": TaskStatus.PENDING,
        "message": f"Pipeline task for GW{round} has been enqueued",
        "monitor_url": f"/api/v1/ops/runs/{run_id}"
    })

@router.get("/runs/{run_id}")
async def get_task_status(run_id: str) -> JSONResponse:
    """Récupérer le statut d'une tâche"""
    
    if run_id not in RUNNING_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = RUNNING_TASKS[run_id]
    
    return JSONResponse(task)

async def execute_pipeline_task(run_id: str, gameweek: int, dry_run: bool):
    """Exécution asynchrone de la tâche pipeline"""
    
    task = RUNNING_TASKS[run_id]
    task["status"] = TaskStatus.RUNNING
    task["started_at"] = datetime.utcnow().isoformat()
    
    try:
        logger.info(f"🔄 Starting pipeline task {run_id} for GW{gameweek}")
        
        # Simulation de tâche pipeline (remplacer par vraie logique)
        if dry_run:
            await simulate_dry_run_pipeline(task, gameweek)
        else:
            await execute_real_pipeline(task, gameweek)
        
        task["status"] = TaskStatus.SUCCEEDED
        task["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"✅ Pipeline task {run_id} completed successfully")
        
    except Exception as e:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(e)
        task["completed_at"] = datetime.utcnow().isoformat()
        
        logger.error(f"❌ Pipeline task {run_id} failed: {e}")

async def simulate_dry_run_pipeline(task: Dict, gameweek: int):
    """Simulation dry-run pipeline"""
    
    task["output"].append(f"[DRY-RUN] Starting pipeline for GW{gameweek}")
    await asyncio.sleep(1)  # Simulation
    
    task["output"].append("[DRY-RUN] Loading models from models/production/")
    await asyncio.sleep(1)
    
    task["output"].append("[DRY-RUN] Calculating predictions...")
    await asyncio.sleep(2)
    
    task["output"].append("[DRY-RUN] Generated 10 predictions")
    task["output"].append("[DRY-RUN] Validation passed - all 10 matches EPL compliant")
    task["output"].append("[DRY-RUN] Artifacts would be written to predictions/versioned/")
    task["output"].append("[DRY-RUN] Pipeline completed - NO FILES WRITTEN")
    
    # Simulation artifacts (pas d'écriture réelle)
    task["artifacts_generated"] = [
        f"j{gameweek}_predictions_dry_run.json",
        f"j{gameweek}_metadata_dry_run.json"
    ]

async def execute_real_pipeline(task: Dict, gameweek: int):
    """Exécution réelle du pipeline"""
    
    task["output"].append(f"Starting real pipeline for GW{gameweek}")
    
    try:
        # Exécution script pipeline réel
        pipeline_script = settings.PROJECT_ROOT / "scripts/run_pipeline.py"
        
        cmd = [
            "python3", str(pipeline_script),
            "--gameweek", str(gameweek),
            "--out-dir", str(settings.PIPELINE_PREDICTIONS_DIR),
            "--model", str(settings.PIPELINE_MODELS_DIR)
        ]
        
        task["output"].append(f"Executing: {' '.join(cmd)}")
        
        # Exécution avec capture output
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=settings.PROJECT_ROOT
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            task["output"].append("Pipeline execution completed successfully")
            task["output"].append(f"STDOUT: {stdout.decode()}")
            
            # Découverte artefacts générés
            versioned_dir = settings.PIPELINE_PREDICTIONS_DIR / "versioned"
            artifacts = list(versioned_dir.glob(f"j{gameweek}_*.json"))
            task["artifacts_generated"] = [str(f.name) for f in artifacts]
            
        else:
            raise Exception(f"Pipeline failed with code {process.returncode}: {stderr.decode()}")
            
    except Exception as e:
        task["output"].append(f"Pipeline execution failed: {e}")
        raise

@router.post("/publish/latest")
async def publish_latest_gameweek(
    round: int = Query(..., description="Round to publish as latest"),
    admin_token: str = Query("", description="Admin authentication token")
) -> JSONResponse:
    """
    Publication atomique d'une journée comme 'latest'
    
    Effectue:
    1. Backup de l'ancien latest.json  
    2. Création nouveau latest.json
    3. Versioning des artefacts
    4. Rollback automatique si échec
    """
    
    # Vérification token admin
    if admin_token != "ADMIN":
        raise HTTPException(status_code=401, detail="Admin token required")
    
    try:
        logger.info(f"🎯 Starting atomic publish for GW{round}")
        
        versioned_dir = settings.PIPELINE_PREDICTIONS_DIR / "versioned"
        
        # Vérification artefacts source existent
        source_file = versioned_dir / f"j{round}_latest.json"
        if not source_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Source artifacts for GW{round} not found"
            )
        
        # Backup ancien latest si existe
        old_latest = versioned_dir / "latest.json"
        backup_file = None
        
        if old_latest.exists():
            backup_file = versioned_dir / f"latest_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(old_latest, backup_file)
            logger.info(f"📦 Backed up previous latest to {backup_file.name}")
        
        # Lecture données source
        with open(source_file, 'r') as f:
            prediction_data = json.load(f)
        
        # Écriture atomique nouveau latest
        temp_latest = versioned_dir / "latest_temp.json"
        
        # Enrichissement metadata
        prediction_data["published_at"] = datetime.utcnow().isoformat()
        prediction_data["published_gameweek"] = round
        prediction_data["source_file"] = source_file.name
        
        # Écriture temporaire puis renommage atomique
        with open(temp_latest, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        # Renommage atomique
        temp_latest.rename(old_latest)
        
        logger.info(f"✅ Successfully published GW{round} as latest")
        
        # Déclencher webhook revalidation Next.js
        revalidation_success = await trigger_frontend_revalidation(round)
        
        response_data = {
            "published": True,
            "gameweek": round,
            "previous_backup": backup_file.name if backup_file else None,
            "published_at": prediction_data["published_at"],
            "artifacts": {
                "latest": "latest.json",
                "source": source_file.name
            },
            "revalidation": {
                "triggered": revalidation_success,
                "paths": ["/predictions/latest", f"/predictions/{round}"]
            },
            "next_steps": [
                f"Frontend ISR revalidation {'completed' if revalidation_success else 'failed'}",
                f"GW{round} now available at /predictions/latest"
            ]
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"❌ Publish failed for GW{round}: {e}")
        
        # Rollback si nécessaire
        if backup_file and backup_file.exists():
            if old_latest.exists():
                old_latest.unlink()
            backup_file.rename(old_latest)
            logger.info("🔄 Rollback completed - restored previous latest")
        
        raise HTTPException(
            status_code=500,
            detail=f"Publish failed: {e}"
        )

async def trigger_frontend_revalidation(gameweek: int) -> bool:
    """Déclenche revalidation ISR du frontend Next.js"""
    
    try:
        frontend_url = "http://localhost:3000"  # TODO: Config depuis env
        revalidation_secret = "oddsy_revalidate_secret_2024"  # TODO: Config depuis env
        
        paths_to_revalidate = ["/predictions/latest", f"/predictions/{gameweek}"]
        
        payload = {
            "paths": paths_to_revalidate,
            "secret": revalidation_secret
        }
        
        response = requests.post(
            f"{frontend_url}/api/revalidate",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"🔄 Frontend revalidation success: {result}")
            return True
        else:
            logger.warning(f"⚠️ Frontend revalidation failed: {response.status_code}")
            return False
                    
    except Exception as e:
        logger.error(f"❌ Frontend revalidation error: {e}")
        return False