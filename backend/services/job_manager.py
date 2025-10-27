#!/usr/bin/env python3
"""
Gestionnaire jobs asynchrones Pipeline
=====================================
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import subprocess
import json
import uuid
from pathlib import Path
import logging

from core.config import settings

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

class PipelineJob:
    """Job Pipeline avec tracking état"""
    
    def __init__(self, job_id: str, round: int):
        self.job_id = job_id
        self.round = round
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "round": self.round,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }

class JobManager:
    """Gestionnaire jobs pipeline asynchrones"""
    
    def __init__(self):
        self.jobs: Dict[str, PipelineJob] = {}
        self.max_concurrent = settings.MAX_CONCURRENT_JOBS
        self.job_status_file = Path("jobs/pipeline_jobs_status.json")
        self.job_status_file.parent.mkdir(exist_ok=True)
        
        # Charger jobs existants
        self._load_jobs_state()
    
    async def submit_pipeline_job(self, round: int) -> str:
        """
        Soumet job Pipeline avec validation
        
        Recommandations appliquées:
        - Éviter subprocess sync via HTTP
        - Empêcher round < current_round
        - Idempotence si déjà générée
        """
        
        # Validation round
        current_round = self._get_current_round()
        if round < current_round:
            raise ValueError(f"Round {round} déjà joué (current={current_round})")
        
        # Check idempotence
        if self._round_already_generated(round):
            existing_job = self._find_completed_job_for_round(round)
            if existing_job:
                logger.info(f"J{round} déjà générée, retour job existant: {existing_job.job_id}")
                return existing_job.job_id
        
        # Vérifier limite concurrence
        running_jobs = [j for j in self.jobs.values() if j.status == JobStatus.RUNNING]
        if len(running_jobs) >= self.max_concurrent:
            raise ValueError(f"Limite concurrence atteinte: {len(running_jobs)}/{self.max_concurrent}")
        
        # Créer nouveau job
        job_id = str(uuid.uuid4())
        job = PipelineJob(job_id, round)
        self.jobs[job_id] = job
        
        # Lancer job en arrière-plan
        asyncio.create_task(self._execute_pipeline_job(job))
        
        logger.info(f"Job Pipeline J{round} soumis: {job_id}")
        self._save_jobs_state()
        
        return job_id
    
    async def _execute_pipeline_job(self, job: PipelineJob):
        """Exécution job Pipeline en arrière-plan"""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            self._save_jobs_state()
            
            logger.info(f"Démarrage job {job.job_id} pour J{job.round}")
            
            # Commande Pipeline Durci
            cmd = [
                "python3", 
                settings.WEEKLY_AUTOMATION_SCRIPT,
                "--round", str(job.round),
                "--mode", "production"
            ]
            
            # Exécution avec timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=settings.PROJECT_ROOT
            )
            
            job.process = process
            
            # Attendre completion avec timeout (30min max)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=1800  # 30 minutes
                )
                
                return_code = process.returncode
                
                if return_code == 0:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    logger.info(f"Job {job.job_id} complété avec succès")
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Code retour: {return_code}, stderr: {stderr.decode()}"
                    logger.error(f"Job {job.job_id} échoué: {job.error_message}")
                    
            except asyncio.TimeoutError:
                job.status = JobStatus.FAILED
                job.error_message = "Timeout après 30 minutes"
                if process:
                    process.terminate()
                logger.error(f"Job {job.job_id} timeout")
                
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Erreur job {job.job_id}: {e}")
        
        finally:
            job.completed_at = datetime.utcnow()
            self._save_jobs_state()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Récupère status d'un job"""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None
    
    def list_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Liste derniers jobs"""
        sorted_jobs = sorted(
            self.jobs.values(), 
            key=lambda x: x.created_at, 
            reverse=True
        )
        return [job.to_dict() for job in sorted_jobs[:limit]]
    
    def _get_current_round(self) -> int:
        """Détermine journée courante basée sur prédictions existantes"""
        predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
        prediction_files = list(predictions_dir.glob("j*_dual_champions_*.json"))
        
        if not prediction_files:
            return 1
        
        # Parse numéros de journées
        rounds = []
        for file in prediction_files:
            try:
                round_num = int(file.name.split('_')[0][1:])  # j7 -> 7
                rounds.append(round_num)
            except (ValueError, IndexError):
                continue
        
        return max(rounds) if rounds else 1
    
    def _round_already_generated(self, round: int) -> bool:
        """Check si journée déjà générée"""
        predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
        pattern = f"j{round}_epl_2025_26_dual_champions_*.json"
        existing_files = list(predictions_dir.glob(pattern))
        return len(existing_files) > 0
    
    def _find_completed_job_for_round(self, round: int) -> Optional[PipelineJob]:
        """Trouve job complété pour journée donnée"""
        for job in self.jobs.values():
            if job.round == round and job.status == JobStatus.COMPLETED:
                return job
        return None
    
    def _save_jobs_state(self):
        """Sauvegarde état jobs sur disque"""
        try:
            jobs_data = {
                job_id: job.to_dict() 
                for job_id, job in self.jobs.items()
            }
            
            with open(self.job_status_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde jobs: {e}")
    
    def _load_jobs_state(self):
        """Charge état jobs depuis disque"""
        try:
            if not self.job_status_file.exists():
                return
            
            with open(self.job_status_file, 'r') as f:
                jobs_data = json.load(f)
            
            for job_id, job_dict in jobs_data.items():
                job = PipelineJob(job_dict["job_id"], job_dict["round"])
                job.status = JobStatus(job_dict["status"])
                job.created_at = datetime.fromisoformat(job_dict["created_at"])
                
                if job_dict["started_at"]:
                    job.started_at = datetime.fromisoformat(job_dict["started_at"])
                if job_dict["completed_at"]:
                    job.completed_at = datetime.fromisoformat(job_dict["completed_at"])
                if job_dict["error_message"]:
                    job.error_message = job_dict["error_message"]
                
                self.jobs[job_id] = job
                
            logger.info(f"Chargé {len(self.jobs)} jobs depuis disque")
            
        except Exception as e:
            logger.error(f"Erreur chargement jobs: {e}")

# Instance globale
job_manager = JobManager()