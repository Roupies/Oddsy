#!/usr/bin/env python3
"""
Schémas Pydantic pour Pipeline Status
====================================
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ComponentHealth(str, Enum):
    HEALTHY = "healthy"
    ERROR = "error"
    UNKNOWN = "unknown"

class PipelineStatus(BaseModel):
    """Status complet Pipeline Durci"""
    pipeline_version: str = Field(description="Version Pipeline Durci")
    last_run: Optional[datetime] = Field(None, description="Dernière exécution")
    components_status: Dict[str, ComponentHealth] = Field(description="Status composants")
    next_scheduled_run: Optional[datetime] = Field(None, description="Prochaine exécution")
    data_freshness: Dict[str, Any] = Field(description="Fraîcheur des données")

class JobStatusResponse(BaseModel):
    """Réponse status job"""
    job_id: str
    round: int
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None