#!/usr/bin/env python3
"""
Schémas Pydantic communs avec génériques corrigés
=================================================
"""

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from typing import Generic, TypeVar, Optional
from datetime import datetime

T = TypeVar('T')

class APIMetadata(BaseModel):
    """Métadonnées communes à toutes les réponses API"""
    api_version: str = Field(default="1.0", description="Version API")
    pipeline_version: str = Field(description="Version Pipeline Durci")
    generated_at: datetime = Field(description="Timestamp génération données")
    git_sha: Optional[str] = Field(None, description="Version code déployée")

class APIResponse(GenericModel, Generic[T]):
    """Wrapper générique formel pour toutes les réponses API"""
    meta: APIMetadata
    data: T
    error: Optional[str] = None