#!/usr/bin/env python3
"""
Exceptions personnalisées Backend FastAPI
========================================
"""

from datetime import datetime
from typing import Optional

class PipelineError(Exception):
    """Erreur Pipeline Durci"""
    pass

class ValidationError(Exception):
    """Erreur validation données"""
    pass

class JobError(Exception):
    """Erreur job asynchrone"""
    pass

class MissingCriticalSource(Exception):
    """Exception pour sources critiques indisponibles
    
    Levée quand une source de données critique (xG Understat, cotes, etc.)
    n'est pas disponible après retries. Force l'arrêt du pipeline au lieu
    d'utiliser des fallbacks silencieux qui contaminent les prédictions.
    """
    
    def __init__(self, source_name: str, details: str = ""):
        self.source_name = source_name
        self.details = details
        self.timestamp = datetime.utcnow()
        
        message = f"CRITICAL SOURCE MISSING: {source_name}"
        if details:
            message += f" - {details}"
            
        super().__init__(message)
    
    def to_dict(self) -> dict:
        """Conversion en dict pour logging structuré"""
        return {
            "error_type": "MissingCriticalSource",
            "source": self.source_name,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "message": str(self)
        }