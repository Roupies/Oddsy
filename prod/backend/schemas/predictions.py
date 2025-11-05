#!/usr/bin/env python3
"""
Schémas Pydantic pour prédictions avec validation corrigée
=========================================================
"""

from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import List, Dict, Any

class PredictionOutcome(str, Enum):
    HOME = "H"
    DRAW = "D" 
    AWAY = "A"

class PredictionProbabilities(BaseModel):
    """Probabilités normalisées et validées"""
    home: float = Field(ge=0.0, le=1.0, description="Probabilité victoire domicile")
    draw: float = Field(ge=0.0, le=1.0, description="Probabilité match nul") 
    away: float = Field(ge=0.0, le=1.0, description="Probabilité victoire extérieur")
    
    @model_validator(mode="after")
    def validate_probabilities_sum(self):
        """Validation somme probabilités ≈ 1.0"""
        total = self.home + self.draw + self.away
        tolerance = 0.01  # ±1%
        
        if not (1.0 - tolerance <= total <= 1.0 + tolerance):
            raise ValueError(f"Probabilités invalides: somme={total:.3f}, attendu≈1.0±{tolerance}")
        
        return self

class ModelPrediction(BaseModel):
    """Prédiction d'un modèle individuel"""
    prediction: PredictionOutcome
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: PredictionProbabilities
    
class EnsembleSystem(BaseModel):
    """Configuration système ensemble"""
    system_name: str = Field(description="Nom système ensemble")
    version: str = Field(description="Version ensemble")
    models: Dict[str, Dict[str, Any]] = Field(description="Config modèles")
    ensemble_strategy: str = Field(description="Stratégie agrégation")
    weights: Dict[str, float] = Field(description="Poids modèles")
    expected_performance: float = Field(ge=0.0, le=1.0, description="Performance attendue")

class MatchPrediction(BaseModel):
    """Prédiction complète d'un match"""
    id: str = Field(description="ID unique match")
    home_team: str
    away_team: str
    date: str = Field(description="Date match DD/MM/YYYY")
    round: int = Field(ge=1, le=38, description="Journée EPL")
    
    # Prédiction ensemble
    ensemble: ModelPrediction
    
    # Prédictions individuelles
    models: Dict[str, ModelPrediction] = Field(description="Prédictions par modèle")
    
    # Métriques additionnelles
    disagreement: float = Field(ge=0.0, le=1.0, description="Niveau désaccord modèles")
    
class RoundPredictions(BaseModel):
    """Prédictions complètes d'une journée"""
    round: int = Field(ge=1, le=38)
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    competition: str = Field(default="Premier League")
    total_matches: int = Field(ge=1, le=10, description="Nombre matchs")
    
    ensemble_system: EnsembleSystem
    matches: List[MatchPrediction] = Field(min_items=1, max_items=10)