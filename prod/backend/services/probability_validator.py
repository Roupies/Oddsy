#!/usr/bin/env python3
"""
Probability Validator avec normalisation Clarke/Power
===================================================

Service de validation et normalisation des probabilités pour garantir
la qualité des données exposées en production API v5.3.
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class ProbabilityValidationError(Exception):
    """Exception spécialisée pour les erreurs de validation de probabilités"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ProbabilityValidator:
    """Validateur et normalisateur de probabilités pour paris sportifs"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialise le validateur
        
        Args:
            tolerance: Tolérance pour la somme des probabilités (défaut: 1%)
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger('ProbabilityValidator')
        
    def _calculate_overround(self, probabilities: Dict[str, float]) -> float:
        """
        Calcule l'overround (marge bookmaker) des probabilités
        
        Args:
            probabilities: Dict avec home/draw/away
            
        Returns:
            Overround en pourcentage (ex: 5.0 pour 5%)
        """
        total = sum(probabilities.values())
        if total <= 0:
            return 0.0
        
        # Overround = (somme des probabilités - 1) * 100
        return (total - 1.0) * 100
    
    def _clarke_normalization(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Normalisation Clarke: division simple par la somme
        
        Args:
            probabilities: Probabilités brutes
            
        Returns:
            Probabilités normalisées
        """
        total = sum(probabilities.values())
        if total <= 0:
            raise ProbabilityValidationError(
                "Cannot normalize: sum of probabilities is zero or negative",
                {"original_probabilities": probabilities, "sum": total}
            )
        
        return {
            outcome: prob / total 
            for outcome, prob in probabilities.items()
        }
    
    def _power_normalization(self, probabilities: Dict[str, float], power: float = 1.1) -> Dict[str, float]:
        """
        Normalisation Power: ajuste les probabilités par une puissance puis normalise
        Réduit la marge bookmaker de manière plus équitable
        
        Args:
            probabilities: Probabilités brutes
            power: Exposant de correction (>1 réduit l'overround)
            
        Returns:
            Probabilités normalisées
        """
        if any(p <= 0 for p in probabilities.values()):
            raise ProbabilityValidationError(
                "Power normalization requires all probabilities > 0",
                {"probabilities": probabilities}
            )
        
        # Appliquer la puissance
        powered_probs = {
            outcome: prob ** power 
            for outcome, prob in probabilities.items()
        }
        
        # Normaliser
        total = sum(powered_probs.values())
        return {
            outcome: prob / total 
            for outcome, prob in powered_probs.items()
        }
    
    def _multiplicative_normalization(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Normalisation multiplicative: conserve les ratios relatifs
        
        Args:
            probabilities: Probabilités brutes
            
        Returns:
            Probabilités normalisées
        """
        total = sum(probabilities.values())
        if total <= 0:
            raise ProbabilityValidationError(
                "Cannot normalize: sum of probabilities is zero or negative",
                {"probabilities": probabilities, "sum": total}
            )
        
        # Facteur de normalisation
        factor = 1.0 / total
        
        return {
            outcome: prob * factor 
            for outcome, prob in probabilities.items()
        }
    
    def normalize_probabilities(self, 
                              probabilities: Dict[str, float], 
                              method: str = "power") -> Dict[str, float]:
        """
        Normalise les probabilités selon la méthode choisie
        
        Args:
            probabilities: Probabilités brutes {home, draw, away}
            method: Méthode ("clarke", "power", "multiplicative")
            
        Returns:
            Probabilités normalisées
            
        Raises:
            ProbabilityValidationError: Si normalisation impossible
        """
        
        # Validation des clés requises
        required_keys = {"home", "draw", "away"}
        if not required_keys.issubset(probabilities.keys()):
            raise ProbabilityValidationError(
                f"Missing required probability keys: {required_keys - probabilities.keys()}",
                {"probabilities": probabilities, "required_keys": list(required_keys)}
            )
        
        # Validation des valeurs
        for outcome, prob in probabilities.items():
            if not isinstance(prob, (int, float)):
                raise ProbabilityValidationError(
                    f"Probability for {outcome} must be numeric, got {type(prob)}",
                    {"outcome": outcome, "value": prob, "type": type(prob).__name__}
                )
            
            if prob < 0:
                raise ProbabilityValidationError(
                    f"Probability for {outcome} cannot be negative: {prob}",
                    {"outcome": outcome, "value": prob}
                )
        
        # Calculer l'overround original
        original_overround = self._calculate_overround(probabilities)
        
        # Appliquer la normalisation selon la méthode
        try:
            if method == "clarke":
                normalized = self._clarke_normalization(probabilities)
            elif method == "power":
                normalized = self._power_normalization(probabilities)
            elif method == "multiplicative":
                normalized = self._multiplicative_normalization(probabilities)
            else:
                raise ProbabilityValidationError(
                    f"Unknown normalization method: {method}",
                    {"method": method, "available_methods": ["clarke", "power", "multiplicative"]}
                )
            
            # Vérification finale
            final_sum = sum(normalized.values())
            if abs(final_sum - 1.0) > self.tolerance:
                raise ProbabilityValidationError(
                    f"Normalization failed: sum = {final_sum:.6f}, tolerance = {self.tolerance}",
                    {
                        "method": method,
                        "final_sum": final_sum,
                        "tolerance": self.tolerance,
                        "normalized_probabilities": normalized
                    }
                )
            
            # Log de succès avec métriques
            self.logger.debug(
                f"Probabilities normalized successfully: {method}, "
                f"original_overround={original_overround:.2f}%, final_sum={final_sum:.6f}"
            )
            
            return normalized
            
        except Exception as e:
            if isinstance(e, ProbabilityValidationError):
                raise
            else:
                raise ProbabilityValidationError(
                    f"Normalization failed with {method} method: {str(e)}",
                    {"method": method, "original_error": str(e), "probabilities": probabilities}
                )
    
    def validate_probability_constraints(self, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Valide les contraintes métier des probabilités
        
        Args:
            probabilities: Probabilités à valider
            
        Returns:
            Rapport de validation avec métriques
        """
        
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Métriques de base
        total_sum = sum(probabilities.values())
        min_prob = min(probabilities.values())
        max_prob = max(probabilities.values())
        
        validation_report["metrics"] = {
            "sum": round(total_sum, 6),
            "min_probability": round(min_prob, 6),
            "max_probability": round(max_prob, 6),
            "overround_percent": round(self._calculate_overround(probabilities), 2),
            "entropy": self._calculate_entropy(probabilities)
        }
        
        # Validation somme = 1 ± tolérance
        if abs(total_sum - 1.0) > self.tolerance:
            validation_report["valid"] = False
            validation_report["errors"].append({
                "type": "sum_constraint_violation",
                "message": f"Probability sum {total_sum:.6f} outside tolerance ±{self.tolerance}",
                "actual_sum": total_sum,
                "tolerance": self.tolerance
            })
        
        # Validation bornes [0, 1]
        for outcome, prob in probabilities.items():
            if prob < 0 or prob > 1:
                validation_report["valid"] = False
                validation_report["errors"].append({
                    "type": "range_constraint_violation",
                    "message": f"Probability for {outcome} outside [0,1]: {prob}",
                    "outcome": outcome,
                    "value": prob
                })
        
        # Validation probabilités minimum (éviter 0 exact)
        min_threshold = 0.001  # 0.1%
        for outcome, prob in probabilities.items():
            if 0 < prob < min_threshold:
                validation_report["warnings"].append({
                    "type": "very_low_probability",
                    "message": f"Very low probability for {outcome}: {prob:.6f}",
                    "outcome": outcome,
                    "value": prob,
                    "threshold": min_threshold
                })
        
        # Validation dominance excessive (éviter >95%)
        max_threshold = 0.95
        for outcome, prob in probabilities.items():
            if prob > max_threshold:
                validation_report["warnings"].append({
                    "type": "very_high_probability", 
                    "message": f"Very high probability for {outcome}: {prob:.6f}",
                    "outcome": outcome,
                    "value": prob,
                    "threshold": max_threshold
                })
        
        # Validation overround raisonnable (< 20%)
        overround = self._calculate_overround(probabilities)
        if overround > 20.0:
            validation_report["warnings"].append({
                "type": "high_overround",
                "message": f"High bookmaker margin: {overround:.2f}%",
                "overround_percent": overround
            })
        
        return validation_report
    
    def _calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """
        Calcule l'entropie de Shannon des probabilités
        Mesure l'incertitude/prédictibilité du match
        
        Args:
            probabilities: Probabilités normalisées
            
        Returns:
            Entropie en bits (0 = certain, log2(3)≈1.58 = maximum incertitude)
        """
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return round(entropy, 4)
    
    def process_prediction_probabilities(self, 
                                       prediction_data: Dict[str, Any],
                                       normalize: bool = True,
                                       method: str = "power") -> Dict[str, Any]:
        """
        Traite et valide les probabilités d'une prédiction complète
        
        Args:
            prediction_data: Données de prédiction avec probabilities
            normalize: Si True, normalise les probabilités
            method: Méthode de normalisation
            
        Returns:
            Données de prédiction avec probabilités validées/normalisées
            
        Raises:
            ProbabilityValidationError: Si validation échoue
        """
        
        # Vérifier structure des données
        if "probabilities" not in prediction_data:
            raise ProbabilityValidationError(
                "Missing 'probabilities' field in prediction data",
                {"available_fields": list(prediction_data.keys())}
            )
        
        probabilities = prediction_data["probabilities"]
        
        # Normaliser si demandé
        if normalize:
            probabilities = self.normalize_probabilities(probabilities, method)
        
        # Valider les contraintes
        validation_report = self.validate_probability_constraints(probabilities)
        
        if not validation_report["valid"]:
            raise ProbabilityValidationError(
                "Probability validation failed",
                {
                    "validation_report": validation_report,
                    "original_probabilities": prediction_data["probabilities"]
                }
            )
        
        # Mettre à jour les données de prédiction
        updated_data = prediction_data.copy()
        updated_data["probabilities"] = probabilities
        
        # Mettre à jour la confiance basée sur la probabilité max
        updated_data["confidence"] = max(probabilities.values())
        
        # Ajouter les métriques de validation
        updated_data["probability_validation"] = {
            "normalized": normalize,
            "method": method if normalize else None,
            "metrics": validation_report["metrics"],
            "warnings": validation_report["warnings"]
        }
        
        return updated_data


# Instance globale du validateur
_probability_validator: Optional[ProbabilityValidator] = None

def get_probability_validator() -> ProbabilityValidator:
    """Récupère l'instance singleton du validateur de probabilités"""
    global _probability_validator
    if _probability_validator is None:
        _probability_validator = ProbabilityValidator()
    return _probability_validator