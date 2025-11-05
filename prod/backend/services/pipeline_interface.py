#!/usr/bin/env python3
"""
Interface Pipeline Durci avec validation et fallbacks
====================================================
"""

import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from core.config import settings
from schemas.predictions import RoundPredictions, EnsembleSystem, MatchPrediction, ModelPrediction, PredictionProbabilities
from schemas.pipeline import PipelineStatus, ComponentHealth

logger = logging.getLogger(__name__)

class PipelineDurciInterface:
    """Interface officielle vers Pipeline Durci v1.0 avec validation stricte"""
    
    def __init__(self):
        self.predictions_dir = settings.PIPELINE_PREDICTIONS_DIR
        self.reports_dir = settings.PIPELINE_REPORTS_DIR
        
    def read_latest_predictions(self, round: int) -> Optional[RoundPredictions]:
        """
        Lit dernières prédictions avec validation et fallback
        
        Recommandations appliquées:
        - Glob + tri par mtime
        - Validation schéma avant retour  
        - Fallback sur fichier précédent si corruption
        """
        try:
            # Pattern pour la journée demandée
            pattern = f"j{round}_epl_2025_26_dual_champions_*.json"
            prediction_files = list(self.predictions_dir.glob(pattern))
            
            if not prediction_files:
                logger.warning(f"Aucun fichier de prédictions trouvé pour J{round}")
                return None
            
            # Tri par modification time (plus récent en premier)
            prediction_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Tentative de lecture du plus récent
            for file_path in prediction_files:
                try:
                    logger.info(f"Lecture prédictions: {file_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    
                    # Validation structure de base
                    if not self._validate_prediction_structure(raw_data):
                        logger.warning(f"Structure invalide: {file_path}")
                        continue
                    
                    # Conversion vers schéma Pydantic
                    validated_predictions = self._convert_to_schema(raw_data)
                    
                    logger.info(f"✅ Prédictions J{round} chargées: {len(validated_predictions.matches)} matchs")
                    return validated_predictions
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Erreur lecture {file_path}: {e}")
                    continue  # Essayer fichier suivant
            
            logger.error(f"Aucun fichier valide trouvé pour J{round}")
            return None
            
        except Exception as e:
            logger.error(f"Erreur interface pipeline J{round}: {e}")
            return None
    
    def _validate_prediction_structure(self, data: Dict[str, Any]) -> bool:
        """Validation structure minimale avant conversion Pydantic"""
        required_keys = ["prediction_metadata", "dual_champion_system", "predictions"]
        
        for key in required_keys:
            if key not in data:
                logger.error(f"Clé manquante: {key}")
                return False
        
        # Vérification métadonnées
        meta = data["prediction_metadata"]
        if not all(k in meta for k in ["round", "season", "total_matches"]):
            logger.error("Métadonnées incomplètes")
            return False
        
        # Vérification nombre de matchs
        expected_matches = meta.get("total_matches", 0)
        actual_matches = len(data.get("predictions", []))
        
        if expected_matches != actual_matches:
            logger.error(f"Nombre matchs incohérent: attendu={expected_matches}, actuel={actual_matches}")
            return False
        
        return True
    
    def _convert_to_schema(self, raw_data: Dict[str, Any]) -> RoundPredictions:
        """Conversion données pipeline vers schéma API validé"""
        meta = raw_data["prediction_metadata"]
        ensemble_data = raw_data["dual_champion_system"]
        
        # Conversion matches avec validation probabilités
        matches = []
        for prediction in raw_data["predictions"]:
            match_info = prediction["match_info"]
            ensemble_pred = prediction["ensemble_prediction"]
            individual_models = prediction["individual_models"]
            
            # Sanity check probabilités ensemble
            ens_probs = ensemble_pred["probabilities"]
            total_prob = ens_probs["home"] + ens_probs["draw"] + ens_probs["away"]
            
            if not (0.98 <= total_prob <= 1.02):
                logger.warning(f"Probabilités suspectes pour {match_info['home_team']} vs {match_info['away_team']}: {total_prob:.3f}")
            
            # Calcul désaccord modèles (Enhanced vs Cascade)
            disagreement = self._calculate_disagreement(individual_models)
            
            match_pred = MatchPrediction(
                id=f"j{meta['round']}_{match_info['home_team'].lower().replace(' ', '_')}_{match_info['away_team'].lower().replace(' ', '_')}",
                home_team=match_info["home_team"],
                away_team=match_info["away_team"], 
                date=match_info["date"],
                round=meta["round"],
                ensemble=ModelPrediction(
                    prediction=ensemble_pred["predicted_outcome"],
                    confidence=ensemble_pred["confidence"],
                    probabilities=PredictionProbabilities(**ens_probs)
                ),
                models={
                    model_name: ModelPrediction(
                        prediction=model_data["prediction"],
                        confidence=model_data["confidence"],
                        probabilities=PredictionProbabilities(**model_data["probabilities"])
                    )
                    for model_name, model_data in individual_models.items()
                },
                disagreement=disagreement
            )
            
            matches.append(match_pred)
        
        # Construction ensemble system
        ensemble_system = EnsembleSystem(
            system_name=ensemble_data["system_name"],
            version=ensemble_data["version"],
            models=ensemble_data["models"],
            ensemble_strategy=ensemble_data["ensemble_strategy"],
            weights={
                "enhanced": ensemble_data["models"]["enhanced_baseline_v24"]["weight"],
                "cascade": ensemble_data["models"]["cascade_v21_optimized"]["weight"]
            },
            expected_performance=ensemble_data["validation_performance"]["expected_ensemble"]
        )
        
        return RoundPredictions(
            round=meta["round"],
            season=meta["season"],
            competition=meta["competition"],
            total_matches=meta["total_matches"],
            ensemble_system=ensemble_system,
            matches=matches
        )
    
    def _calculate_disagreement(self, models: Dict[str, Any]) -> float:
        """Calcule niveau de désaccord entre modèles (|p1_H - p2_H|)"""
        if len(models) < 2:
            return 0.0
        
        model_names = list(models.keys())
        p1_home = models[model_names[0]]["probabilities"]["home"]
        p2_home = models[model_names[1]]["probabilities"]["home"]
        
        return abs(p1_home - p2_home)
    
    def get_pipeline_status(self) -> PipelineStatus:
        """
        Status Pipeline avec parsing des rapports réels
        
        Recommandations appliquées:
        - Parser rapports JSON existants  
        - Ne pas inventer d'états
        - Sanity checks sur composants
        """
        try:
            # Dernière exécution automation
            last_run = self._get_last_automation_run()
            
            # Status composants depuis rapports
            components_status = {
                "understat_extractor": self._check_component_health("understat"),
                "temporal_calculator": self._check_component_health("temporal"), 
                "dual_champions": self._check_component_health("ensemble")
            }
            
            # Prochaine exécution planifiée
            next_run = self._get_next_scheduled_run()
            
            return PipelineStatus(
                pipeline_version="Pipeline_Durci_v1.0",
                last_run=last_run,
                components_status=components_status,
                next_scheduled_run=next_run,
                data_freshness=self._check_data_freshness()
            )
            
        except Exception as e:
            logger.error(f"Erreur status pipeline: {e}")
            return PipelineStatus(
                pipeline_version="Pipeline_Durci_v1.0",
                last_run=None,
                components_status={
                    "understat_extractor": ComponentHealth.ERROR,
                    "temporal_calculator": ComponentHealth.ERROR,
                    "dual_champions": ComponentHealth.ERROR
                },
                next_scheduled_run=None,
                data_freshness={}
            )
    
    def _get_last_automation_run(self) -> Optional[datetime]:
        """Parse dernier rapport d'automation pour timestamp"""
        try:
            # Chercher derniers rapports automation
            automation_reports = list(self.reports_dir.glob("weekly_automation_*.json"))
            
            if not automation_reports:
                return None
            
            # Plus récent par nom (timestamp dans nom)
            latest_report = max(automation_reports, key=lambda x: x.name)
            
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            timestamp_str = report_data.get("timestamp")
            if timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lecture dernière automation: {e}")
            return None
    
    def _check_component_health(self, component: str) -> ComponentHealth:
        """Check santé composant depuis rapports existants"""
        try:
            if component == "temporal":
                # Check rapport temporal
                temporal_report = self.reports_dir / "strict_temporal_report.json"
                if temporal_report.exists():
                    with open(temporal_report, 'r') as f:
                        data = json.load(f)
                    
                    if data.get("status") == "✅ SUCCESS":
                        return ComponentHealth.HEALTHY
                    else:
                        return ComponentHealth.ERROR
            
            elif component == "ensemble":
                # Check dernières prédictions générées
                recent_predictions = list(self.predictions_dir.glob("j*_dual_champions_*.json"))
                if recent_predictions:
                    return ComponentHealth.HEALTHY
                else:
                    return ComponentHealth.ERROR
            
            # Default fallback
            return ComponentHealth.UNKNOWN
            
        except Exception:
            return ComponentHealth.ERROR
    
    def _get_next_scheduled_run(self) -> Optional[datetime]:
        """Parse prochaine exécution depuis automation"""
        try:
            automation_reports = list(self.reports_dir.glob("weekly_automation_*.json"))
            if not automation_reports:
                return None
                
            latest_report = max(automation_reports, key=lambda x: x.name)
            
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            next_exec_str = report_data.get("next_execution")
            if next_exec_str:
                return datetime.fromisoformat(next_exec_str.replace('Z', '+00:00'))
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur next scheduled run: {e}")
            return None
    
    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check fraîcheur des données sources"""
        return {
            "understat_last_update": self._check_understat_freshness(),
            "e0_last_download": self._check_e0_freshness(), 
            "predictions_last_generation": self._check_predictions_freshness()
        }
    
    def _check_understat_freshness(self) -> Optional[str]:
        """Check dernière maj Understat depuis rapports"""
        # Implémenter selon structure des rapports
        return None
    
    def _check_e0_freshness(self) -> Optional[str]:
        """Check dernier download E0"""
        # Implémenter selon structure des rapports  
        return None
    
    def _check_predictions_freshness(self) -> Optional[str]:
        """Check dernière génération prédictions"""
        try:
            prediction_files = list(self.predictions_dir.glob("j*_dual_champions_*.json"))
            if not prediction_files:
                return None
            
            latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
            return datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
            
        except Exception:
            return None