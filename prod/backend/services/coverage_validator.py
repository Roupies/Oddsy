#!/usr/bin/env python3
"""
Coverage Validator pour API v5.3
================================

Service de validation stricte de la couverture des fixtures
pour garantir 10/10 fixtures par gameweek EPL avant publication.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

from services.probability_validator import get_probability_validator, ProbabilityValidationError


class CoverageValidationError(Exception):
    """Exception spécialisée pour les erreurs de validation de couverture"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class CoverageValidator:
    """Validateur de couverture stricte pour gameweeks EPL"""
    
    # Teams EPL officielles 2025-26 (exemple)
    EPL_TEAMS = {
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
        "Leeds", "Liverpool", "Manchester City", "Manchester United", 
        "Newcastle", "Nottingham Forest", "Sunderland", "Tottenham",
        "West Ham", "Wolverhampton"
    }
    
    # Alias et variations de noms acceptés
    TEAM_ALIASES = {
        "Man City": "Manchester City",
        "Man Utd": "Manchester United", 
        "Man United": "Manchester United",
        "Spurs": "Tottenham",
        "Nott'm Forest": "Nottingham Forest",
        "Wolves": "Wolverhampton"
    }
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialise le validateur de couverture
        
        Args:
            strict_mode: Si True, rejette toute gameweek <10 fixtures
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger('CoverageValidator')
        self.probability_validator = get_probability_validator()
        
    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normalise le nom d'équipe selon les standards EPL
        
        Args:
            team_name: Nom d'équipe brut
            
        Returns:
            Nom d'équipe normalisé
        """
        # Nettoyer les espaces
        clean_name = team_name.strip()
        
        # Vérifier les alias
        if clean_name in self.TEAM_ALIASES:
            return self.TEAM_ALIASES[clean_name]
        
        # Vérifier si c'est déjà un nom officiel
        if clean_name in self.EPL_TEAMS:
            return clean_name
        
        # Recherche fuzzy pour les variations mineures
        for official_team in self.EPL_TEAMS:
            if clean_name.lower() == official_team.lower():
                return official_team
            
            # Vérifier les noms partiels (ex: "Brighton" pour "Brighton & Hove Albion")
            if clean_name.lower() in official_team.lower() or official_team.lower() in clean_name.lower():
                return official_team
        
        # Retourner le nom original si pas de correspondance
        return clean_name
    
    def _extract_fixtures_from_predictions(self, predictions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extrait les fixtures des données de prédictions
        
        Args:
            predictions_data: Données de prédictions (format API v5)
            
        Returns:
            Liste des fixtures avec métadonnées
        """
        fixtures = []
        
        # Gérer différents formats de données
        if "predictions" in predictions_data:
            predictions = predictions_data["predictions"]
            
            for match_key, prediction_data in predictions.items():
                # Parser la clé de match (format: "Team1_vs_Team2")
                if "_vs_" in match_key:
                    home_team, away_team = match_key.split("_vs_", 1)
                else:
                    # Format alternatif, essayer d'extraire des match_info
                    match_info = prediction_data.get("match_info", {})
                    home_team = match_info.get("home", "Unknown")
                    away_team = match_info.get("away", "Unknown")
                
                # Normaliser les noms d'équipes
                home_normalized = self._normalize_team_name(home_team)
                away_normalized = self._normalize_team_name(away_team)
                
                fixture = {
                    "match_key": match_key,
                    "home_team": home_normalized,
                    "away_team": away_normalized,
                    "home_team_raw": home_team,
                    "away_team_raw": away_team,
                    "prediction": prediction_data.get("prediction"),
                    "confidence": prediction_data.get("confidence"),
                    "probabilities": prediction_data.get("probabilities", {}),
                    "match_info": prediction_data.get("match_info", {}),
                    "date": prediction_data.get("match_info", {}).get("date")
                }
                
                fixtures.append(fixture)
        
        elif "matches" in predictions_data:
            # Format legacy avec liste de matchs
            matches = predictions_data["matches"]
            
            for match in matches:
                home_normalized = self._normalize_team_name(match.get("home_team", "Unknown"))
                away_normalized = self._normalize_team_name(match.get("away_team", "Unknown"))
                
                fixture = {
                    "match_key": f"{home_normalized}_vs_{away_normalized}",
                    "home_team": home_normalized,
                    "away_team": away_normalized,
                    "home_team_raw": match.get("home_team"),
                    "away_team_raw": match.get("away_team"),
                    "prediction": match.get("prediction"),
                    "confidence": match.get("confidence"),
                    "probabilities": match.get("probabilities", {}),
                    "date": match.get("date")
                }
                
                fixtures.append(fixture)
        
        return fixtures
    
    def _validate_epl_teams_only(self, fixtures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valide que seules les équipes EPL sont présentes
        
        Args:
            fixtures: Liste des fixtures
            
        Returns:
            Rapport de validation EPL
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "teams_found": set(),
            "non_epl_teams": set(),
            "team_count": 0
        }
        
        all_teams = set()
        
        for fixture in fixtures:
            home_team = fixture["home_team"]
            away_team = fixture["away_team"]
            
            all_teams.add(home_team)
            all_teams.add(away_team)
            
            # Vérifier que les équipes sont dans la EPL
            if home_team not in self.EPL_TEAMS:
                report["non_epl_teams"].add(home_team)
                report["errors"].append({
                    "type": "non_epl_home_team",
                    "message": f"Home team not in EPL: {home_team}",
                    "fixture": fixture["match_key"],
                    "raw_name": fixture.get("home_team_raw")
                })
                report["valid"] = False
            
            if away_team not in self.EPL_TEAMS:
                report["non_epl_teams"].add(away_team)
                report["errors"].append({
                    "type": "non_epl_away_team", 
                    "message": f"Away team not in EPL: {away_team}",
                    "fixture": fixture["match_key"],
                    "raw_name": fixture.get("away_team_raw")
                })
                report["valid"] = False
        
        report["teams_found"] = all_teams
        report["team_count"] = len(all_teams)
        
        # Vérifier si on a exactement 20 équipes ou un multiple cohérent
        if len(all_teams) > 20:
            report["warnings"].append({
                "type": "too_many_teams",
                "message": f"Found {len(all_teams)} teams, expected max 20 for EPL",
                "team_count": len(all_teams)
            })
        
        return report
    
    def _validate_fixture_count(self, fixtures: List[Dict[str, Any]], gameweek: int) -> Dict[str, Any]:
        """
        Valide le nombre exact de fixtures pour une gameweek
        
        Args:
            fixtures: Liste des fixtures
            gameweek: Numéro de gameweek
            
        Returns:
            Rapport de validation du comptage
        """
        fixture_count = len(fixtures)
        expected_count = 10  # EPL standard: 20 équipes = 10 matchs par gameweek
        
        report = {
            "valid": fixture_count == expected_count,
            "actual_count": fixture_count,
            "expected_count": expected_count,
            "errors": [],
            "warnings": []
        }
        
        if fixture_count < expected_count:
            report["errors"].append({
                "type": "insufficient_fixtures",
                "message": f"Only {fixture_count}/{expected_count} fixtures for GW{gameweek}",
                "missing_fixtures": expected_count - fixture_count
            })
        elif fixture_count > expected_count:
            report["warnings"].append({
                "type": "excess_fixtures",
                "message": f"Found {fixture_count}/{expected_count} fixtures for GW{gameweek}",
                "excess_fixtures": fixture_count - expected_count
            })
        
        return report
    
    def _validate_fixture_uniqueness(self, fixtures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valide l'unicité des fixtures (pas de doublons)
        
        Args:
            fixtures: Liste des fixtures
            
        Returns:
            Rapport de validation unicité
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "unique_fixtures": 0,
            "duplicate_fixtures": []
        }
        
        seen_fixtures = set()
        duplicates = []
        
        for fixture in fixtures:
            # Créer une clé unique normalisée
            teams = sorted([fixture["home_team"], fixture["away_team"]])
            fixture_key = f"{teams[0]}_vs_{teams[1]}"
            
            if fixture_key in seen_fixtures:
                duplicates.append({
                    "fixture_key": fixture_key,
                    "original_key": fixture["match_key"],
                    "home_team": fixture["home_team"],
                    "away_team": fixture["away_team"]
                })
                report["valid"] = False
            else:
                seen_fixtures.add(fixture_key)
        
        report["unique_fixtures"] = len(seen_fixtures)
        report["duplicate_fixtures"] = duplicates
        
        if duplicates:
            report["errors"].append({
                "type": "duplicate_fixtures",
                "message": f"Found {len(duplicates)} duplicate fixtures",
                "duplicates": duplicates
            })
        
        return report
    
    def _validate_team_balance(self, fixtures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valide l'équilibre des équipes (chaque équipe joue exactement 1 fois)
        
        Args:
            fixtures: Liste des fixtures
            
        Returns:
            Rapport de validation équilibre
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "team_appearances": {},
            "teams_playing_multiple": [],
            "teams_not_playing": []
        }
        
        # Compter les apparitions par équipe
        team_count = {}
        for fixture in fixtures:
            home_team = fixture["home_team"]
            away_team = fixture["away_team"]
            
            team_count[home_team] = team_count.get(home_team, 0) + 1
            team_count[away_team] = team_count.get(away_team, 0) + 1
        
        report["team_appearances"] = team_count
        
        # Vérifier que chaque équipe joue exactement 1 fois
        for team, count in team_count.items():
            if count > 1:
                report["teams_playing_multiple"].append({
                    "team": team,
                    "appearances": count
                })
                report["valid"] = False
        
        # Vérifier les équipes qui ne jouent pas
        playing_teams = set(team_count.keys())
        not_playing = self.EPL_TEAMS - playing_teams
        
        if not_playing:
            report["teams_not_playing"] = list(not_playing)
            # Ce n'est qu'un warning car certaines gameweeks peuvent avoir des reports
            report["warnings"].append({
                "type": "teams_not_playing",
                "message": f"{len(not_playing)} teams not playing this gameweek",
                "teams": list(not_playing)
            })
        
        if report["teams_playing_multiple"]:
            report["errors"].append({
                "type": "teams_playing_multiple_times",
                "message": f"{len(report['teams_playing_multiple'])} teams playing multiple times",
                "teams": report["teams_playing_multiple"]
            })
        
        return report
    
    def _validate_predictions_quality(self, fixtures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valide la qualité des prédictions (probabilités, confiance)
        
        Args:
            fixtures: Liste des fixtures
            
        Returns:
            Rapport de validation qualité
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "fixtures_validated": 0,
            "probability_issues": [],
            "confidence_issues": []
        }
        
        for fixture in fixtures:
            fixture_key = fixture["match_key"]
            
            # Valider les probabilités si présentes
            probabilities = fixture.get("probabilities", {})
            if probabilities:
                try:
                    validation_result = self.probability_validator.validate_probability_constraints(probabilities)
                    
                    if not validation_result["valid"]:
                        report["probability_issues"].append({
                            "fixture": fixture_key,
                            "errors": validation_result["errors"],
                            "probabilities": probabilities
                        })
                        report["valid"] = False
                    
                    # Ajouter les warnings
                    if validation_result["warnings"]:
                        report["warnings"].extend([
                            {**warning, "fixture": fixture_key} 
                            for warning in validation_result["warnings"]
                        ])
                
                except ProbabilityValidationError as e:
                    report["probability_issues"].append({
                        "fixture": fixture_key,
                        "error": str(e),
                        "probabilities": probabilities
                    })
                    report["valid"] = False
            
            # Valider la confiance
            confidence = fixture.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    report["confidence_issues"].append({
                        "fixture": fixture_key,
                        "issue": "confidence_not_numeric",
                        "value": confidence
                    })
                    report["valid"] = False
                elif not (0 <= confidence <= 1):
                    report["confidence_issues"].append({
                        "fixture": fixture_key,
                        "issue": "confidence_out_of_range",
                        "value": confidence
                    })
                    report["valid"] = False
            
            report["fixtures_validated"] += 1
        
        if report["probability_issues"]:
            report["errors"].append({
                "type": "probability_validation_failed",
                "message": f"{len(report['probability_issues'])} fixtures have probability issues",
                "fixtures_affected": len(report["probability_issues"])
            })
        
        if report["confidence_issues"]:
            report["errors"].append({
                "type": "confidence_validation_failed", 
                "message": f"{len(report['confidence_issues'])} fixtures have confidence issues",
                "fixtures_affected": len(report["confidence_issues"])
            })
        
        return report
    
    def validate_gameweek_coverage(self, 
                                 predictions_data: Dict[str, Any], 
                                 gameweek: int,
                                 allow_partial: bool = False) -> Dict[str, Any]:
        """
        Validation complète de la couverture d'une gameweek
        
        Args:
            predictions_data: Données de prédictions
            gameweek: Numéro de gameweek
            allow_partial: Si True, autorise <10 fixtures (mode développement)
            
        Returns:
            Rapport de validation complet
            
        Raises:
            CoverageValidationError: Si validation stricte échoue
        """
        validation_report = {
            "gameweek": gameweek,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_valid": True,
            "ready_for_production": False,
            "validation_mode": "strict" if self.strict_mode else "permissive",
            "allow_partial": allow_partial,
            "fixtures": [],
            "summary": {},
            "validations": {}
        }
        
        try:
            # Extraire les fixtures
            fixtures = self._extract_fixtures_from_predictions(predictions_data)
            validation_report["fixtures"] = fixtures
            
            # 1. Validation du nombre de fixtures
            fixture_count_report = self._validate_fixture_count(fixtures, gameweek)
            validation_report["validations"]["fixture_count"] = fixture_count_report
            
            if not fixture_count_report["valid"] and not allow_partial:
                validation_report["overall_valid"] = False
            
            # 2. Validation équipes EPL uniquement
            epl_teams_report = self._validate_epl_teams_only(fixtures)
            validation_report["validations"]["epl_teams"] = epl_teams_report
            
            if not epl_teams_report["valid"]:
                validation_report["overall_valid"] = False
            
            # 3. Validation unicité des fixtures
            uniqueness_report = self._validate_fixture_uniqueness(fixtures)
            validation_report["validations"]["uniqueness"] = uniqueness_report
            
            if not uniqueness_report["valid"]:
                validation_report["overall_valid"] = False
            
            # 4. Validation équilibre des équipes
            balance_report = self._validate_team_balance(fixtures)
            validation_report["validations"]["team_balance"] = balance_report
            
            if not balance_report["valid"]:
                validation_report["overall_valid"] = False
            
            # 5. Validation qualité des prédictions
            quality_report = self._validate_predictions_quality(fixtures)
            validation_report["validations"]["prediction_quality"] = quality_report
            
            if not quality_report["valid"]:
                validation_report["overall_valid"] = False
            
            # Générer le résumé
            validation_report["summary"] = {
                "fixtures_count": len(fixtures),
                "expected_fixtures": 10,
                "epl_teams_only": epl_teams_report["valid"],
                "unique_fixtures": uniqueness_report["valid"],
                "balanced_teams": balance_report["valid"],
                "quality_predictions": quality_report["valid"],
                "total_validations": 5,
                "passed_validations": sum([
                    fixture_count_report["valid"] or allow_partial,
                    epl_teams_report["valid"],
                    uniqueness_report["valid"], 
                    balance_report["valid"],
                    quality_report["valid"]
                ])
            }
            
            # Déterminer si prêt pour production
            validation_report["ready_for_production"] = (
                validation_report["overall_valid"] and
                len(fixtures) == 10 and
                all([
                    epl_teams_report["valid"],
                    uniqueness_report["valid"],
                    balance_report["valid"],
                    quality_report["valid"]
                ])
            )
            
            # Si mode strict et validation échoue, lever exception
            if self.strict_mode and not validation_report["ready_for_production"]:
                failed_validations = []
                for validation_name, validation_result in validation_report["validations"].items():
                    if not validation_result["valid"]:
                        failed_validations.append(validation_name)
                
                raise CoverageValidationError(
                    f"Gameweek {gameweek} failed strict coverage validation",
                    {
                        "failed_validations": failed_validations,
                        "fixtures_count": len(fixtures),
                        "validation_report": validation_report
                    }
                )
            
            self.logger.info(
                f"Coverage validation GW{gameweek}: "
                f"{validation_report['summary']['passed_validations']}/5 checks passed, "
                f"{len(fixtures)} fixtures, ready_for_production={validation_report['ready_for_production']}"
            )
            
            return validation_report
            
        except Exception as e:
            if isinstance(e, CoverageValidationError):
                raise
            else:
                raise CoverageValidationError(
                    f"Coverage validation failed for GW{gameweek}: {str(e)}",
                    {"gameweek": gameweek, "original_error": str(e)}
                )


# Instance globale du validateur de couverture
_coverage_validator: Optional[CoverageValidator] = None

def get_coverage_validator() -> CoverageValidator:
    """Récupère l'instance singleton du validateur de couverture"""
    global _coverage_validator
    if _coverage_validator is None:
        _coverage_validator = CoverageValidator()
    return _coverage_validator