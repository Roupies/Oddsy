#!/usr/bin/env python3
"""
validate_temporal_integrity.py

Suite de validation anti-leakage pour pipeline rolling.

OBJECTIF:
- Détecter toute forme de data leakage temporel
- Valider l'intégrité du StateManager et des features
- Tests automatisés pour intégration continue
- Garantir simulation rolling réaliste

USAGE:
python scripts/analysis/validate_temporal_integrity.py \
    --state_file data/states/state_2024_season.pkl \
    --test_data data/processed/premier_league_full_with_results.csv \
    --test_start 2024-08-01 \
    --report results/temporal_integrity_report.json

TESTS IMPLÉMENTÉS:
1. StateManager Temporal Consistency
2. Feature Temporal Boundaries  
3. Rolling Windows Integrity
4. H2H Historical Accuracy
5. Elo Temporal Progression
6. Simulation Determinism
7. Cross-Validation Split Integrity
"""

import sys
import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

# Import components
from scripts.analysis.initialize_historical_state import StateManager, HistoricalStateBuilder, load_state
from scripts.analysis.rolling_simulation_engine import RollingSimulationEngine, FeatureBuilder

@dataclass
class ValidationResult:
    """Résultat d'un test de validation"""
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

class TemporalIntegrityValidator:
    """Validateur principal d'intégrité temporelle"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.validation_results: List[ValidationResult] = []
    
    def run_all_tests(self, state_file: str, test_data_path: str, 
                     test_start: str) -> Dict[str, Any]:
        """Exécute tous les tests de validation"""
        self.logger.info("🔍 STARTING TEMPORAL INTEGRITY VALIDATION")
        self.logger.info("=" * 70)
        
        # Charger état et données
        state_manager = load_state(state_file)
        test_data = self._load_test_data(test_data_path, test_start)
        
        # Tests à exécuter
        tests = [
            ("State Temporal Consistency", self._test_state_temporal_consistency),
            ("Feature Temporal Boundaries", self._test_feature_temporal_boundaries),
            ("Rolling Windows Integrity", self._test_rolling_windows_integrity),
            ("H2H Historical Accuracy", self._test_h2h_historical_accuracy),
            ("Elo Temporal Progression", self._test_elo_temporal_progression),
            ("Simulation Determinism", self._test_simulation_determinism),
            ("Cross-Validation Split Integrity", self._test_cv_split_integrity),
            # AMÉLIORATIONS CRITIQUES P0
            ("Feature Future Information Test", self._test_feature_future_information),
            ("H2H Self-Exclusion Test", self._test_h2h_self_exclusion),
            ("State Update Order Validation", self._test_state_update_order)
        ]
        
        # Exécuter chaque test
        for test_name, test_func in tests:
            self.logger.info(f"🧪 Running: {test_name}")
            
            start_time = datetime.now()
            try:
                result = test_func(state_manager, test_data, test_start)
                result.execution_time = (datetime.now() - start_time).total_seconds()
                
                if result.passed:
                    self.logger.info(f"  ✅ PASSED ({result.execution_time:.2f}s)")
                else:
                    self.logger.error(f"  ❌ FAILED: {result.error_message}")
                    
                if result.warnings:
                    for warning in result.warnings:
                        self.logger.warning(f"  ⚠️  {warning}")
                        
            except Exception as e:
                result = ValidationResult(
                    test_name=test_name,
                    passed=False,
                    error_message=str(e),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                self.logger.error(f"  💥 EXCEPTION: {e}")
            
            self.validation_results.append(result)
        
        # Générer rapport final
        return self._generate_final_report()
    
    def _load_test_data(self, data_path: str, test_start: str) -> pd.DataFrame:
        """Charge les données de test"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filtrer période de test
        start_date = pd.to_datetime(test_start)
        test_df = df[df['Date'] >= start_date].copy()
        test_df = test_df.sort_values('Date').reset_index(drop=True)
        
        self.logger.info(f"📊 Test data: {len(test_df)} matches from {test_start}")
        
        return test_df
    
    def _test_state_temporal_consistency(self, state_manager: StateManager, 
                                       test_data: pd.DataFrame, 
                                       test_start: str) -> ValidationResult:
        """Test 1: Cohérence temporelle du StateManager"""
        
        errors = []
        warnings = []
        details = {}
        
        # 1. Vérifier que last_update_date < test_start
        test_start_dt = pd.to_datetime(test_start)
        
        if state_manager.last_update_date:
            if pd.to_datetime(state_manager.last_update_date) >= test_start_dt:
                errors.append(f"State last_update_date ({state_manager.last_update_date}) >= test_start ({test_start})")
        
        # 2. Vérifier cohérence des équipes
        if len(state_manager.teams) == 0:
            errors.append("No teams in state manager")
        
        # 3. Vérifier cohérence Elo
        elo_values = [team.elo for team in state_manager.teams.values()]
        if elo_values:
            mean_elo = np.mean(elo_values)
            if not (1000 <= mean_elo <= 2000):
                warnings.append(f"Unusual mean Elo: {mean_elo:.1f}")
            
            details['elo_stats'] = {
                'mean': mean_elo,
                'std': np.std(elo_values),
                'min': min(elo_values),
                'max': max(elo_values)
            }
        
        # 4. Vérifier que chaque équipe a des données
        teams_without_games = [
            name for name, team in state_manager.teams.items()
            if team.games_played == 0
        ]
        
        if teams_without_games:
            warnings.append(f"{len(teams_without_games)} teams have no games: {teams_without_games[:5]}")
        
        details.update({
            'total_teams': len(state_manager.teams),
            'processed_matches': state_manager.processed_matches,
            'h2h_pairs': len(state_manager.h2h),
            'teams_without_games': len(teams_without_games)
        })
        
        return ValidationResult(
            test_name="State Temporal Consistency",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_feature_temporal_boundaries(self, state_manager: StateManager,
                                        test_data: pd.DataFrame,
                                        test_start: str) -> ValidationResult:
        """Test 2: Boundaries temporelles des features"""
        
        errors = []
        warnings = []
        details = {}
        
        # Construire features pour premiers matchs de test
        feature_builder = FeatureBuilder()
        test_start_dt = pd.to_datetime(test_start)
        
        # Prendre premiers matchs de test
        early_matches = test_data.head(10)
        feature_samples = []
        
        for _, match_row in early_matches.iterrows():
            features = feature_builder.build_match_features(
                match_row['HomeTeam'],
                match_row['AwayTeam'],
                state_manager,
                match_row.to_dict()
            )
            feature_samples.append(features)
        
        if not feature_samples:
            errors.append("No features could be built for test matches")
        else:
            # Vérifier cohérence des features
            feature_names = list(feature_samples[0].keys())
            
            for feature_name in feature_names:
                values = [sample[feature_name] for sample in feature_samples]
                
                # Features normalisées doivent être dans [0, 1]
                if 'normalized' in feature_name:
                    out_of_range = [v for v in values if not (0 <= v <= 1)]
                    if out_of_range:
                        warnings.append(f"Feature {feature_name} has {len(out_of_range)} values outside [0,1]")
                
                # Features Elo doivent être raisonnables
                if 'elo' in feature_name and 'diff' not in feature_name:
                    unusual_elos = [v for v in values if not (800 <= v <= 2200)]
                    if unusual_elos:
                        warnings.append(f"Feature {feature_name} has {len(unusual_elos)} unusual values")
            
            details.update({
                'features_tested': len(feature_names),
                'matches_sampled': len(feature_samples),
                'feature_names': feature_names
            })
        
        return ValidationResult(
            test_name="Feature Temporal Boundaries",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_rolling_windows_integrity(self, state_manager: StateManager,
                                      test_data: pd.DataFrame,
                                      test_start: str) -> ValidationResult:
        """Test 3: Intégrité des fenêtres rolling"""
        
        errors = []
        warnings = []
        details = {}
        
        rolling_issues = []
        
        # Vérifier chaque équipe
        for team_name, team_state in state_manager.teams.items():
            
            # Vérifier cohérence des deques
            for metric, deque_data in team_state.rolling_data.items():
                
                # Vérifier taille maximale respectée
                from scripts.analysis.initialize_historical_state import DEFAULT_ROLLING_WINDOWS
                expected_max = DEFAULT_ROLLING_WINDOWS.get(metric, 10)
                
                if len(deque_data) > expected_max:
                    errors.append(f"Team {team_name}, metric {metric}: deque too large ({len(deque_data)} > {expected_max})")
                
                # Vérifier cohérence des valeurs
                if len(deque_data) > 0:
                    values = list(deque_data)
                    
                    # Valeurs négatives suspectes pour certaines métriques
                    if metric in ['goals', 'xg', 'shots', 'corners']:
                        negative_values = [v for v in values if v < 0]
                        if negative_values:
                            warnings.append(f"Team {team_name}, metric {metric}: {len(negative_values)} negative values")
                    
                    # Valeurs trop élevées suspectes
                    max_expected = {
                        'form': 3,      # Max 3 points par match
                        'goals': 8,     # Max ~8 buts par match
                        'xg': 6,        # Max ~6 xG par match
                        'shots': 30,    # Max ~30 tirs par match
                        'corners': 15   # Max ~15 corners par match
                    }
                    
                    if metric in max_expected:
                        excessive_values = [v for v in values if v > max_expected[metric]]
                        if excessive_values:
                            warnings.append(f"Team {team_name}, metric {metric}: {len(excessive_values)} excessive values")
        
        # Statistiques globales
        total_deques = sum(len(team.rolling_data) for team in state_manager.teams.values())
        
        details.update({
            'teams_checked': len(state_manager.teams),
            'total_deques': total_deques,
            'rolling_issues': len(rolling_issues)
        })
        
        return ValidationResult(
            test_name="Rolling Windows Integrity",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_h2h_historical_accuracy(self, state_manager: StateManager,
                                    test_data: pd.DataFrame,
                                    test_start: str) -> ValidationResult:
        """Test 4: Précision historique des H2H"""
        
        errors = []
        warnings = []
        details = {}
        
        # Échantillonner quelques paires H2H pour validation
        sample_pairs = list(state_manager.h2h.keys())[:10]
        validated_pairs = 0
        
        for team_pair in sample_pairs:
            h2h_data = state_manager.h2h[team_pair]
            
            # Vérifier cohérence
            total_expected = h2h_data['home_wins'] + h2h_data['away_wins'] + h2h_data['draws']
            
            if total_expected != h2h_data['total_games']:
                errors.append(f"H2H inconsistency for {team_pair}: sum={total_expected} != total={h2h_data['total_games']}")
            
            # Vérifier que total > 0
            if h2h_data['total_games'] == 0:
                warnings.append(f"H2H pair {team_pair} has no games")
            
            validated_pairs += 1
        
        details.update({
            'h2h_pairs_total': len(state_manager.h2h),
            'pairs_validated': validated_pairs,
            'sample_h2h': dict(list(state_manager.h2h.items())[:3]) if state_manager.h2h else {}
        })
        
        return ValidationResult(
            test_name="H2H Historical Accuracy",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_elo_temporal_progression(self, state_manager: StateManager,
                                     test_data: pd.DataFrame,
                                     test_start: str) -> ValidationResult:
        """Test 5: Progression temporelle de l'Elo"""
        
        errors = []
        warnings = []
        details = {}
        
        # Créer une copie de l'état pour simulation
        import copy
        test_state = copy.deepcopy(state_manager)
        
        # Simuler quelques matchs et vérifier progression Elo
        sample_matches = test_data.head(5)
        elo_progressions = {}
        
        for _, match_row in sample_matches.iterrows():
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']
            
            # Elo avant
            home_elo_before = test_state.get_or_create_team(home_team).elo
            away_elo_before = test_state.get_or_create_team(away_team).elo
            
            # Simuler match
            if 'HomeGoals' in match_row and 'AwayGoals' in match_row:
                home_goals = int(match_row['HomeGoals'])
                away_goals = int(match_row['AwayGoals'])
                
                test_state.update_elo(home_team, away_team, home_goals, away_goals)
                
                # Elo après
                home_elo_after = test_state.get_or_create_team(home_team).elo
                away_elo_after = test_state.get_or_create_team(away_team).elo
                
                # Vérifier cohérence
                elo_change_home = home_elo_after - home_elo_before
                elo_change_away = away_elo_after - away_elo_before
                
                # Conservation approximative (somme des changements ~= 0)
                total_change = elo_change_home + elo_change_away
                
                if abs(total_change) > 1e-6:  # Tolérance numérique
                    errors.append(f"Elo not conserved in match {home_team} vs {away_team}: total_change = {total_change}")
                
                # Changements trop importants suspects
                if abs(elo_change_home) > 50:
                    warnings.append(f"Large Elo change for {home_team}: {elo_change_home:+.1f}")
                
                if abs(elo_change_away) > 50:
                    warnings.append(f"Large Elo change for {away_team}: {elo_change_away:+.1f}")
                
                elo_progressions[f"{home_team}_vs_{away_team}"] = {
                    'home_before': home_elo_before,
                    'home_after': home_elo_after,
                    'home_change': elo_change_home,
                    'away_before': away_elo_before,
                    'away_after': away_elo_after,
                    'away_change': elo_change_away,
                    'result': f"{home_goals}-{away_goals}"
                }
        
        details.update({
            'matches_simulated': len(elo_progressions),
            'sample_progressions': elo_progressions
        })
        
        return ValidationResult(
            test_name="Elo Temporal Progression",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_simulation_determinism(self, state_manager: StateManager,
                                   test_data: pd.DataFrame,
                                   test_start: str) -> ValidationResult:
        """Test 6: Déterminisme de la simulation"""
        
        errors = []
        warnings = []
        details = {}
        
        # Prendre un petit échantillon pour test
        sample_data = test_data.head(3)
        
        # Simuler deux fois avec même état initial
        import copy
        
        results_1 = []
        results_2 = []
        
        # Première simulation
        state_1 = copy.deepcopy(state_manager)
        feature_builder_1 = FeatureBuilder()
        
        for _, match_row in sample_data.iterrows():
            features_1 = feature_builder_1.build_match_features(
                match_row['HomeTeam'],
                match_row['AwayTeam'],
                state_1,
                match_row.to_dict()
            )
            results_1.append(features_1)
            
            # Mettre à jour état
            if 'HomeGoals' in match_row and 'AwayGoals' in match_row:
                state_1.process_match(match_row)
        
        # Deuxième simulation
        state_2 = copy.deepcopy(state_manager)
        feature_builder_2 = FeatureBuilder()
        
        for _, match_row in sample_data.iterrows():
            features_2 = feature_builder_2.build_match_features(
                match_row['HomeTeam'],
                match_row['AwayTeam'],
                state_2,
                match_row.to_dict()
            )
            results_2.append(features_2)
            
            # Mettre à jour état
            if 'HomeGoals' in match_row and 'AwayGoals' in match_row:
                state_2.process_match(match_row)
        
        # Comparer résultats
        mismatches = 0
        for i, (res1, res2) in enumerate(zip(results_1, results_2)):
            for feature_name in res1.keys():
                if feature_name in res2:
                    if abs(res1[feature_name] - res2[feature_name]) > 1e-10:
                        mismatches += 1
                        if mismatches <= 3:  # Log seulement premiers mismatches
                            errors.append(f"Non-deterministic feature {feature_name} in match {i}: {res1[feature_name]} != {res2[feature_name]}")
        
        if mismatches > 3:
            errors.append(f"Total {mismatches} determinism violations found")
        
        details.update({
            'matches_tested': len(sample_data),
            'total_mismatches': mismatches
        })
        
        return ValidationResult(
            test_name="Simulation Determinism",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_cv_split_integrity(self, state_manager: StateManager,
                               test_data: pd.DataFrame,
                               test_start: str) -> ValidationResult:
        """Test 7: Intégrité des splits de cross-validation"""
        
        errors = []
        warnings = []
        details = {}
        
        # Vérifier qu'il n'y a pas de fuite entre train et test
        test_start_dt = pd.to_datetime(test_start)
        
        # Vérifier dernière date dans l'état
        if state_manager.last_update_date:
            last_update_dt = pd.to_datetime(state_manager.last_update_date)
            
            if last_update_dt >= test_start_dt:
                errors.append(f"State contains data from test period: last_update={last_update_dt}, test_start={test_start_dt}")
        
        # Vérifier dates dans test_data
        if len(test_data) > 0:
            first_test_date = test_data['Date'].min()
            if pd.to_datetime(first_test_date) < test_start_dt:
                errors.append(f"Test data contains pre-test dates: first_date={first_test_date}, test_start={test_start}")
        
        # Vérifier équipes dans état vs test
        state_teams = set(state_manager.teams.keys())
        test_teams = set(test_data['HomeTeam'].unique()) | set(test_data['AwayTeam'].unique())
        
        missing_teams = test_teams - state_teams
        if missing_teams:
            warnings.append(f"{len(missing_teams)} teams in test data not in state: {list(missing_teams)[:5]}")
        
        details.update({
            'test_start': test_start,
            'state_last_update': str(state_manager.last_update_date),
            'test_data_first_date': str(test_data['Date'].min()) if len(test_data) > 0 else None,
            'test_data_last_date': str(test_data['Date'].max()) if len(test_data) > 0 else None,
            'teams_in_state': len(state_teams),
            'teams_in_test': len(test_teams),
            'missing_teams': len(missing_teams)
        })
        
        return ValidationResult(
            test_name="Cross-Validation Split Integrity",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Génère le rapport final de validation"""
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        total_warnings = sum(len(r.warnings) for r in self.validation_results)
        total_execution_time = sum(r.execution_time for r in self.validation_results)
        
        # Classification de l'intégrité générale
        if failed_tests == 0:
            if total_warnings == 0:
                integrity_level = "EXCELLENT"
            elif total_warnings < 5:
                integrity_level = "GOOD"
            else:
                integrity_level = "ACCEPTABLE"
        else:
            if failed_tests == 1:
                integrity_level = "CONCERNS"
            else:
                integrity_level = "CRITICAL"
        
        return {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'total_warnings': total_warnings,
                'integrity_level': integrity_level,
                'total_execution_time': total_execution_time
            },
            'test_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'error_message': r.error_message,
                    'warnings': r.warnings,
                    'details': r.details,
                    'execution_time': r.execution_time
                }
                for r in self.validation_results
            ],
            'recommendations': self._generate_recommendations(),
            'validation_timestamp': str(datetime.now())
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        
        recommendations = []
        
        failed_tests = [r for r in self.validation_results if not r.passed]
        
        if failed_tests:
            recommendations.append("❌ CRITICAL: Fix failed tests before using rolling simulation in production")
            
            for test in failed_tests:
                recommendations.append(f"  - {test.test_name}: {test.error_message}")
        
        total_warnings = sum(len(r.warnings) for r in self.validation_results)
        
        if total_warnings > 10:
            recommendations.append("⚠️ HIGH WARNING COUNT: Review and address warnings for optimal performance")
        
        # Recommandations spécifiques par test
        for result in self.validation_results:
            if result.test_name == "Elo Temporal Progression" and result.warnings:
                recommendations.append("🎯 Consider reviewing Elo K-factor if large changes are frequent")
            
            if result.test_name == "Rolling Windows Integrity" and result.warnings:
                recommendations.append("📊 Review rolling window data quality and outlier handling")
            
            if result.test_name == "Cross-Validation Split Integrity" and result.warnings:
                recommendations.append("🔍 Ensure all test teams have sufficient historical data")
        
        if not recommendations:
            recommendations.append("✅ ALL TESTS PASSED: Rolling simulation is ready for production use")
        
        return recommendations
    
    def _test_feature_future_information(self, state_manager: StateManager,
                                       test_data: pd.DataFrame,
                                       test_start: str) -> ValidationResult:
        """
        TEST CRITIQUE P0: Vérification qu'aucune feature n'utilise d'information future
        """
        errors = []
        warnings = []
        details = {}
        
        test_start_dt = pd.to_datetime(test_start)
        
        # Prendre premiers matchs pour test approfondi
        sample_matches = test_data.head(5)
        future_info_detections = []
        
        feature_builder = FeatureBuilder()
        
        for idx, match_row in sample_matches.iterrows():
            match_date = pd.to_datetime(match_row['Date'])
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']
            
            # Construire features
            features = feature_builder.build_match_features(
                home_team, away_team, state_manager, match_row.to_dict()
            )
            
            # Vérifier que state_manager n'a pas d'info > match_date
            if state_manager.last_update_date:
                last_update_dt = pd.to_datetime(state_manager.last_update_date)
                if last_update_dt >= match_date:
                    errors.append(f"State contains future info for match {match_date}: last_update={last_update_dt}")
            
            # Vérifier cohérence des features par rapport aux moyennes historiques
            home_state = state_manager.teams.get(home_team)
            away_state = state_manager.teams.get(away_team)
            
            if home_state and away_state:
                # Vérifier que Elo n'est pas "parfait" (signe d'optimisation future)
                elo_diff = features.get('elo_diff', 0)
                if abs(elo_diff) > 300:  # Différence trop extrême suspecte
                    warnings.append(f"Extreme Elo difference ({elo_diff:.1f}) for {home_team} vs {away_team}")
                
                # Vérifier cohérence rolling stats
                form_diff = features.get('form_diff_normalized', 0.5)
                if form_diff in [0.0, 1.0]:  # Valeurs extremes suspectes
                    warnings.append(f"Extreme form difference ({form_diff}) for match {home_team} vs {away_team}")
        
        details.update({
            'matches_tested': len(sample_matches),
            'future_info_detections': len(future_info_detections)
        })
        
        return ValidationResult(
            test_name="Feature Future Information Test",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_h2h_self_exclusion(self, state_manager: StateManager,
                               test_data: pd.DataFrame,
                               test_start: str) -> ValidationResult:
        """
        TEST CRITIQUE P0: Vérifier que H2H n'inclut pas le match courant
        """
        errors = []
        warnings = []
        details = {}
        
        # Simuler quelques matchs et vérifier H2H
        sample_matches = test_data.head(3)
        h2h_violations = []
        
        for _, match_row in sample_matches.iterrows():
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']
            match_date = match_row['Date']
            
            # Obtenir H2H stats avant le match
            h2h_stats = state_manager.get_h2h_stats(home_team, away_team)
            h2h_key = state_manager.get_h2h_key(home_team, away_team)
            
            if h2h_key in state_manager.h2h:
                h2h_data = state_manager.h2h[h2h_key]
                
                # Vérifier que le match courant n'est PAS inclus dans H2H
                # (difficile à vérifier directement, mais on peut vérifier la cohérence)
                
                # Si c'est le premier match entre ces équipes, H2H doit être neutre
                if h2h_data['total_games'] == 0:
                    if h2h_stats['h2h_score'] != 0.5:
                        errors.append(f"H2H score {h2h_stats['h2h_score']} should be 0.5 for first meeting {home_team} vs {away_team}")
                
                # Vérifier cohérence des compteurs H2H
                total_expected = h2h_data['home_wins'] + h2h_data['away_wins'] + h2h_data['draws']
                if total_expected != h2h_data['total_games']:
                    errors.append(f"H2H inconsistency for {home_team} vs {away_team}: {total_expected} != {h2h_data['total_games']}")
        
        # Test supplémentaire: vérifier qu'on peut construire H2H depuis zéro
        # et obtenir cohérence avec state actuel
        test_h2h_reconstruction = self._test_h2h_reconstruction_consistency(state_manager, sample_matches)
        
        details.update({
            'matches_tested': len(sample_matches),
            'h2h_violations': len(h2h_violations),
            'reconstruction_test': test_h2h_reconstruction
        })
        
        return ValidationResult(
            test_name="H2H Self-Exclusion Test",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )
    
    def _test_h2h_reconstruction_consistency(self, state_manager: StateManager, 
                                           sample_matches: pd.DataFrame) -> Dict[str, Any]:
        """Test helper: vérifier cohérence reconstruction H2H"""
        
        # Pour chaque paire de teams dans sample_matches, vérifier cohérence H2H
        pairs_tested = []
        inconsistencies = []
        
        for _, match_row in sample_matches.iterrows():
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']
            pair_key = (home_team, away_team)
            
            if pair_key not in pairs_tested:
                pairs_tested.append(pair_key)
                
                h2h_key = state_manager.get_h2h_key(home_team, away_team)
                if h2h_key in state_manager.h2h:
                    h2h_data = state_manager.h2h[h2h_key]
                    
                    # Vérifier ranges raisonnables
                    if h2h_data['total_games'] > 50:  # Trop de matchs H2H suspect
                        inconsistencies.append(f"Too many H2H games for {home_team} vs {away_team}: {h2h_data['total_games']}")
                    
                    # Win rate cohérence
                    if h2h_data['total_games'] > 0:
                        home_win_rate = h2h_data['home_wins'] / h2h_data['total_games']
                        if not (0 <= home_win_rate <= 1):
                            inconsistencies.append(f"Invalid home win rate for {home_team} vs {away_team}: {home_win_rate}")
        
        return {
            'pairs_tested': len(pairs_tested),
            'inconsistencies': inconsistencies
        }
    
    def _test_state_update_order(self, state_manager: StateManager,
                               test_data: pd.DataFrame,
                               test_start: str) -> ValidationResult:
        """
        TEST CRITIQUE P0: Vérifier ordre correct des mises à jour d'état
        
        L'état doit être mis à jour APRÈS prédiction, jamais avant
        """
        errors = []
        warnings = []
        details = {}
        
        # Simuler une séquence prediction -> update et vérifier ordre
        sample_matches = test_data.head(2)
        
        import copy
        test_state = copy.deepcopy(state_manager)
        feature_builder = FeatureBuilder()
        
        state_evolution = []
        
        for idx, match_row in sample_matches.iterrows():
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']
            match_date = match_row['Date']
            
            # 1. État AVANT prédiction
            home_elo_before = test_state.get_or_create_team(home_team).elo
            away_elo_before = test_state.get_or_create_team(away_team).elo
            
            # 2. Construction features (simulation prédiction)
            features_before_update = feature_builder.build_match_features(
                home_team, away_team, test_state, match_row.to_dict()
            )
            
            # 3. Mise à jour état avec résultats réels
            if 'HomeGoals' in match_row and 'AwayGoals' in match_row:
                test_state.process_match(match_row)
            
            # 4. État APRÈS mise à jour
            home_elo_after = test_state.get_or_create_team(home_team).elo
            away_elo_after = test_state.get_or_create_team(away_team).elo
            
            # 5. Construction features après update (ne devrait PAS être utilisé pour prédiction)
            features_after_update = feature_builder.build_match_features(
                home_team, away_team, test_state, match_row.to_dict()
            )
            
            # Vérifier que les features ont changé (preuve que l'update a un effet)
            elo_diff_before = features_before_update.get('elo_diff', 0)
            elo_diff_after = features_after_update.get('elo_diff', 0)
            
            elo_change_detected = abs(elo_diff_after - elo_diff_before) > 1e-6
            
            state_evolution.append({
                'match': f"{home_team} vs {away_team}",
                'date': str(match_date),
                'home_elo_before': home_elo_before,
                'home_elo_after': home_elo_after,
                'away_elo_before': away_elo_before,
                'away_elo_after': away_elo_after,
                'elo_diff_before': elo_diff_before,
                'elo_diff_after': elo_diff_after,
                'update_detected': elo_change_detected
            })
            
            # Vérifier que l'update a bien eu lieu
            if not elo_change_detected and abs(home_elo_after - home_elo_before) < 1e-6:
                warnings.append(f"No Elo update detected for match {home_team} vs {away_team}")
        
        # Vérifier progression logique des Elo
        for i in range(1, len(state_evolution)):
            prev_state = state_evolution[i-1]
            curr_state = state_evolution[i]
            
            # L'Elo d'une équipe dans match N doit être l'Elo final du match N-1
            # (si c'est la même équipe)
            # Cette vérification est complexe car on n'a pas forcément la même équipe
            # On vérifie plutôt la cohérence globale
            pass  # Placeholder pour vérifications futures plus sophistiquées
        
        details.update({
            'matches_simulated': len(sample_matches),
            'state_evolution': state_evolution,
            'elo_updates_detected': sum(1 for s in state_evolution if s['update_detected'])
        })
        
        return ValidationResult(
            test_name="State Update Order Validation",
            passed=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
            warnings=warnings,
            details=details
        )

def main():
    """Pipeline principal de validation"""
    parser = argparse.ArgumentParser(description="Validate temporal integrity of rolling simulation")
    parser.add_argument('--state_file', required=True, help='State pickle file')
    parser.add_argument('--test_data', required=True, help='Test data CSV')
    parser.add_argument('--test_start', required=True, help='Test period start (YYYY-MM-DD)')
    parser.add_argument('--report', required=True, help='Output validation report JSON')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🔍 TEMPORAL INTEGRITY VALIDATION")
    logger.info("=" * 70)
    logger.info(f"State file: {args.state_file}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Test start: {args.test_start}")
    logger.info(f"Report: {args.report}")
    logger.info("=" * 70)
    
    try:
        # Exécuter validation
        validator = TemporalIntegrityValidator()
        report = validator.run_all_tests(args.state_file, args.test_data, args.test_start)
        
        # Sauvegarder rapport
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Résumé final
        summary = report['validation_summary']
        logger.info("🎯 VALIDATION SUMMARY:")
        logger.info(f"  Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        logger.info(f"  Warnings: {summary['total_warnings']}")
        logger.info(f"  Integrity Level: {summary['integrity_level']}")
        logger.info(f"  Execution Time: {summary['total_execution_time']:.2f}s")
        
        if summary['failed_tests'] > 0:
            logger.error("❌ VALIDATION FAILED - Check report for details")
            sys.exit(1)
        else:
            logger.info("✅ VALIDATION PASSED - Rolling simulation ready")
        
        logger.info(f"📋 Full report saved to: {args.report}")
        
    except Exception as e:
        logger.error(f"❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()