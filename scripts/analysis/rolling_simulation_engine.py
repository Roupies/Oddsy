#!/usr/bin/env python3
"""
rolling_simulation_engine.py

Moteur de simulation rolling temporellement sûr pour Oddsy.

OBJECTIF:
- Simulation match-day par match-day avec état incrémental
- Features construites UNIQUEMENT à partir de l'état passé 
- Zéro data leakage temporel
- Production-ready evaluation pipeline

USAGE:
python scripts/analysis/rolling_simulation_engine.py \
    --initial_state data/states/state_2024_season.pkl \
    --test_data data/processed/premier_league_full_with_results.csv \
    --test_start 2024-08-01 \
    --test_end 2025-05-01 \
    --model models/v13_production_model.joblib \
    --output results/rolling_simulation_2024.json

ARCHITECTURE:
1. RollingSimulationEngine - Moteur principal de simulation
2. FeatureBuilder - Construction des features à partir de l'état
3. MatchDayProcessor - Traitement journée par journée
4. ResultsAggregator - Agrégation et métriques
"""

import sys
import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# ML imports
import joblib
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

# Import state manager
from scripts.analysis.initialize_historical_state import StateManager, load_state

class DynamicNormalizer:
    """
    AMÉLIORATION P2: Normalisation dynamique basée sur distribution réelle
    
    Au lieu de ranges fixes hardcodés, calcule normalisation basée sur:
    1. Distribution actuelle des features dans StateManager
    2. Mise à jour progressive des stats de normalisation
    3. Fallbacks pour features nouvelles
    """
    
    def __init__(self, update_frequency: int = 50):
        self.update_frequency = update_frequency  # Recalcul tous les N matchs
        self.normalization_stats = {}
        self.call_count = 0
        self.logger = setup_logging()
        
        # Stats par défaut pour fallback
        self.default_stats = {
            'elo_diff': {'mean': 0, 'std': 150, 'min': -400, 'max': 400},
            'form_diff': {'mean': 0, 'std': 1.5, 'min': -3, 'max': 3},
            'goals_diff': {'mean': 0, 'std': 1.0, 'min': -3, 'max': 3},
            'xg_diff': {'mean': 0, 'std': 1.0, 'min': -3, 'max': 3},
            'shots_diff': {'mean': 0, 'std': 8, 'min': -20, 'max': 20},
            'corners_diff': {'mean': 0, 'std': 3, 'min': -8, 'max': 8}
        }
    
    def update_normalization_stats(self, state_manager: StateManager):
        """Met à jour les statistiques de normalisation"""
        
        if not state_manager.teams:
            return
        
        self.call_count += 1
        
        # Mise à jour périodique seulement
        if self.call_count % self.update_frequency != 0:
            return
        
        self.logger.debug(f"🔄 Updating normalization stats (call {self.call_count})")
        
        # Collecter valeurs actuelles pour chaque métrique
        elo_values = [team.elo for team in state_manager.teams.values()]
        
        if len(elo_values) < 2:
            return
        
        # Calculer stats Elo
        elo_array = np.array(elo_values)
        elo_combinations = []
        
        # Générer toutes combinaisons possibles de différences Elo
        teams = list(state_manager.teams.keys())
        for i in range(min(len(teams), 100)):  # Limiter pour performance
            for j in range(i+1, min(len(teams), 100)):
                home_elo = state_manager.teams[teams[i]].elo
                away_elo = state_manager.teams[teams[j]].elo
                elo_combinations.append(home_elo - away_elo)
        
        if elo_combinations:
            elo_diff_array = np.array(elo_combinations)
            self.normalization_stats['elo_diff'] = {
                'mean': np.mean(elo_diff_array),
                'std': np.std(elo_diff_array),
                'min': np.min(elo_diff_array),
                'max': np.max(elo_diff_array)
            }
        
        # Calculer stats forme, goals, etc. si disponible
        form_values = []
        goals_values = []
        
        for team in state_manager.teams.values():
            if hasattr(team, 'data_handler') and team.data_handler:
                form_avg = team.get_rolling_mean('form', team.data_handler)
                goals_avg = team.get_rolling_mean('goals', team.data_handler)
                form_values.append(form_avg)
                goals_values.append(goals_avg)
        
        # Stats forme
        if len(form_values) >= 2:
            form_combinations = []
            for i in range(min(len(form_values), 50)):
                for j in range(i+1, min(len(form_values), 50)):
                    form_combinations.append(form_values[i] - form_values[j])
            
            if form_combinations:
                form_array = np.array(form_combinations)
                self.normalization_stats['form_diff'] = {
                    'mean': np.mean(form_array),
                    'std': np.std(form_array),
                    'min': np.min(form_array), 
                    'max': np.max(form_array)
                }
        
        # Stats goals
        if len(goals_values) >= 2:
            goals_combinations = []
            for i in range(min(len(goals_values), 50)):
                for j in range(i+1, min(len(goals_values), 50)):
                    goals_combinations.append(goals_values[i] - goals_values[j])
            
            if goals_combinations:
                goals_array = np.array(goals_combinations)
                self.normalization_stats['goals_diff'] = {
                    'mean': np.mean(goals_array),
                    'std': np.std(goals_array),
                    'min': np.min(goals_array),
                    'max': np.max(goals_array)
                }
        
        self.logger.debug(f"✅ Updated normalization stats: {len(self.normalization_stats)} metrics")
    
    def normalize_value(self, value: float, metric_type: str, method: str = 'z_score') -> float:
        """
        Normalise une valeur selon la méthode choisie
        
        Methods:
        - 'z_score': (value - mean) / std
        - 'min_max': (value - min) / (max - min)  
        - 'robust': Utilise IQR pour robustesse aux outliers
        """
        
        # Utiliser stats dynamiques si disponibles, sinon fallback
        if metric_type in self.normalization_stats:
            stats = self.normalization_stats[metric_type]
        elif metric_type in self.default_stats:
            stats = self.default_stats[metric_type]
        else:
            # Fallback ultime
            return np.clip((value + 3) / 6, 0, 1)  # Approximation [-3, 3] -> [0, 1]
        
        if method == 'z_score':
            if stats['std'] > 1e-6:
                z_score = (value - stats['mean']) / stats['std']
                # Mapper z-score à [0, 1] en assumant ~95% dans [-2, 2]
                return np.clip((z_score + 2) / 4, 0, 1)
            else:
                return 0.5
        
        elif method == 'min_max':
            value_range = stats['max'] - stats['min']
            if value_range > 1e-6:
                return np.clip((value - stats['min']) / value_range, 0, 1)
            else:
                return 0.5
        
        elif method == 'robust':
            # Utiliser IQR approximé (pour robustesse)
            # Approximation: mean ± 0.675*std couvre ~50% des données
            q25_approx = stats['mean'] - 0.675 * stats['std'] 
            q75_approx = stats['mean'] + 0.675 * stats['std']
            iqr_approx = q75_approx - q25_approx
            
            if iqr_approx > 1e-6:
                normalized = (value - q25_approx) / iqr_approx
                # Clipper à range raisonnable et mapper à [0,1]
                normalized = np.clip(normalized, -2, 3)
                return (normalized + 2) / 5  # [-2, 3] -> [0, 1]
            else:
                return 0.5
        
        else:
            # Method non reconnue, fallback z_score
            return self.normalize_value(value, metric_type, 'z_score')
    
    def get_normalization_summary(self) -> Dict[str, Any]:
        """Retourne résumé des stats de normalisation pour debugging"""
        return {
            'call_count': self.call_count,
            'update_frequency': self.update_frequency,
            'metrics_tracked': list(self.normalization_stats.keys()),
            'normalization_stats': self.normalization_stats.copy(),
            'last_update': self.call_count // self.update_frequency * self.update_frequency
        }

@dataclass
class MatchPrediction:
    """Résultat de prédiction pour un match"""
    match_id: str
    date: str
    home_team: str
    away_team: str
    predicted_result: int  # 0=Home, 1=Draw, 2=Away
    predicted_proba: List[float]  # [P(Home), P(Draw), P(Away)]
    actual_result: Optional[int] = None
    features: Dict[str, float] = field(default_factory=dict)

@dataclass  
class MatchDayResults:
    """Résultats d'une journée"""
    date: str
    match_day: Optional[str]
    predictions: List[MatchPrediction]
    accuracy: Optional[float] = None
    log_loss_score: Optional[float] = None

class FeatureBuilder:
    """Construction des features à partir de l'état courant"""
    
    def __init__(self, feature_config: Optional[Dict] = None, 
                 use_dynamic_normalization: bool = True,
                 normalization_method: str = 'robust'):
        self.logger = setup_logging()
        self.feature_config = feature_config or self._default_feature_config()
        
        # AMÉLIORATION P2: Normalisation dynamique
        self.use_dynamic_normalization = use_dynamic_normalization
        self.normalization_method = normalization_method
        
        if self.use_dynamic_normalization:
            self.dynamic_normalizer = DynamicNormalizer()
            self.logger.info(f"🧮 Dynamic normalization enabled (method: {normalization_method})")
        else:
            self.dynamic_normalizer = None
        
    def _default_feature_config(self) -> Dict:
        """Configuration par défaut des features"""
        return {
            'elo_features': True,
            'form_features': True,
            'goals_features': True,
            'xg_features': True,
            'shots_features': True,
            'corners_features': True,
            'h2h_features': True,
            'normalize_features': True
        }
    
    def build_match_features(self, home_team: str, away_team: str, 
                           state_manager: StateManager, 
                           match_data: Optional[Dict] = None) -> Dict[str, float]:
        """Construit les features pour un match à partir de l'état courant"""
        features = {}
        
        # États des équipes
        home_state = state_manager.get_or_create_team(home_team)
        away_state = state_manager.get_or_create_team(away_team)
        
        # Mise à jour stats de normalisation si dynamique activée
        if self.dynamic_normalizer:
            self.dynamic_normalizer.update_normalization_stats(state_manager)
        
        # 1. Features Elo
        if self.feature_config['elo_features']:
            features['home_elo'] = home_state.elo
            features['away_elo'] = away_state.elo
            features['elo_diff'] = home_state.elo - away_state.elo
            features['elo_diff_normalized'] = self._normalize_feature(features['elo_diff'], 'elo_diff')
        
        # 2. Features forme (points récents)
        if self.feature_config['form_features']:
            home_form = home_state.get_rolling_mean('form')
            away_form = away_state.get_rolling_mean('form')
            features['home_form_5'] = home_form
            features['away_form_5'] = away_form
            features['form_diff'] = home_form - away_form
            features['form_diff_normalized'] = self._normalize_feature(features['form_diff'], 'form_diff')
        
        # 3. Features buts
        if self.feature_config['goals_features']:
            home_goals = home_state.get_rolling_mean('goals')
            away_goals = away_state.get_rolling_mean('goals')
            features['home_goals_avg_5'] = home_goals
            features['away_goals_avg_5'] = away_goals
            features['goals_diff'] = home_goals - away_goals
            features['goals_diff_normalized'] = self._normalize_feature(features['goals_diff'], 'goals_diff')
        
        # 4. Features xG (si disponible)
        if self.feature_config['xg_features']:
            home_xg = home_state.get_rolling_mean('xg')
            away_xg = away_state.get_rolling_mean('xg')
            features['home_xg_avg_5'] = home_xg
            features['away_xg_avg_5'] = away_xg
            features['xg_diff'] = home_xg - away_xg
            features['xg_diff_normalized'] = self._normalize_feature(features['xg_diff'], 'xg_diff')
        
        # 5. Features tirs (si disponible)
        if self.feature_config['shots_features']:
            home_shots = home_state.get_rolling_mean('shots')
            away_shots = away_state.get_rolling_mean('shots')
            features['home_shots_avg_5'] = home_shots
            features['away_shots_avg_5'] = away_shots
            features['shots_diff'] = home_shots - away_shots
            features['shots_diff_normalized'] = self._normalize_feature(features['shots_diff'], 'shots_diff')
        
        # 6. Features corners
        if self.feature_config['corners_features']:
            home_corners = home_state.get_rolling_mean('corners')
            away_corners = away_state.get_rolling_mean('corners')
            features['home_corners_avg_5'] = home_corners
            features['away_corners_avg_5'] = away_corners
            features['corners_diff'] = home_corners - away_corners
            features['corners_diff_normalized'] = self._normalize_feature(features['corners_diff'], 'corners_diff')
        
        # 7. Features H2H
        if self.feature_config['h2h_features']:
            h2h_stats = state_manager.get_h2h_stats(home_team, away_team)
            features['h2h_score'] = h2h_stats['h2h_score']
        
        # 8. Features contextuelles (si données disponibles)
        if match_data:
            if 'Matchday' in match_data:
                features['matchday'] = float(match_data['Matchday'])
                features['matchday_normalized'] = features['matchday'] / 38.0
        
        return features
    
    def build_v21_compatible_features(self, home_team: str, away_team: str,
                                     state_manager: StateManager, 
                                     match_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Construit exactement les 5 features compatibles avec le modèle v2.1 (54.2%)
        
        Features exactes requises:
        1. elo_diff_normalized
        2. form_diff_normalized  
        3. h2h_score
        4. matchday_normalized
        5. corners_diff_normalized
        """
        features = {}
        
        # États des équipes
        home_state = state_manager.get_or_create_team(home_team)
        away_state = state_manager.get_or_create_team(away_team)
        
        # Mise à jour stats de normalisation si dynamique activée
        if self.dynamic_normalizer:
            self.dynamic_normalizer.update_normalization_stats(state_manager)
        
        # 1. elo_diff_normalized
        elo_diff = home_state.elo - away_state.elo
        features['elo_diff_normalized'] = self._normalize_feature(elo_diff, 'elo_diff')
        
        # 2. form_diff_normalized
        home_form = home_state.get_rolling_mean('form')
        away_form = away_state.get_rolling_mean('form')
        form_diff = home_form - away_form
        features['form_diff_normalized'] = self._normalize_feature(form_diff, 'form_diff')
        
        # 3. h2h_score
        h2h_stats = state_manager.get_h2h_stats(home_team, away_team)
        features['h2h_score'] = h2h_stats['h2h_score']
        
        # 4. matchday_normalized
        if match_data and 'Matchday' in match_data:
            matchday = float(match_data['Matchday'])
            features['matchday_normalized'] = matchday / 38.0
        else:
            # Fallback: estimer based on date if available
            features['matchday_normalized'] = 0.5  # Mid-season par défaut
        
        # 5. corners_diff_normalized
        home_corners = home_state.get_rolling_mean('corners')
        away_corners = away_state.get_rolling_mean('corners')
        corners_diff = home_corners - away_corners
        features['corners_diff_normalized'] = self._normalize_feature(corners_diff, 'corners_diff')
        
        # Validation: exactement 5 features
        assert len(features) == 5, f"Expected 5 features, got {len(features)}: {list(features.keys())}"
        
        return features
    
    def _normalize_feature(self, value: float, feature_type: str) -> float:
        """
        AMÉLIORATION P2: Normalisation unifiée avec support dynamique
        """
        if self.dynamic_normalizer:
            return self.dynamic_normalizer.normalize_value(value, feature_type, self.normalization_method)
        else:
            # Fallback vers normalisation statique (ancienne méthode)
            return self._static_normalize(value, feature_type)
    
    def _static_normalize(self, value: float, feature_type: str) -> float:
        """Normalisation statique (fallback)"""
        static_ranges = {
            'elo_diff': (-400, 400),
            'form_diff': (-3, 3),
            'goals_diff': (-2, 2),
            'xg_diff': (-2, 2),
            'shots_diff': (-10, 10),
            'corners_diff': (-5, 5)
        }
        
        if feature_type in static_ranges:
            min_val, max_val = static_ranges[feature_type]
            return np.clip((value - min_val) / (max_val - min_val), 0, 1)
        else:
            # Fallback générique
            return np.clip((value + 3) / 6, 0, 1)
    
    def get_normalization_summary(self) -> Optional[Dict[str, Any]]:
        """Retourne résumé de normalisation pour debugging"""
        if self.dynamic_normalizer:
            return self.dynamic_normalizer.get_normalization_summary()
        else:
            return {'type': 'static', 'method': 'fixed_ranges'}

class MatchDayProcessor:
    """Traitement d'une journée de matchs"""
    
    def __init__(self, model, feature_builder: FeatureBuilder):
        self.model = model
        self.feature_builder = feature_builder
        self.logger = setup_logging()
    
    def process_matchday(self, matchday_data: pd.DataFrame, 
                        state_manager: StateManager) -> MatchDayResults:
        """Traite tous les matchs d'une journée"""
        date = matchday_data.iloc[0]['Date']
        match_day = matchday_data.iloc[0].get('Matchday', 'Unknown')
        
        self.logger.debug(f"Processing {len(matchday_data)} matches for {date}")
        
        predictions = []
        X_features = []
        feature_names = None
        
        # 1. Construire features pour tous les matchs de la journée
        for idx, match_row in matchday_data.iterrows():
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']
            
            # Features à partir de l'état ACTUEL (pas de données futures)
            features = self.feature_builder.build_match_features(
                home_team, away_team, state_manager, match_row.to_dict()
            )
            
            if feature_names is None:
                feature_names = list(features.keys())
            
            # Assurer l'ordre des features
            feature_vector = [features.get(name, 0.0) for name in feature_names]
            X_features.append(feature_vector)
            
            # Préparer objet prédiction
            match_pred = MatchPrediction(
                match_id=f"{date}_{home_team}_{away_team}",
                date=str(date),
                home_team=home_team,
                away_team=away_team,
                predicted_result=-1,  # À remplir
                predicted_proba=[],
                features=features
            )
            
            # Résultat réel si disponible
            if 'HomeGoals' in match_row and 'AwayGoals' in match_row:
                home_goals = int(match_row['HomeGoals'])
                away_goals = int(match_row['AwayGoals'])
                
                if home_goals > away_goals:
                    match_pred.actual_result = 0  # Home win
                elif home_goals == away_goals:
                    match_pred.actual_result = 1  # Draw
                else:
                    match_pred.actual_result = 2  # Away win
            
            predictions.append(match_pred)
        
        # 2. Prédiction batch pour la journée
        if X_features:
            X_array = np.array(X_features)
            
            try:
                # Prédictions
                y_pred = self.model.predict(X_array)
                y_proba = self.model.predict_proba(X_array)
                
                # Remplir les prédictions
                for i, pred in enumerate(predictions):
                    pred.predicted_result = int(y_pred[i])
                    pred.predicted_proba = y_proba[i].tolist()
                
            except Exception as e:
                self.logger.error(f"Prediction failed for {date}: {e}")
                # Prédictions par défaut (équiprobables)
                for pred in predictions:
                    pred.predicted_result = 0  # Home par défaut
                    pred.predicted_proba = [0.33, 0.33, 0.34]
        
        # 3. Calcul métriques si résultats disponibles
        accuracy = None
        log_loss_score = None
        
        actual_results = [p.actual_result for p in predictions if p.actual_result is not None]
        predicted_results = [p.predicted_result for p in predictions if p.actual_result is not None]
        predicted_probas = [p.predicted_proba for p in predictions if p.actual_result is not None]
        
        if actual_results:
            accuracy = accuracy_score(actual_results, predicted_results)
            
            if predicted_probas:
                try:
                    log_loss_score = log_loss(actual_results, predicted_probas)
                except:
                    log_loss_score = None
        
        return MatchDayResults(
            date=str(date),
            match_day=str(match_day),
            predictions=predictions,
            accuracy=accuracy,
            log_loss_score=log_loss_score
        )
    
    def update_state_after_matchday(self, matchday_results: MatchDayResults, 
                                   matchday_data: pd.DataFrame, 
                                   state_manager: StateManager):
        """Met à jour l'état après une journée avec les VRAIS résultats"""
        for idx, match_row in matchday_data.iterrows():
            if 'HomeGoals' in match_row and 'AwayGoals' in match_row:
                # Utiliser les vrais résultats pour mise à jour
                state_manager.process_match(match_row)

class RollingSimulationEngine:
    """Moteur principal de simulation rolling"""
    
    def __init__(self, initial_state: StateManager, model, 
                 feature_builder: FeatureBuilder):
        self.state_manager = initial_state
        self.model = model
        self.feature_builder = feature_builder
        self.match_day_processor = MatchDayProcessor(model, feature_builder)
        self.logger = setup_logging()
        self.results_history = []
    
    def load_test_data(self, test_data_path: str, test_start: str, 
                      test_end: Optional[str] = None) -> pd.DataFrame:
        """Charge les données de test dans la période spécifiée"""
        self.logger.info(f"📊 LOADING TEST DATA")
        self.logger.info(f"Test data: {test_data_path}")
        self.logger.info(f"Period: {test_start} to {test_end or 'end'}")
        
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"❌ Test data not found: {test_data_path}")
        
        df = pd.read_csv(test_data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filtrer période de test
        start_date = pd.to_datetime(test_start)
        df = df[df['Date'] >= start_date]
        
        if test_end:
            end_date = pd.to_datetime(test_end)
            df = df[df['Date'] <= end_date]
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        self.logger.info(f"✅ Test data loaded:")
        self.logger.info(f"  Matches: {len(df)}")
        self.logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        self.logger.info(f"  Teams: {df['HomeTeam'].nunique()}")
        
        return df
    
    def run_simulation(self, test_data: pd.DataFrame) -> List[MatchDayResults]:
        """Exécute la simulation rolling complète"""
        self.logger.info(f"🎮 STARTING ROLLING SIMULATION")
        self.logger.info("=" * 60)
        
        # Grouper par date pour traitement journée par journée
        grouped = test_data.groupby('Date')
        total_matchdays = len(grouped)
        
        self.logger.info(f"Processing {total_matchdays} match days...")
        
        all_results = []
        
        for date, matchday_data in grouped:
            self.logger.info(f"📅 Processing {date} ({len(matchday_data)} matches)")
            
            # 1. Prédire tous les matchs de la journée
            matchday_results = self.match_day_processor.process_matchday(
                matchday_data, self.state_manager
            )
            
            # 2. Mettre à jour l'état avec les VRAIS résultats
            self.match_day_processor.update_state_after_matchday(
                matchday_results, matchday_data, self.state_manager
            )
            
            all_results.append(matchday_results)
            
            # Log périodique
            if matchday_results.accuracy is not None:
                self.logger.info(f"  ✅ {date}: {matchday_results.accuracy:.3f} accuracy")
            else:
                self.logger.info(f"  ⚠️  {date}: No results available")
        
        self.results_history = all_results
        
        self.logger.info(f"🎯 SIMULATION COMPLETE")
        self.logger.info(f"Processed {total_matchdays} match days")
        
        return all_results

class ResultsAggregator:
    """Agrégation et analyse des résultats"""
    
    def __init__(self):
        self.logger = setup_logging()
    
    def aggregate_results(self, simulation_results: List[MatchDayResults]) -> Dict[str, Any]:
        """Agrège tous les résultats de simulation"""
        all_predictions = []
        daily_accuracies = []
        daily_log_losses = []
        
        # Collecter toutes les prédictions
        for matchday_result in simulation_results:
            all_predictions.extend(matchday_result.predictions)
            
            if matchday_result.accuracy is not None:
                daily_accuracies.append(matchday_result.accuracy)
            
            if matchday_result.log_loss_score is not None:
                daily_log_losses.append(matchday_result.log_loss_score)
        
        # Filtrer prédictions avec résultats réels
        valid_predictions = [p for p in all_predictions if p.actual_result is not None]
        
        if not valid_predictions:
            return {'error': 'No valid predictions found'}
        
        # Métriques globales
        y_true = [p.actual_result for p in valid_predictions]
        y_pred = [p.predicted_result for p in valid_predictions] 
        y_proba = [p.predicted_proba for p in valid_predictions]
        
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        try:
            overall_log_loss = log_loss(y_true, y_proba)
        except:
            overall_log_loss = None
        
        # Rapport de classification
        class_report = classification_report(
            y_true, y_pred, 
            target_names=['Home', 'Draw', 'Away'],
            output_dict=True
        )
        
        # Distribution des résultats
        result_distribution = {
            'actual': {
                'home': sum(1 for r in y_true if r == 0) / len(y_true),
                'draw': sum(1 for r in y_true if r == 1) / len(y_true),
                'away': sum(1 for r in y_true if r == 2) / len(y_true)
            },
            'predicted': {
                'home': sum(1 for r in y_pred if r == 0) / len(y_pred),
                'draw': sum(1 for r in y_pred if r == 1) / len(y_pred),
                'away': sum(1 for r in y_pred if r == 2) / len(y_pred)
            }
        }
        
        # Calcul baselines
        majority_baseline = max(
            result_distribution['actual']['home'],
            result_distribution['actual']['draw'],
            result_distribution['actual']['away']
        )
        
        random_baseline = 1/3
        
        return {
            'summary': {
                'total_matches': len(valid_predictions),
                'total_matchdays': len([r for r in simulation_results if r.accuracy is not None]),
                'overall_accuracy': overall_accuracy,
                'overall_log_loss': overall_log_loss,
                'mean_daily_accuracy': np.mean(daily_accuracies) if daily_accuracies else None,
                'std_daily_accuracy': np.std(daily_accuracies) if daily_accuracies else None
            },
            'baselines': {
                'random': random_baseline,
                'majority_class': majority_baseline,
                'good_target': 0.50,
                'excellent_target': 0.55
            },
            'performance_vs_baselines': {
                'vs_random': overall_accuracy - random_baseline,
                'vs_majority': overall_accuracy - majority_baseline,
                'vs_good': overall_accuracy - 0.50,
                'vs_excellent': overall_accuracy - 0.55
            },
            'classification_report': class_report,
            'result_distribution': result_distribution,
            'daily_performance': {
                'accuracies': daily_accuracies,
                'log_losses': daily_log_losses,
                'dates': [r.date for r in simulation_results if r.accuracy is not None]
            }
        }
    
    def save_results(self, aggregated_results: Dict[str, Any], output_path: str):
        """Sauvegarde les résultats agrégés"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        self.logger.info(f"✅ Results saved to: {output_path}")

def main():
    """Pipeline principal de simulation rolling"""
    parser = argparse.ArgumentParser(description="Rolling simulation engine for football prediction")
    parser.add_argument('--initial_state', required=True, help='Initial state pickle file')
    parser.add_argument('--test_data', required=True, help='Test data CSV file')
    parser.add_argument('--test_start', required=True, help='Test period start (YYYY-MM-DD)')
    parser.add_argument('--test_end', help='Test period end (YYYY-MM-DD)')
    parser.add_argument('--model', required=True, help='Trained model file')
    parser.add_argument('--output', required=True, help='Output results JSON file')
    parser.add_argument('--feature_config', help='Optional feature configuration JSON')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🎮 ROLLING SIMULATION ENGINE")
    logger.info("=" * 80)
    logger.info(f"Initial state: {args.initial_state}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Test period: {args.test_start} to {args.test_end or 'end'}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)
    
    try:
        # 1. Charger état initial
        logger.info("📋 Loading initial state...")
        state_manager = load_state(args.initial_state)
        
        # 2. Charger modèle
        logger.info("🤖 Loading model...")
        model = joblib.load(args.model)
        
        # 3. Configuration features
        feature_config = None
        if args.feature_config and os.path.exists(args.feature_config):
            with open(args.feature_config, 'r') as f:
                feature_config = json.load(f)
        
        feature_builder = FeatureBuilder(feature_config)
        
        # 4. Initialiser moteur de simulation
        engine = RollingSimulationEngine(state_manager, model, feature_builder)
        
        # 5. Charger données de test
        test_data = engine.load_test_data(args.test_data, args.test_start, args.test_end)
        
        # 6. Exécuter simulation
        simulation_results = engine.run_simulation(test_data)
        
        # 7. Agréger résultats
        logger.info("📊 Aggregating results...")
        aggregator = ResultsAggregator()
        aggregated_results = aggregator.aggregate_results(simulation_results)
        
        # 8. Sauvegarder
        aggregator.save_results(aggregated_results, args.output)
        
        # 9. Résumé final
        logger.info("🎯 SIMULATION RESULTS:")
        if 'summary' in aggregated_results:
            summary = aggregated_results['summary']
            logger.info(f"  Total matches: {summary['total_matches']}")
            logger.info(f"  Overall accuracy: {summary['overall_accuracy']:.4f}")
            
            if 'performance_vs_baselines' in aggregated_results:
                perf = aggregated_results['performance_vs_baselines']
                logger.info(f"  vs Random: {perf['vs_random']:+.4f}")
                logger.info(f"  vs Majority: {perf['vs_majority']:+.4f}")
                logger.info(f"  vs Good (50%): {perf['vs_good']:+.4f}")
                logger.info(f"  vs Excellent (55%): {perf['vs_excellent']:+.4f}")
        
        logger.info("🎉 ROLLING SIMULATION COMPLETE!")
        
    except Exception as e:
        logger.error(f"❌ SIMULATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()