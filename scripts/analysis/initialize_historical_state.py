#!/usr/bin/env python3
"""
initialize_historical_state.py

Correcte initialisation de l'état historique pour simulation rolling temporellement sûre.

OBJECTIF CRITIQUE:
- Construire Elo, Rolling stats, H2H avec SEULEMENT les données disponibles avant season_start_date
- Éviter tout data leakage temporel
- Replay complet de l'historique match par match
- État initial prêt pour simulation rolling réaliste

USAGE:
python scripts/analysis/initialize_historical_state.py --season_start 2024-08-01 --output_state state_2024_season.pkl

ARCHITECTURE:
1. HistoricalStateBuilder - Replay match par match de l'historique
2. TeamState - État complet d'une équipe (Elo, rolling, stats)
3. StateManager - Gestion globale de l'état (teams + H2H)
4. Validation temporelle - Vérification zéro leakage
"""

import sys
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class RobustDataHandler:
    """
    AMÉLIORATION CRITIQUE: Gestion robuste des données manquantes
    
    Au lieu de fallbacks arbitraires, calcule des valeurs basées sur:
    1. Moyennes historiques par équipe
    2. Moyennes de la ligue par période
    3. Fallbacks intelligents par position dans la ligue
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        self.logger = setup_logging()
        self.historical_data = historical_data.copy()
        self.team_historical_stats = {}
        self.league_averages = {}
        
        self._compute_historical_baselines()
    
    def _compute_historical_baselines(self):
        """Calcule les moyennes historiques par équipe et ligue"""
        self.logger.info("📊 COMPUTING ROBUST DATA BASELINES")
        
        # Construire dict d'agrégation dynamique
        agg_dict = {'HomeGoals': ['mean', 'std', 'count']}
        
        # Ajouter colonnes disponibles
        optional_cols = ['HomeShots', 'HomeCorners', 'HomeXG']
        for col in optional_cols:
            if col in self.historical_data.columns:
                agg_dict[col] = ['mean', 'std']
        
        # 1. Stats par équipe (Home)
        home_stats = self.historical_data.groupby('HomeTeam').agg(agg_dict).fillna(0)
        
        # Même logique pour Away
        agg_dict_away = {'AwayGoals': ['mean', 'std', 'count']}
        optional_cols_away = ['AwayShots', 'AwayCorners', 'AwayXG']
        for col in optional_cols_away:
            if col in self.historical_data.columns:
                agg_dict_away[col] = ['mean', 'std']
        
        # 2. Stats par équipe (Away) 
        away_stats = self.historical_data.groupby('AwayTeam').agg(agg_dict_away).fillna(0)
        
        # 3. Combiner Home + Away par équipe
        all_teams = set(self.historical_data['HomeTeam'].unique()) | set(self.historical_data['AwayTeam'].unique())
        
        for team in all_teams:
            team_stats = {}
            
            # Goals
            home_goals = home_stats.loc[team, ('HomeGoals', 'mean')] if team in home_stats.index else 0
            away_goals = away_stats.loc[team, ('AwayGoals', 'mean')] if team in away_stats.index else 0
            home_games = home_stats.loc[team, ('HomeGoals', 'count')] if team in home_stats.index else 0
            away_games = away_stats.loc[team, ('AwayGoals', 'count')] if team in away_stats.index else 0
            
            if home_games + away_games > 0:
                team_stats['goals'] = (home_goals * home_games + away_goals * away_games) / (home_games + away_games)
            else:
                team_stats['goals'] = 1.2  # Fallback
                
            # XG si disponible
            if 'HomeXG' in self.historical_data.columns:
                home_xg = home_stats.loc[team, ('HomeXG', 'mean')] if team in home_stats.index else 0
                away_xg = away_stats.loc[team, ('AwayXG', 'mean')] if team in away_stats.index else 0
                
                if home_games + away_games > 0:
                    team_stats['xg'] = (home_xg * home_games + away_xg * away_games) / (home_games + away_games)
                else:
                    team_stats['xg'] = 1.3
            
            # Shots si disponible
            if 'HomeShots' in self.historical_data.columns:
                home_shots = home_stats.loc[team, ('HomeShots', 'mean')] if team in home_stats.index else 0
                away_shots = away_stats.loc[team, ('AwayShots', 'mean')] if team in away_stats.index else 0
                
                if home_games + away_games > 0:
                    team_stats['shots'] = (home_shots * home_games + away_shots * away_games) / (home_games + away_games)
                else:
                    team_stats['shots'] = 12.0
            
            # Corners si disponible  
            if 'HomeCorners' in self.historical_data.columns:
                home_corners = home_stats.loc[team, ('HomeCorners', 'mean')] if team in home_stats.index else 0
                away_corners = away_stats.loc[team, ('AwayCorners', 'mean')] if team in away_stats.index else 0
                
                if home_games + away_games > 0:
                    team_stats['corners'] = (home_corners * home_games + away_corners * away_games) / (home_games + away_games)
                else:
                    team_stats['corners'] = 5.0
            
            team_stats['total_games'] = home_games + away_games
            self.team_historical_stats[team] = team_stats
        
        # 4. Moyennes de la ligue comme fallback ultime
        self.league_averages = {
            'goals': self.historical_data[['HomeGoals', 'AwayGoals']].mean().mean(),
            'xg': self.historical_data[['HomeXG', 'AwayXG']].mean().mean() if 'HomeXG' in self.historical_data.columns else 1.3,
            'shots': self.historical_data[['HomeShots', 'AwayShots']].mean().mean() if 'HomeShots' in self.historical_data.columns else 12.0,
            'corners': self.historical_data[['HomeCorners', 'AwayCorners']].mean().mean() if 'HomeCorners' in self.historical_data.columns else 5.0
        }
        
        self.logger.info(f"✅ Baselines computed for {len(self.team_historical_stats)} teams")
        self.logger.info(f"📊 League averages: Goals={self.league_averages['goals']:.2f}, xG={self.league_averages['xg']:.2f}")
    
    def get_robust_value(self, team_name: str, metric: str, fallback_context: Optional[str] = None) -> float:
        """
        Récupère une valeur robuste pour une équipe/métrique
        
        Priorité:
        1. Moyenne historique de l'équipe
        2. Moyenne de la ligue 
        3. Fallback par défaut
        """
        
        # 1. Essayer moyenne historique équipe
        if team_name in self.team_historical_stats:
            team_stats = self.team_historical_stats[team_name]
            if metric in team_stats and team_stats['total_games'] >= 5:  # Min 5 matchs
                return team_stats[metric]
        
        # 2. Moyenne de la ligue
        if metric in self.league_averages:
            return self.league_averages[metric]
        
        # 3. Fallback par défaut (dernier recours)
        defaults = {
            'goals': 1.2,
            'xg': 1.3,
            'shots': 12.0,
            'corners': 5.0,
            'form': 1.0
        }
        
        return defaults.get(metric, 0.0)
    
    def get_team_quality_adjustment(self, team_name: str) -> float:
        """
        Ajustement basé sur la qualité historique de l'équipe
        Retourne multiplicateur [0.7, 1.3]
        """
        if team_name not in self.team_historical_stats:
            return 1.0
        
        team_stats = self.team_historical_stats[team_name]
        if team_stats['total_games'] < 10:
            return 1.0
            
        # Calculer qualité relative basée sur goals scored vs league average
        team_goals_avg = team_stats['goals']
        league_goals_avg = self.league_averages['goals']
        
        if league_goals_avg > 0:
            quality_ratio = team_goals_avg / league_goals_avg
            # Clipper dans une range raisonnable
            return np.clip(quality_ratio, 0.7, 1.3)
        
        return 1.0

# Configuration par défaut
DEFAULT_ELO_INIT = 1500.0
DEFAULT_ELO_K = 32
DEFAULT_ROLLING_WINDOWS = {
    'form': 5,         # Points des 5 derniers matchs
    'goals': 5,        # Buts des 5 derniers matchs  
    'xg': 5,           # xG des 5 derniers matchs
    'shots': 5,        # Tirs des 5 derniers matchs
    'corners': 5,      # Corners des 5 derniers matchs
    'goals_10': 10,    # Fenêtre 10 matchs pour goals
    'xg_10': 10        # Fenêtre 10 matchs pour xG
}

@dataclass
class TeamState:
    """État complet d'une équipe à un moment T"""
    team_name: str
    elo: float = DEFAULT_ELO_INIT
    games_played: int = 0
    
    # Rolling stats par fenêtre
    rolling_data: Dict[str, deque] = field(default_factory=dict)
    
    # Stats cumulés
    total_points: int = 0
    total_goals_for: int = 0
    total_goals_against: int = 0
    total_wins: int = 0
    total_draws: int = 0
    total_losses: int = 0
    
    def __post_init__(self):
        """Initialise les deques pour rolling stats"""
        if not self.rolling_data:
            for metric, window in DEFAULT_ROLLING_WINDOWS.items():
                self.rolling_data[metric] = deque(maxlen=window)
    
    def get_rolling_mean(self, metric: str, data_handler: Optional['RobustDataHandler'] = None) -> float:
        """Calcule moyenne des N derniers matchs pour une métrique"""
        if metric not in self.rolling_data or len(self.rolling_data[metric]) == 0:
            return self._get_default_value(metric, data_handler)
        
        values = list(self.rolling_data[metric])
        return np.mean(values) if values else self._get_default_value(metric, data_handler)
    
    def get_rolling_sum(self, metric: str) -> float:
        """Calcule somme des N derniers matchs pour une métrique"""
        if metric not in self.rolling_data or len(self.rolling_data[metric]) == 0:
            return 0.0
        
        values = list(self.rolling_data[metric])
        return np.sum(values) if values else 0.0
    
    def _get_default_value(self, metric: str, data_handler: Optional['RobustDataHandler'] = None) -> float:
        """
        AMÉLIORATION CRITIQUE: Valeurs par défaut intelligentes
        
        Utilise RobustDataHandler si disponible, sinon fallbacks classiques
        """
        if data_handler:
            return data_handler.get_robust_value(self.team_name, metric)
        
        # Fallbacks classiques si pas de data handler
        defaults = {
            'form': 1.0,       # ~1 point par match
            'goals': 1.2,      # ~1.2 buts par match
            'goals_10': 1.2,
            'xg': 1.3,         # ~1.3 xG par match  
            'xg_10': 1.3,
            'shots': 12.0,     # ~12 tirs par match
            'corners': 5.0     # ~5 corners par match
        }
        return defaults.get(metric, 0.0)
    
    def update_after_match(self, is_home: bool, home_goals: int, away_goals: int, 
                          match_data: Dict[str, Any]):
        """Met à jour l'état après un match"""
        # Calcul du résultat
        if is_home:
            goals_for = home_goals
            goals_against = away_goals
        else:
            goals_for = away_goals
            goals_against = home_goals
        
        # Points
        if goals_for > goals_against:
            points = 3
            self.total_wins += 1
        elif goals_for == goals_against:
            points = 1
            self.total_draws += 1
        else:
            points = 0
            self.total_losses += 1
        
        # Mise à jour stats cumulées
        self.games_played += 1
        self.total_points += points
        self.total_goals_for += goals_for
        self.total_goals_against += goals_against
        
        # Mise à jour rolling data
        self.rolling_data['form'].append(points)
        self.rolling_data['goals'].append(goals_for)
        self.rolling_data['goals_10'].append(goals_for)
        
        # xG avec gestion robuste des données manquantes
        xg_value = None
        if is_home and 'HomeXG' in match_data and pd.notna(match_data['HomeXG']):
            xg_value = match_data['HomeXG']
        elif not is_home and 'AwayXG' in match_data and pd.notna(match_data['AwayXG']):
            xg_value = match_data['AwayXG']
        
        if xg_value is not None:
            self.rolling_data['xg'].append(xg_value)
            self.rolling_data['xg_10'].append(xg_value)
        # Si pas de xG disponible, on ne force pas de valeur - rolling window s'adaptera
        
        # Autres stats avec gestion robuste
        if is_home:
            if 'HomeShots' in match_data and pd.notna(match_data['HomeShots']):
                self.rolling_data['shots'].append(match_data['HomeShots'])
            if 'HomeCorners' in match_data and pd.notna(match_data['HomeCorners']):
                self.rolling_data['corners'].append(match_data['HomeCorners'])
        else:
            if 'AwayShots' in match_data and pd.notna(match_data['AwayShots']):
                self.rolling_data['shots'].append(match_data['AwayShots'])
            if 'AwayCorners' in match_data and pd.notna(match_data['AwayCorners']):
                self.rolling_data['corners'].append(match_data['AwayCorners'])

class StateManager:
    """Gestionnaire global de l'état - Teams + H2H"""
    
    def __init__(self, elo_k: float = DEFAULT_ELO_K, data_handler: Optional['RobustDataHandler'] = None):
        self.teams: Dict[str, TeamState] = {}
        self.h2h: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(self._default_h2h_dict)
        self.elo_k = elo_k
        self.processed_matches = 0
        self.last_update_date = None
        self.data_handler = data_handler  # AMÉLIORATION: Handler pour données manquantes
    
    def _default_h2h_dict(self):
        """Factory function for H2H default values (pickle-safe)"""
        return {'home_wins': 0, 'away_wins': 0, 'draws': 0, 'total_games': 0}
    
    def get_or_create_team(self, team_name: str) -> TeamState:
        """Récupère ou crée l'état d'une équipe"""
        if team_name not in self.teams:
            self.teams[team_name] = TeamState(team_name=team_name)
        return self.teams[team_name]
    
    def get_h2h_key(self, home_team: str, away_team: str) -> Tuple[str, str]:
        """Clé H2H normalisée (ordre alphabétique)"""
        return tuple(sorted([home_team, away_team]))
    
    def get_h2h_stats(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Statistiques H2H pour un match"""
        key = self.get_h2h_key(home_team, away_team)
        h2h_data = self.h2h[key]
        
        if h2h_data['total_games'] == 0:
            return {'h2h_score': 0.5}  # Neutre si pas d'historique
        
        # Score H2H pour l'équipe à domicile
        if key[0] == home_team:
            home_advantage = h2h_data['home_wins'] / h2h_data['total_games']
        else:
            home_advantage = h2h_data['away_wins'] / h2h_data['total_games']
        
        return {'h2h_score': home_advantage}
    
    def update_elo(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        """Met à jour les ratings Elo après un match"""
        home_state = self.get_or_create_team(home_team)
        away_state = self.get_or_create_team(away_team)
        
        # Résultat du match (perspective équipe domicile)
        if home_goals > away_goals:
            result = 1.0  # Victoire domicile
        elif home_goals == away_goals:
            result = 0.5  # Match nul
        else:
            result = 0.0  # Victoire extérieur
        
        # Probabilité attendue
        expected_home = 1 / (1 + 10**((away_state.elo - home_state.elo) / 400))
        
        # Mise à jour Elo
        home_state.elo += self.elo_k * (result - expected_home)
        away_state.elo += self.elo_k * ((1 - result) - (1 - expected_home))
    
    def update_h2h(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        """Met à jour les statistiques H2H"""
        key = self.get_h2h_key(home_team, away_team)
        h2h_data = self.h2h[key]
        
        h2h_data['total_games'] += 1
        
        if home_goals > away_goals:
            if key[0] == home_team:
                h2h_data['home_wins'] += 1
            else:
                h2h_data['away_wins'] += 1
        elif home_goals == away_goals:
            h2h_data['draws'] += 1
        else:
            if key[0] == away_team:
                h2h_data['home_wins'] += 1
            else:
                h2h_data['away_wins'] += 1
    
    def process_match(self, match_row: pd.Series):
        """Traite un match et met à jour tout l'état"""
        home_team = match_row['HomeTeam']
        away_team = match_row['AwayTeam']
        home_goals = int(match_row['HomeGoals'])
        away_goals = int(match_row['AwayGoals'])
        match_date = match_row['Date']
        
        # Créer ou récupérer les états des équipes
        home_state = self.get_or_create_team(home_team)
        away_state = self.get_or_create_team(away_team)
        
        # Préparer les données du match
        match_data = match_row.to_dict()
        
        # Mise à jour des équipes (AVANT Elo pour cohérence)
        home_state.update_after_match(True, home_goals, away_goals, match_data)
        away_state.update_after_match(False, home_goals, away_goals, match_data)
        
        # Mise à jour Elo
        self.update_elo(home_team, away_team, home_goals, away_goals)
        
        # Mise à jour H2H
        self.update_h2h(home_team, away_team, home_goals, away_goals)
        
        # Suivi
        self.processed_matches += 1
        self.last_update_date = match_date

class HistoricalStateBuilder:
    """Constructeur d'état historique avec validation temporelle"""
    
    def __init__(self, data_path: str, elo_k: float = DEFAULT_ELO_K):
        self.data_path = data_path
        self.elo_k = elo_k
        self.logger = setup_logging()
    
    def load_and_validate_data(self, season_start_date: str) -> pd.DataFrame:
        """Charge et valide les données historiques"""
        self.logger.info(f"📊 LOADING HISTORICAL DATA")
        self.logger.info(f"Data path: {self.data_path}")
        self.logger.info(f"Season start: {season_start_date}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Dataset not found: {self.data_path}")
        
        # Charger données
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filtrer SEULEMENT les matchs avant season_start_date
        cutoff_date = pd.to_datetime(season_start_date)
        historical_df = df[df['Date'] < cutoff_date].copy()
        
        # Gestion des noms de colonnes différents
        column_mappings = {
            'FullTimeHomeTeamGoals': 'HomeGoals',
            'FullTimeAwayTeamGoals': 'AwayGoals'
        }
        
        # Appliquer les mappings si nécessaire
        for old_col, new_col in column_mappings.items():
            if old_col in historical_df.columns and new_col not in historical_df.columns:
                historical_df[new_col] = historical_df[old_col]
                self.logger.info(f"✅ Mapped column: {old_col} → {new_col}")
        
        # Si pas de HomeGoals/AwayGoals, merger depuis raw data
        if 'HomeGoals' not in historical_df.columns:
            self.logger.info("🔗 HomeGoals missing, merging from raw data...")
            raw_df = pd.read_csv('data/raw/premier_league_2019_2024.csv', parse_dates=['Date'])
            
            # Créer merge key unique
            historical_df['merge_key'] = historical_df['Date'].dt.strftime('%Y-%m-%d') + '_' + historical_df['HomeTeam'] + '_' + historical_df['AwayTeam']
            raw_df['merge_key'] = raw_df['Date'].dt.strftime('%Y-%m-%d') + '_' + raw_df['HomeTeam'] + '_' + raw_df['AwayTeam']
            
            # Merger les colonnes buts
            goals_df = raw_df[['merge_key', 'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals']].rename(columns={
                'FullTimeHomeTeamGoals': 'HomeGoals',
                'FullTimeAwayTeamGoals': 'AwayGoals'
            })
            
            historical_df = historical_df.merge(goals_df, on='merge_key', how='left')
            historical_df.drop('merge_key', axis=1, inplace=True)
            self.logger.info(f"✅ Merged goal data from raw dataset")
        
        # Validation des colonnes requises
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']
        missing_cols = [col for col in required_cols if col not in historical_df.columns]
        
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")
        
        # Tri chronologique STRICT
        historical_df = historical_df.sort_values('Date').reset_index(drop=True)
        
        self.logger.info(f"✅ Historical data loaded:")
        self.logger.info(f"  Total matches: {len(historical_df)}")
        self.logger.info(f"  Date range: {historical_df['Date'].min()} to {historical_df['Date'].max()}")
        self.logger.info(f"  Teams: {historical_df['HomeTeam'].nunique()}")
        
        # Validation temporelle critique
        if historical_df['Date'].max() >= cutoff_date:
            raise ValueError(f"❌ TEMPORAL LEAKAGE DETECTED! Some matches are >= {cutoff_date}")
        
        return historical_df
    
    def build_initial_state(self, season_start_date: str, 
                           use_elo_carry_over: bool = True, 
                           use_robust_data_handling: bool = True) -> StateManager:
        """Construit l'état initial par replay complet avec carry-over Elo intelligent"""
        self.logger.info(f"🏗️  BUILDING INITIAL STATE")
        self.logger.info("=" * 60)
        
        # Charger données historiques
        historical_df = self.load_and_validate_data(season_start_date)
        
        # AMÉLIORATION CRITIQUE: Initialiser handler pour données manquantes
        data_handler = None
        if use_robust_data_handling and len(historical_df) > 0:
            self.logger.info("🛠️ INITIALIZING ROBUST DATA HANDLER")
            data_handler = RobustDataHandler(historical_df)
        
        # Initialiser StateManager avec data handler
        state_manager = StateManager(elo_k=self.elo_k, data_handler=data_handler)
        
        if use_elo_carry_over and len(historical_df) > 0:
            # AMÉLIORATION CRITIQUE: Initialisation Elo intelligente
            self.logger.info("🧠 SMART ELO INITIALIZATION")
            state_manager = self._initialize_elo_from_recent_performance(
                state_manager, historical_df, season_start_date
            )
        
        # Replay match par match
        self.logger.info(f"🔄 REPLAYING {len(historical_df)} MATCHES...")
        
        for idx, match_row in historical_df.iterrows():
            state_manager.process_match(match_row)
            
            # Log périodique
            if (idx + 1) % 500 == 0:
                self.logger.info(f"  Processed {idx + 1}/{len(historical_df)} matches...")
        
        self.logger.info(f"✅ REPLAY COMPLETE")
        self.logger.info(f"  Matches processed: {state_manager.processed_matches}")
        self.logger.info(f"  Teams in state: {len(state_manager.teams)}")
        self.logger.info(f"  H2H pairs: {len(state_manager.h2h)}")
        self.logger.info(f"  Last update: {state_manager.last_update_date}")
        
        return state_manager
    
    def _initialize_elo_from_recent_performance(self, state_manager: StateManager, 
                                              historical_df: pd.DataFrame, 
                                              season_start_date: str) -> StateManager:
        """
        AMÉLIORATION CRITIQUE: Initialise Elo basé sur performance récente
        
        Stratégie:
        1. Identifier dernière saison complète avant season_start_date
        2. Calculer Elo final de cette saison pour chaque équipe
        3. Initialiser StateManager avec ces valeurs réalistes
        4. Gérer équipes promues/nouvelles avec Elo ajusté
        """
        cutoff_date = pd.to_datetime(season_start_date)
        
        # 1. Identifier période de réchauffement (derniers N mois)
        lookback_months = 18  # Regarder 18 mois en arrière max
        lookback_date = cutoff_date - pd.DateOffset(months=lookback_months)
        
        recent_matches = historical_df[historical_df['Date'] >= lookback_date].copy()
        
        if len(recent_matches) == 0:
            self.logger.warning("⚠️ No recent matches found for Elo initialization")
            return state_manager
        
        self.logger.info(f"📊 Elo initialization from {len(recent_matches)} recent matches")
        self.logger.info(f"Period: {recent_matches['Date'].min()} to {recent_matches['Date'].max()}")
        
        # 2. Simuler Elo sur période récente avec StateManager temporaire
        temp_state = StateManager(elo_k=self.elo_k)
        
        for _, match_row in recent_matches.iterrows():
            temp_state.process_match(match_row)
        
        # 3. Extraire Elo finaux et statistiques
        team_final_elos = {}
        team_recent_performance = {}
        
        for team_name, team_state in temp_state.teams.items():
            team_final_elos[team_name] = team_state.elo
            
            # Calculer performance récente pour validation
            if team_state.games_played > 0:
                win_rate = team_state.total_wins / team_state.games_played
                points_per_game = team_state.total_points / team_state.games_played
                team_recent_performance[team_name] = {
                    'games': team_state.games_played,
                    'win_rate': win_rate,
                    'ppg': points_per_game,
                    'final_elo': team_state.elo
                }
        
        # 4. Valider cohérence des Elo calculés
        elo_values = list(team_final_elos.values())
        if elo_values:
            mean_elo = np.mean(elo_values)
            std_elo = np.std(elo_values)
            
            self.logger.info(f"📈 Computed Elo statistics:")
            self.logger.info(f"  Teams: {len(team_final_elos)}")
            self.logger.info(f"  Mean Elo: {mean_elo:.1f}")
            self.logger.info(f"  Std Elo: {std_elo:.1f}")
            self.logger.info(f"  Range: {min(elo_values):.1f} - {max(elo_values):.1f}")
            
            # Top 5 et Bottom 5 pour sanity check
            sorted_teams = sorted(team_final_elos.items(), key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"🏆 Top 5 teams by Elo:")
            for i, (team, elo) in enumerate(sorted_teams[:5]):
                self.logger.info(f"  {i+1}. {team}: {elo:.1f}")
            
            self.logger.info(f"📉 Bottom 5 teams by Elo:")
            for i, (team, elo) in enumerate(sorted_teams[-5:]):
                self.logger.info(f"  {len(sorted_teams)-4+i}. {team}: {elo:.1f}")
        
        # 5. Initialiser StateManager principal avec Elo calculés
        for team_name, final_elo in team_final_elos.items():
            team_state = state_manager.get_or_create_team(team_name)
            team_state.elo = final_elo
            
            self.logger.debug(f"Initialized {team_name} with Elo {final_elo:.1f}")
        
        # 6. Gérer équipes qui apparaîtront dans test mais pas dans historique récent
        # (équipes promues, renommées, etc.)
        all_historical_teams = set(historical_df['HomeTeam'].unique()) | set(historical_df['AwayTeam'].unique())
        teams_with_elo = set(team_final_elos.keys())
        missing_teams = all_historical_teams - teams_with_elo
        
        if missing_teams:
            # Elo par défaut pour équipes manquantes (légèrement en dessous de la moyenne)
            default_elo = mean_elo - 0.5 * std_elo if elo_values else DEFAULT_ELO_INIT
            
            self.logger.info(f"⚠️ {len(missing_teams)} teams missing recent data:")
            for team in list(missing_teams)[:5]:  # Log premiers 5 seulement
                team_state = state_manager.get_or_create_team(team)
                team_state.elo = default_elo
                self.logger.info(f"  {team}: default Elo {default_elo:.1f}")
            
            if len(missing_teams) > 5:
                self.logger.info(f"  ... and {len(missing_teams) - 5} more")
        
        self.logger.info(f"✅ Smart Elo initialization complete")
        self.logger.info(f"  {len(team_final_elos)} teams with computed Elo")
        self.logger.info(f"  {len(missing_teams)} teams with default Elo")
        
        return state_manager
    
    def validate_state_integrity(self, state_manager: StateManager, 
                                season_start_date: str) -> bool:
        """Validation finale de l'intégrité de l'état"""
        self.logger.info(f"🔍 VALIDATING STATE INTEGRITY")
        
        errors = []
        warnings = []
        
        # 1. Validation Elo
        for team_name, team_state in state_manager.teams.items():
            if team_state.elo < 800 or team_state.elo > 2200:
                warnings.append(f"Unusual Elo for {team_name}: {team_state.elo:.1f}")
            
            if team_state.games_played == 0:
                warnings.append(f"Team {team_name} has no matches")
        
        # 2. Validation temporelle
        cutoff_date = pd.to_datetime(season_start_date)
        if state_manager.last_update_date and state_manager.last_update_date >= cutoff_date:
            errors.append(f"Last update date {state_manager.last_update_date} >= cutoff {cutoff_date}")
        
        # 3. Validation cohérence rolling
        for team_name, team_state in state_manager.teams.items():
            for metric, deque_data in team_state.rolling_data.items():
                if len(deque_data) > DEFAULT_ROLLING_WINDOWS.get(metric, 10):
                    errors.append(f"Rolling deque too large for {team_name}.{metric}")
        
        # Rapport
        if errors:
            self.logger.error(f"❌ VALIDATION FAILED:")
            for error in errors:
                self.logger.error(f"  {error}")
            return False
        
        if warnings:
            self.logger.warning(f"⚠️  VALIDATION WARNINGS:")
            for warning in warnings:
                self.logger.warning(f"  {warning}")
        
        self.logger.info(f"✅ STATE VALIDATION PASSED")
        return True
    
    def generate_state_summary(self, state_manager: StateManager) -> Dict[str, Any]:
        """Génère un résumé de l'état pour debugging"""
        teams_elo = {name: state.elo for name, state in state_manager.teams.items()}
        
        summary = {
            'total_teams': len(state_manager.teams),
            'total_matches_processed': state_manager.processed_matches,
            'h2h_pairs': len(state_manager.h2h),
            'last_update_date': str(state_manager.last_update_date),
            'elo_stats': {
                'mean': np.mean(list(teams_elo.values())),
                'std': np.std(list(teams_elo.values())),
                'min': min(teams_elo.values()),
                'max': max(teams_elo.values()),
                'top_5': dict(sorted(teams_elo.items(), key=lambda x: x[1], reverse=True)[:5])
            },
            'sample_team_state': None
        }
        
        # Ajouter un exemple d'état d'équipe
        if state_manager.teams:
            sample_team = list(state_manager.teams.values())[0]
            summary['sample_team_state'] = {
                'name': sample_team.team_name,
                'elo': sample_team.elo,
                'games_played': sample_team.games_played,
                'form_last_5': list(sample_team.rolling_data['form']),
                'goals_last_5': list(sample_team.rolling_data['goals'])
            }
        
        return summary

def save_state(state_manager: StateManager, output_path: str):
    """Sauvegarde l'état dans un fichier pickle"""
    logger = setup_logging()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(state_manager, f)
    
    logger.info(f"✅ State saved to: {output_path}")

def load_state(state_path: str) -> StateManager:
    """Charge l'état depuis un fichier pickle"""
    logger = setup_logging()
    
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"❌ State file not found: {state_path}")
    
    with open(state_path, 'rb') as f:
        state_manager = pickle.load(f)
    
    logger.info(f"✅ State loaded from: {state_path}")
    logger.info(f"  Teams: {len(state_manager.teams)}")
    logger.info(f"  Matches processed: {state_manager.processed_matches}")
    
    return state_manager

def main():
    """Pipeline principal d'initialisation de l'état historique"""
    parser = argparse.ArgumentParser(description="Initialize historical state for rolling simulation")
    parser.add_argument('--data', required=True, help='Path to historical data CSV')
    parser.add_argument('--season_start', required=True, help='Season start date (YYYY-MM-DD)')
    parser.add_argument('--output_state', required=True, help='Output state pickle file')
    parser.add_argument('--elo_k', type=float, default=DEFAULT_ELO_K, help='Elo K factor')
    parser.add_argument('--summary_json', help='Optional state summary JSON file')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🚀 HISTORICAL STATE INITIALIZATION")
    logger.info("=" * 80)
    logger.info(f"Input data: {args.data}")
    logger.info(f"Season start: {args.season_start}")
    logger.info(f"Output state: {args.output_state}")
    logger.info(f"Elo K-factor: {args.elo_k}")
    logger.info("=" * 80)
    
    try:
        # Construire l'état
        builder = HistoricalStateBuilder(args.data, args.elo_k)
        state_manager = builder.build_initial_state(args.season_start)
        
        # Valider l'état
        if not builder.validate_state_integrity(state_manager, args.season_start):
            logger.error("❌ State validation failed!")
            sys.exit(1)
        
        # Sauvegarder l'état
        save_state(state_manager, args.output_state)
        
        # Résumé optionnel
        if args.summary_json:
            import json
            summary = builder.generate_state_summary(state_manager)
            os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
            with open(args.summary_json, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"📋 Summary saved to: {args.summary_json}")
        
        logger.info("🎉 INITIALIZATION COMPLETE!")
        logger.info(f"State ready for rolling simulation starting {args.season_start}")
        
    except Exception as e:
        logger.error(f"❌ INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()