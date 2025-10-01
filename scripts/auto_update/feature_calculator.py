#!/usr/bin/env python3
"""
⚙️ FEATURE CALCULATOR - Calcul modulaire features avec anti-leakage
================================================================

Système modulaire et extensible pour calculer features ML 
avec respect strict de l'anti-leakage temporel.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("feature_calculator")

class BaseFeature(ABC):
    """Classe abstraite pour features modulaires"""
    
    @abstractmethod
    def calculate(self, match, historical_data):
        """Calcule feature pour un match donné"""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Nom de la feature"""
        pass

class FormDifferenceFeature(BaseFeature):
    """Feature différence de forme récente (5 derniers matchs)"""
    
    def __init__(self, window=5):
        self.window = window
    
    @property
    def name(self):
        return 'form_diff_normalized'
    
    def calculate(self, match, historical_data):
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        try:
            # Points récents home
            home_matches = historical_data[
                (historical_data['HomeTeam'] == home_team) | 
                (historical_data['AwayTeam'] == home_team)
            ].tail(self.window)
            
            home_points = sum([
                3 if ((row['HomeTeam'] == home_team and row['FullTimeResult'] == 'H') or 
                      (row['AwayTeam'] == home_team and row['FullTimeResult'] == 'A'))
                else 1 if row['FullTimeResult'] == 'D' else 0
                for _, row in home_matches.iterrows()
            ])
            
            # Points récents away
            away_matches = historical_data[
                (historical_data['HomeTeam'] == away_team) | 
                (historical_data['AwayTeam'] == away_team)
            ].tail(self.window)
            
            away_points = sum([
                3 if ((row['HomeTeam'] == away_team and row['FullTimeResult'] == 'H') or 
                      (row['AwayTeam'] == away_team and row['FullTimeResult'] == 'A'))
                else 1 if row['FullTimeResult'] == 'D' else 0
                for _, row in away_matches.iterrows()
            ])
            
            # Normaliser différence
            max_points = self.window * 3
            form_diff = (home_points - away_points) / max_points
            return np.clip(0.5 + form_diff/2, 0, 1)
            
        except Exception as e:
            logger.warning(f"⚠️  Form feature failed: {e}")
            return 0.5

class EloFeature(BaseFeature):
    """Feature basée sur différence Elo (simplifié)"""
    
    def __init__(self, initial_elo=1500, k_factor=20):
        self.initial_elo = initial_elo
        self.k_factor = k_factor
    
    @property
    def name(self):
        return 'elo_diff_normalized'
    
    def calculate(self, match, historical_data):
        try:
            # Calcul Elo simplifié basé sur victoires récentes
            home_team = match['HomeTeam'] 
            away_team = match['AwayTeam']
            
            # Estimation Elo basée sur performance récente (10 derniers matchs)
            home_elo = self._estimate_elo(home_team, historical_data, window=10)
            away_elo = self._estimate_elo(away_team, historical_data, window=10)
            
            # Normaliser différence Elo
            elo_diff = (home_elo - away_elo) / 400  # 400 points = différence significative
            return np.clip(0.5 + elo_diff/2, 0, 1)
            
        except Exception as e:
            logger.warning(f"⚠️  Elo feature failed: {e}")
            return 0.5
    
    def _estimate_elo(self, team, historical_data, window=10):
        """Estime Elo équipe basé sur performances récentes"""
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | 
            (historical_data['AwayTeam'] == team)
        ].tail(window)
        
        if len(team_matches) == 0:
            return self.initial_elo
        
        # Points moyens -> estimation Elo
        points = []
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                if match['FullTimeResult'] == 'H':
                    points.append(3)
                elif match['FullTimeResult'] == 'D':
                    points.append(1)
                else:
                    points.append(0)
            else:  # Away
                if match['FullTimeResult'] == 'A':
                    points.append(3)
                elif match['FullTimeResult'] == 'D':
                    points.append(1)
                else:
                    points.append(0)
        
        avg_points = np.mean(points)
        # Conversion points -> Elo approximatif
        elo_estimate = self.initial_elo + (avg_points - 1.5) * 200  # 1.5 = moyenne neutre
        return elo_estimate

class HeadToHeadFeature(BaseFeature):
    """Feature historique confrontations directes"""
    
    def __init__(self, window=10):
        self.window = window
    
    @property  
    def name(self):
        return 'h2h_score'
    
    def calculate(self, match, historical_data):
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        try:
            # Confrontations directes récentes
            h2h_matches = historical_data[
                ((historical_data['HomeTeam'] == home_team) & (historical_data['AwayTeam'] == away_team)) |
                ((historical_data['HomeTeam'] == away_team) & (historical_data['AwayTeam'] == home_team))
            ].tail(self.window)
            
            if len(h2h_matches) == 0:
                return 0.5  # Neutre si pas d'historique
            
            # Calculer score home vs away
            home_wins = len(h2h_matches[
                ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'H')) |
                ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'A'))
            ])
            
            total_matches = len(h2h_matches)
            h2h_ratio = home_wins / total_matches
            
            return np.clip(h2h_ratio, 0, 1)
            
        except Exception as e:
            logger.warning(f"⚠️  H2H feature failed: {e}")
            return 0.5

class MatchdayFeature(BaseFeature):
    """Feature progression saison (journée normalisée)"""
    
    @property
    def name(self):
        return 'matchday_normalized'
    
    def calculate(self, match, historical_data):
        try:
            # Estimer journée basée sur nombre matchs joués par équipe
            home_team = match['HomeTeam']
            
            team_matches = historical_data[
                (historical_data['HomeTeam'] == home_team) | 
                (historical_data['AwayTeam'] == home_team)
            ]
            
            # Filtrer même saison
            match_season = match.get('Season', '2025-2026')
            season_matches = team_matches[team_matches['Season'] == match_season]
            
            matchday_estimate = len(season_matches) + 1  # +1 pour match courant
            
            # Normaliser (EPL = 38 journées)
            return np.clip(matchday_estimate / 38, 0, 1)
            
        except Exception as e:
            logger.warning(f"⚠️  Matchday feature failed: {e}")
            return 0.1  # Début saison par défaut

class XGEfficiencyFeature(BaseFeature):
    """Feature efficacité xG (si données disponibles)"""
    
    def __init__(self, window=10, home_or_away='home'):
        self.window = window
        self.home_or_away = home_or_away
    
    @property
    def name(self):
        return f'{self.home_or_away}_xg_eff_{self.window}'
    
    def calculate(self, match, historical_data):
        try:
            team = match['HomeTeam'] if self.home_or_away == 'home' else match['AwayTeam']
            
            # Chercher données xG si disponibles
            if 'HomeXG' not in historical_data.columns:
                return 1.0  # Valeur neutre si pas de xG
            
            # Matchs récents avec xG
            team_matches = historical_data[
                (historical_data['HomeTeam'] == team) | 
                (historical_data['AwayTeam'] == team)
            ].tail(self.window)
            
            if len(team_matches) == 0:
                return 1.0
            
            # Calculer efficacité Goals/xG
            goals = []
            xg_values = []
            
            for _, row in team_matches.iterrows():
                if row['HomeTeam'] == team:
                    goals.append(row.get('FTHG', 0))
                    xg_values.append(row.get('HomeXG', 1.0))
                else:
                    goals.append(row.get('FTAG', 0))
                    xg_values.append(row.get('AwayXG', 1.0))
            
            total_goals = sum(goals)
            total_xg = sum(xg_values)
            
            if total_xg == 0:
                return 1.0
            
            efficiency = total_goals / total_xg
            return np.clip(efficiency, 0.2, 3.0)  # Borner valeurs extrêmes
            
        except Exception as e:
            logger.warning(f"⚠️  xG efficiency feature failed: {e}")
            return 1.0

class MarketEntropyHistoricalFeature(BaseFeature):
    """Enhanced market entropy using ALL historical odds data"""
    
    @property
    def name(self):
        return 'market_entropy_historical'
    
    def calculate(self, match, historical_data):
        try:
            # Try to get odds from raw data columns (B365H, B365D, B365A)
            home_odds = match.get('B365H', None)
            draw_odds = match.get('B365D', None)  
            away_odds = match.get('B365A', None)
            
            # If direct odds available, calculate entropy
            if all([home_odds, draw_odds, away_odds]) and all(pd.notna([home_odds, draw_odds, away_odds])):
                return self._calculate_market_entropy(home_odds, draw_odds, away_odds)
            
            # Fallback: use existing market_entropy_norm if available
            if 'market_entropy_norm' in match and pd.notna(match['market_entropy_norm']):
                return match['market_entropy_norm']
            
            # Final fallback: moderate entropy
            return 0.6
            
        except Exception as e:
            logger.warning(f"⚠️ Market entropy historical failed: {e}")
            return 0.6
    
    def _calculate_market_entropy(self, h_odds, d_odds, a_odds):
        """Calculate normalized market entropy from odds"""
        # Convert odds to probabilities
        h_prob = 1 / h_odds
        d_prob = 1 / d_odds
        a_prob = 1 / a_odds
        total = h_prob + d_prob + a_prob
        
        # Normalize probabilities
        h_prob /= total
        d_prob /= total  
        a_prob /= total
        
        # Calculate Shannon entropy
        entropy = -(h_prob * np.log(h_prob) + d_prob * np.log(d_prob) + a_prob * np.log(a_prob))
        
        # Normalize to [0,1] (max entropy for 3 outcomes = log(3))
        return entropy / np.log(3)

class OddsSpreadFeature(BaseFeature):
    """Spread between home and away odds (dual-purpose feature)"""
    
    @property
    def name(self):
        return 'odds_spread_normalized'
    
    def calculate(self, match, historical_data):
        try:
            home_odds = match.get('B365H', 2.0)
            away_odds = match.get('B365A', 2.0)
            
            # Handle missing odds
            if pd.isna(home_odds): home_odds = 2.0
            if pd.isna(away_odds): away_odds = 2.0
            
            # Calculate absolute spread
            spread = abs(home_odds - away_odds)
            
            # Normalize to [0,1] (typical EPL spread range: 0-3)
            normalized_spread = np.clip(spread / 3.0, 0, 1)
            
            return normalized_spread
            
        except Exception as e:
            logger.warning(f"⚠️ Odds spread feature failed: {e}")
            return 0.3  # Moderate spread

class DrawMarginFeature(BaseFeature):
    """Draw probability margin vs average H/A probability"""
    
    @property
    def name(self):
        return 'draw_margin_normalized'
    
    def calculate(self, match, historical_data):
        try:
            h_odds = match.get('B365H', 2.5)
            d_odds = match.get('B365D', 3.3)
            a_odds = match.get('B365A', 2.5)
            
            # Handle missing odds with EPL typical values
            if pd.isna(h_odds): h_odds = 2.5
            if pd.isna(d_odds): d_odds = 3.3  
            if pd.isna(a_odds): a_odds = 2.5
            
            # Convert to implied probabilities
            h_prob = 1 / h_odds
            d_prob = 1 / d_odds
            a_prob = 1 / a_odds
            
            # Normalize probabilities
            total = h_prob + d_prob + a_prob
            h_prob_norm = h_prob / total
            d_prob_norm = d_prob / total
            a_prob_norm = a_prob / total
            
            # Draw margin = draw_prob - average(home_prob, away_prob)
            ha_average = (h_prob_norm + a_prob_norm) / 2
            draw_margin = d_prob_norm - ha_average
            
            # Normalize to [0,1] (typical range: -0.2 to +0.2)
            normalized_margin = np.clip((draw_margin + 0.2) / 0.4, 0, 1)
            
            return normalized_margin
            
        except Exception as e:
            logger.warning(f"⚠️ Draw margin feature failed: {e}")
            return 0.5  # Neutral margin

class FormVarianceFeature(BaseFeature):
    """Team performance instability indicator (draws occur with inconsistent teams)"""
    
    def __init__(self, window=5):
        self.window = window
    
    @property
    def name(self):
        return 'form_variance_normalized'
    
    def calculate(self, match, historical_data):
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        try:
            # Get recent results for both teams
            home_recent = self._get_recent_points(home_team, historical_data)
            away_recent = self._get_recent_points(away_team, historical_data)
            
            # Calculate variance in performance (points: 0=loss, 1=draw, 3=win)
            home_variance = np.var(home_recent) if len(home_recent) > 1 else 0
            away_variance = np.var(away_recent) if len(away_recent) > 1 else 0
            
            # Combined variance (higher = more inconsistent = more draws likely)
            combined_variance = (home_variance + away_variance) / 2
            
            # Normalize (max theoretical variance ≈ 2.25 for [0,1,3] values)
            normalized_variance = np.clip(combined_variance / 2.25, 0, 1)
            
            return normalized_variance
            
        except Exception as e:
            logger.warning(f"⚠️ Form variance feature failed: {e}")
            return 0.3  # Moderate variance
    
    def _get_recent_points(self, team, historical_data):
        """Get recent match points for a team"""
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | 
            (historical_data['AwayTeam'] == team)
        ].tail(self.window)
        
        points = []
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                if match['FullTimeResult'] == 'H':
                    points.append(3)
                elif match['FullTimeResult'] == 'D':
                    points.append(1)
                else:
                    points.append(0)
            else:  # Away team
                if match['FullTimeResult'] == 'A':
                    points.append(3)
                elif match['FullTimeResult'] == 'D':
                    points.append(1)
                else:
                    points.append(0)
        
        return points

class FeatureCalculator:
    """Calculateur principal features avec système modulaire"""
    
    def __init__(self):
        # Features par défaut
        self.features = {
            'form_diff_normalized': FormDifferenceFeature(window=5),
            'elo_diff_normalized': EloFeature(),
            'h2h_score': HeadToHeadFeature(window=10),
            'matchday_normalized': MatchdayFeature(),
            'home_xg_eff_10': XGEfficiencyFeature(window=10, home_or_away='home'),
            'away_xg_eff_10': XGEfficiencyFeature(window=10, home_or_away='away')
        }
        
        # Enhanced draw-focused features
        self.enhanced_features = {
            'market_entropy_historical': MarketEntropyHistoricalFeature(),
            'odds_spread_normalized': OddsSpreadFeature(),
            'draw_margin_normalized': DrawMarginFeature(),
            'form_variance_normalized': FormVarianceFeature(window=5)
        }
        
        # Features additionnelles placeholder
        self.default_features = {
            'shots_diff_normalized': 0.5,
            'corners_diff_normalized': 0.5, 
            'market_entropy_norm': 0.5,
            'away_goals_sum_5': 5.0
        }
    
    def add_feature(self, feature_instance):
        """Ajoute nouvelle feature modulaire"""
        self.features[feature_instance.name] = feature_instance
        logger.info(f"✅ Feature ajoutée: {feature_instance.name}")
    
    def calculate_safe_features(self, match, historical_data, strict_cutoff=True, include_enhanced=True):
        """Calcule toutes features avec anti-leakage strict"""
        match_datetime = match['Date']
        
        # ANTI-LEAKAGE STRICT: seulement données AVANT ce match
        if strict_cutoff:
            cutoff = match_datetime - timedelta(minutes=1)
            safe_data = historical_data[historical_data['Date'] < cutoff]
            
            # Vérification matchs même jour
            same_day = historical_data[historical_data['Date'].dt.date == match_datetime.date()]
            if len(same_day) > 0:
                logger.warning(f"🔒 {len(same_day)} matchs même jour exclus pour {match['HomeTeam']} vs {match['AwayTeam']}")
        else:
            safe_data = historical_data
        
        logger.info(f"🔒 Calcul features avec {len(safe_data)} matchs historiques (cutoff: {cutoff if strict_cutoff else 'None'})")
        
        # Calculer features modulaires
        calculated_features = {}
        
        # Standard features
        for name, feature_instance in self.features.items():
            try:
                value = feature_instance.calculate(match, safe_data)
                calculated_features[name] = value
                logger.debug(f"  ✅ {name}: {value:.3f}")
            except Exception as e:
                logger.error(f"  ❌ {name} failed: {e}")
                calculated_features[name] = 0.5  # Valeur neutre safe
        
        # Enhanced draw-focused features
        if include_enhanced:
            for name, feature_instance in self.enhanced_features.items():
                try:
                    value = feature_instance.calculate(match, safe_data)
                    calculated_features[name] = value
                    logger.debug(f"  🎯 {name}: {value:.3f}")
                except Exception as e:
                    logger.error(f"  ❌ {name} failed: {e}")
                    calculated_features[name] = 0.5  # Valeur neutre safe
        
        # Ajouter features par défaut
        for name, default_value in self.default_features.items():
            if name not in calculated_features:
                calculated_features[name] = default_value
        
        return calculated_features
    
    def get_feature_names(self, include_enhanced=True):
        """Liste toutes features disponibles"""
        all_features = list(self.features.keys()) + list(self.default_features.keys())
        if include_enhanced:
            all_features += list(self.enhanced_features.keys())
        return sorted(all_features)
    
    def get_draw_focused_features(self):
        """Liste des features spécialement conçues pour draw detection"""
        return list(self.enhanced_features.keys()) + ['matchday_normalized']  # Early season has more draws
    
    def get_classical_features(self):
        """Liste des features classiques pour H/A classification"""
        classical = list(self.features.keys()) + [
            'shots_diff_normalized', 'corners_diff_normalized', 
            'away_goals_sum_5', 'odds_spread_normalized'  # odds_spread useful for H/A too
        ]
        return classical

# Interface simple
def calculate_match_features(match, historical_data, strict_anti_leakage=True):
    """Interface simple calcul features pour un match"""
    calculator = FeatureCalculator()
    return calculator.calculate_safe_features(match, historical_data, strict_anti_leakage)

if __name__ == "__main__":
    # Test basique
    logger.info("🧪 Test Feature Calculator...")
    
    # Mock data pour test
    mock_match = {
        'HomeTeam': 'Liverpool',
        'AwayTeam': 'Arsenal', 
        'Date': pd.to_datetime('2025-08-15'),
        'Season': '2025-2026'
    }
    
    mock_historical = pd.DataFrame({
        'Date': pd.to_datetime(['2025-08-10', '2025-08-12']),
        'HomeTeam': ['Liverpool', 'Arsenal'],
        'AwayTeam': ['Chelsea', 'Brighton'],
        'FullTimeResult': ['H', 'A'],
        'Season': ['2025-2026', '2025-2026']
    })
    
    features = calculate_match_features(mock_match, mock_historical)
    
    logger.info("✅ Features calculées:")
    for name, value in features.items():
        logger.info(f"  {name}: {value}")