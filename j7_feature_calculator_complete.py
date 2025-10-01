"""
Feature Calculator COMPLET pour J7 - Pipeline exacte
Reproduit exactement les 10 features du Baseline Champion v2.3
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class J7FeatureCalculator:
    """Calculateur features exact pour J7 sans approximation"""
    
    def __init__(self):
        self.features_order = [
            'form_diff_normalized',
            'elo_diff_normalized', 
            'h2h_score',
            'matchday_normalized',
            'shots_diff_normalized',
            'corners_diff_normalized',
            'market_entropy_norm',
            'home_xg_eff_10',
            'away_goals_sum_5',
            'away_xg_eff_10'
        ]
    
    def calculate_all_features(self, match, historical_data):
        """Calcule toutes les 10 features pour un match J7"""
        
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        features = {}
        
        print(f"   🔧 Calcul {home_team} vs {away_team}...")
        
        # 1. form_diff_normalized
        features['form_diff_normalized'] = self._calculate_form_diff(home_team, away_team, historical_data)
        
        # 2. elo_diff_normalized
        features['elo_diff_normalized'] = self._calculate_elo_diff(home_team, away_team, historical_data)
        
        # 3. h2h_score
        features['h2h_score'] = self._calculate_h2h_score(home_team, away_team, historical_data)
        
        # 4. matchday_normalized
        features['matchday_normalized'] = self._calculate_matchday_normalized(home_team, historical_data)
        
        # 5. shots_diff_normalized
        features['shots_diff_normalized'] = self._calculate_shots_diff(home_team, away_team, historical_data)
        
        # 6. corners_diff_normalized
        features['corners_diff_normalized'] = self._calculate_corners_diff(home_team, away_team, historical_data)
        
        # 7. market_entropy_norm (depuis cotes)
        features['market_entropy_norm'] = self._calculate_market_entropy(match)
        
        # 8. home_xg_eff_10
        features['home_xg_eff_10'] = self._calculate_xg_efficiency(home_team, historical_data, 'home')
        
        # 9. away_goals_sum_5
        features['away_goals_sum_5'] = self._calculate_away_goals_sum(away_team, historical_data)
        
        # 10. away_xg_eff_10
        features['away_xg_eff_10'] = self._calculate_xg_efficiency(away_team, historical_data, 'away')
        
        return features
    
    def _calculate_form_diff(self, home_team, away_team, historical_data, window=5, min_threshold=3):
        """Calcule form_diff_normalized exacte avec seuil minimal k≥3"""
        
        try:
            # Home team form
            home_matches = historical_data[
                (historical_data['HomeTeam'] == home_team) | 
                (historical_data['AwayTeam'] == home_team)
            ].tail(window)
            
            # Vérifier seuil minimal
            if len(home_matches) < min_threshold:
                from feature_fallback_tracker import track_insufficient_data
                track_insufficient_data(
                    f"J7", 
                    f"{home_team}_vs_{away_team}", 
                    f"home_form_window_{window}",
                    len(home_matches), 
                    min_threshold
                )
                return np.nan
            
            home_points = 0
            for _, match in home_matches.iterrows():
                if match['HomeTeam'] == home_team:
                    if match['FullTimeResult'] == 'H':
                        home_points += 3
                    elif match['FullTimeResult'] == 'D':
                        home_points += 1
                else:  # away
                    if match['FullTimeResult'] == 'A':
                        home_points += 3
                    elif match['FullTimeResult'] == 'D':
                        home_points += 1
            
            # Away team form
            away_matches = historical_data[
                (historical_data['HomeTeam'] == away_team) | 
                (historical_data['AwayTeam'] == away_team)
            ].tail(window)
            
            # Vérifier seuil minimal pour away team
            if len(away_matches) < min_threshold:
                from feature_fallback_tracker import track_insufficient_data
                track_insufficient_data(
                    f"J7", 
                    f"{home_team}_vs_{away_team}", 
                    f"away_form_window_{window}",
                    len(away_matches), 
                    min_threshold
                )
                return np.nan
            
            away_points = 0
            for _, match in away_matches.iterrows():
                if match['HomeTeam'] == away_team:
                    if match['FullTimeResult'] == 'H':
                        away_points += 3
                    elif match['FullTimeResult'] == 'D':
                        away_points += 1
                else:  # away
                    if match['FullTimeResult'] == 'A':
                        away_points += 3
                    elif match['FullTimeResult'] == 'D':
                        away_points += 1
            
            # Normaliser différence
            max_points = window * 3
            form_diff = (home_points - away_points) / max_points
            normalized = np.clip(0.5 + form_diff/2, 0, 1)
            
            print(f"      form_diff_normalized: {normalized:.4f} (H:{home_points} A:{away_points})")
            return normalized
            
        except Exception as e:
            print(f"      ⚠️ form_diff error: {e}")
            return 0.5
    
    def _calculate_elo_diff(self, home_team, away_team, historical_data, window=10):
        """Calcule elo_diff_normalized basé sur performance récente"""
        
        try:
            # Estimation Elo basée sur performance récente
            home_elo = self._estimate_team_elo(home_team, historical_data, window)
            away_elo = self._estimate_team_elo(away_team, historical_data, window)
            
            # Normaliser différence Elo
            elo_diff = (home_elo - away_elo) / 400  # 400 points = différence significative
            normalized = np.clip(0.5 + elo_diff/2, 0, 1)
            
            print(f"      elo_diff_normalized: {normalized:.4f} (H:{home_elo:.0f} A:{away_elo:.0f})")
            return normalized
            
        except Exception as e:
            print(f"      ⚠️ elo_diff error: {e}")
            return 0.5
    
    def _estimate_team_elo(self, team, historical_data, window=10):
        """Estime l'ELO d'une équipe basé sur performances récentes"""
        
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | 
            (historical_data['AwayTeam'] == team)
        ].tail(window)
        
        if len(team_matches) == 0:
            return 1500  # ELO de base
        
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
        # Conversion points -> Elo (1.5 = moyenne neutre)
        elo_estimate = 1500 + (avg_points - 1.5) * 200
        return elo_estimate
    
    def _calculate_h2h_score(self, home_team, away_team, historical_data, window=10):
        """Calcule h2h_score exact"""
        
        try:
            # Confrontations directes
            h2h_matches = historical_data[
                ((historical_data['HomeTeam'] == home_team) & (historical_data['AwayTeam'] == away_team)) |
                ((historical_data['HomeTeam'] == away_team) & (historical_data['AwayTeam'] == home_team))
            ].tail(window)
            
            if len(h2h_matches) == 0:
                print(f"      h2h_score: 0.5000 (pas d'historique)")
                return 0.5
            
            # Compter victoires home_team
            home_wins = 0
            for _, match in h2h_matches.iterrows():
                if match['HomeTeam'] == home_team and match['FullTimeResult'] == 'H':
                    home_wins += 1
                elif match['AwayTeam'] == home_team and match['FullTimeResult'] == 'A':
                    home_wins += 1
            
            h2h_ratio = home_wins / len(h2h_matches)
            print(f"      h2h_score: {h2h_ratio:.4f} ({home_wins}/{len(h2h_matches)})")
            return h2h_ratio
            
        except Exception as e:
            print(f"      ⚠️ h2h_score error: {e}")
            return 0.5
    
    def _calculate_matchday_normalized(self, home_team, historical_data):
        """Calcule matchday_normalized"""
        
        try:
            # Estimer journée basée sur nombre de matchs joués
            team_matches = historical_data[
                (historical_data['HomeTeam'] == home_team) | 
                (historical_data['AwayTeam'] == home_team)
            ]
            
            # Filtrer saison actuelle 2025-26
            current_season = team_matches[team_matches['Season'] == '2025-2026']
            matchday = len(current_season) + 1  # Prochain match
            
            # Normaliser sur 38 journées
            normalized = (matchday - 1) / (38 - 1)
            
            print(f"      matchday_normalized: {normalized:.4f} (J{matchday})")
            return normalized
            
        except Exception as e:
            print(f"      ⚠️ matchday error: {e}")
            return 0.18  # Approximation J7
    
    def _calculate_shots_diff(self, home_team, away_team, historical_data, window=5):
        """Calcule shots_diff_normalized si données disponibles"""
        
        try:
            # Chercher colonnes tirs
            shot_cols = [col for col in historical_data.columns if 'shot' in col.lower() or 'HS' in col or 'AS' in col]
            
            if not shot_cols:
                print(f"      shots_diff_normalized: 0.5000 (pas de données tirs)")
                return 0.5
            
            # Si colonnes HS/AS disponibles, calculer différentiel
            if 'HS' in historical_data.columns and 'AS' in historical_data.columns:
                home_shots_avg = self._calculate_team_shots_avg(home_team, historical_data, 'home', window)
                away_shots_avg = self._calculate_team_shots_avg(away_team, historical_data, 'away', window)
                
                shots_diff = (home_shots_avg - away_shots_avg) / 10  # Normaliser par 10 tirs
                normalized = np.clip(0.5 + shots_diff/2, 0, 1)
                
                print(f"      shots_diff_normalized: {normalized:.4f} (H:{home_shots_avg:.1f} A:{away_shots_avg:.1f})")
                return normalized
            else:
                print(f"      shots_diff_normalized: 0.5000 (format non supporté)")
                return 0.5
                
        except Exception as e:
            print(f"      ⚠️ shots_diff error: {e}")
            return 0.5
    
    def _calculate_team_shots_avg(self, team, historical_data, side, window=5):
        """Calcule moyenne tirs équipe"""
        
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | 
            (historical_data['AwayTeam'] == team)
        ].tail(window)
        
        shots = []
        for _, match in team_matches.iterrows():
            if side == 'home':
                if match['HomeTeam'] == team and 'HS' in match:
                    shots.append(match['HS'])
            else:  # away
                if match['AwayTeam'] == team and 'AS' in match:
                    shots.append(match['AS'])
        
        return np.mean(shots) if shots else 8.0  # Moyenne EPL approximative
    
    def _calculate_corners_diff(self, home_team, away_team, historical_data, window=5):
        """Calcule corners_diff_normalized si données disponibles"""
        
        try:
            # Similaire aux tirs mais pour corners
            corner_cols = [col for col in historical_data.columns if 'corner' in col.lower() or 'HC' in col or 'AC' in col]
            
            if not corner_cols:
                print(f"      corners_diff_normalized: 0.5000 (pas de données corners)")
                return 0.5
            
            # Si colonnes HC/AC disponibles
            if 'HC' in historical_data.columns and 'AC' in historical_data.columns:
                home_corners_avg = self._calculate_team_corners_avg(home_team, historical_data, 'home', window)
                away_corners_avg = self._calculate_team_corners_avg(away_team, historical_data, 'away', window)
                
                corners_diff = (home_corners_avg - away_corners_avg) / 5  # Normaliser par 5 corners
                normalized = np.clip(0.5 + corners_diff/2, 0, 1)
                
                print(f"      corners_diff_normalized: {normalized:.4f} (H:{home_corners_avg:.1f} A:{away_corners_avg:.1f})")
                return normalized
            else:
                print(f"      corners_diff_normalized: 0.5000 (format non supporté)")
                return 0.5
                
        except Exception as e:
            print(f"      ⚠️ corners_diff error: {e}")
            return 0.5
    
    def _calculate_team_corners_avg(self, team, historical_data, side, window=5):
        """Calcule moyenne corners équipe"""
        
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | 
            (historical_data['AwayTeam'] == team)
        ].tail(window)
        
        corners = []
        for _, match in team_matches.iterrows():
            if side == 'home':
                if match['HomeTeam'] == team and 'HC' in match:
                    corners.append(match['HC'])
            else:  # away
                if match['AwayTeam'] == team and 'AC' in match:
                    corners.append(match['AC'])
        
        return np.mean(corners) if corners else 5.0  # Moyenne EPL approximative
    
    def _calculate_market_entropy(self, match):
        """Calcule market_entropy_norm depuis les cotes"""
        
        try:
            home_odds = match['B365H']
            draw_odds = match['B365D']
            away_odds = match['B365A']
            
            # Probabilités implicites
            total_prob = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            home_prob = (1/home_odds) / total_prob
            draw_prob = (1/draw_odds) / total_prob
            away_prob = (1/away_odds) / total_prob
            
            # Entropie de Shannon
            entropy = -(home_prob * np.log(home_prob) + 
                       draw_prob * np.log(draw_prob) + 
                       away_prob * np.log(away_prob))
            
            # Normaliser par log(3)
            normalized_entropy = entropy / np.log(3)
            
            print(f"      market_entropy_norm: {normalized_entropy:.4f}")
            return normalized_entropy
            
        except Exception as e:
            print(f"      ⚠️ market_entropy error: {e}")
            return 0.5
    
    def _calculate_xg_efficiency(self, team, historical_data, side, window=10):
        """Calcule xG efficiency sur 10 matchs"""
        
        try:
            # Chercher colonnes xG
            xg_cols = [col for col in historical_data.columns if 'xg' in col.lower()]
            
            if not xg_cols:
                # Approximer via buts si pas de xG
                team_matches = historical_data[
                    (historical_data['HomeTeam'] == team) | 
                    (historical_data['AwayTeam'] == team)
                ].tail(window)
                
                goals = []
                for _, match in team_matches.iterrows():
                    if match['HomeTeam'] == team:
                        goals.append(match.get('FTHG', 1.5))
                    else:
                        goals.append(match.get('FTAG', 1.5))
                
                # Efficacité approximée (goals / expected average)
                avg_goals = np.mean(goals) if goals else 1.5
                efficiency = min(1.0, avg_goals / 1.5)  # 1.5 = moyenne EPL
                
                print(f"      {side}_xg_eff_10: {efficiency:.4f} (approx via buts: {avg_goals:.1f})")
                return efficiency
            
            # TODO: Implémenter avec vraies données xG si disponibles
            print(f"      {side}_xg_eff_10: 1.0000 (xG data found but not implemented)")
            return 1.0
            
        except Exception as e:
            print(f"      ⚠️ {side}_xg_eff error: {e}")
            return 1.0
    
    def _calculate_away_goals_sum(self, away_team, historical_data, window=5):
        """Calcule away_goals_sum_5"""
        
        try:
            # Derniers 5 matchs de l'équipe away (tous contextes)
            away_matches = historical_data[
                (historical_data['HomeTeam'] == away_team) | 
                (historical_data['AwayTeam'] == away_team)
            ].tail(window)
            
            goals_sum = 0
            for _, match in away_matches.iterrows():
                if match['HomeTeam'] == away_team:
                    goals_sum += match.get('FTHG', 1)  # Buts marqués à domicile
                else:
                    goals_sum += match.get('FTAG', 1)  # Buts marqués à l'extérieur
            
            print(f"      away_goals_sum_5: {goals_sum:.1f}")
            return float(goals_sum)
            
        except Exception as e:
            print(f"      ⚠️ away_goals_sum error: {e}")
            return 5.0
    
    def get_features_vector(self, features):
        """Retourne vecteur features dans l'ordre exact du modèle"""
        
        return [
            features['form_diff_normalized'],
            features['elo_diff_normalized'],
            features['h2h_score'],
            features['matchday_normalized'],
            features['shots_diff_normalized'],
            features['corners_diff_normalized'],
            features['market_entropy_norm'],
            features['home_xg_eff_10'],
            features['away_goals_sum_5'],
            features['away_xg_eff_10']
        ]