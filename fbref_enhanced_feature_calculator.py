"""
FBref Enhanced Feature Calculator - Features exactes avec vraies données
=======================================================================
Calculateur features utilisant données FBref fusionnées (xG, tirs, corners)
pour remplacer les approximations dans les prédictions J7
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FBrefEnhancedFeatureCalculator:
    """Calculateur features enhanced avec données FBref"""
    
    def __init__(self, fbref_data_path=None):
        self.fbref_data_path = fbref_data_path
        self.fbref_data = None
        self.features_order = [
            'form_diff_normalized',
            'elo_diff_normalized', 
            'h2h_score',
            'matchday_normalized',
            'shots_diff_normalized',      # ← Maintenant avec vraies données
            'corners_diff_normalized',    # ← Maintenant avec vraies données
            'market_entropy_norm',
            'home_xg_eff_10',            # ← Maintenant avec vraies données xG
            'away_goals_sum_5',
            'away_xg_eff_10'             # ← Maintenant avec vraies données xG
        ]
        
        # Charger données FBref si disponibles
        if fbref_data_path:
            self.load_fbref_data()
    
    def load_fbref_data(self):
        """Charge données FBref fusionnées"""
        try:
            if self.fbref_data_path and pd.io.common.file_exists(self.fbref_data_path):
                self.fbref_data = pd.read_csv(self.fbref_data_path)
                self.fbref_data['Date'] = pd.to_datetime(self.fbref_data['Date'])
                print(f"✅ Données FBref chargées: {len(self.fbref_data)} matchs")
                
                # Afficher colonnes FBref disponibles
                fbref_cols = [col for col in self.fbref_data.columns if any(x in col for x in ['xG', 'Shots', 'Corner', 'H_', 'A_'])]
                if fbref_cols:
                    print(f"📊 Colonnes FBref: {', '.join(fbref_cols)}")
                
                return True
            else:
                print(f"⚠️ Données FBref non trouvées: {self.fbref_data_path}")
                return False
        except Exception as e:
            print(f"❌ Erreur chargement FBref: {e}")
            return False
    
    def has_fbref_data(self):
        """Vérifie si données FBref disponibles"""
        return self.fbref_data is not None and len(self.fbref_data) > 0
    
    def get_fbref_stats_for_team(self, team, before_date, window=10, min_threshold=3):
        """Récupère stats FBref pour une équipe avant une date avec seuil minimal k≥3"""
        if not self.has_fbref_data():
            return None
        
        try:
            # Filtrer matchs équipe avant date
            team_matches = self.fbref_data[
                (
                    (self.fbref_data['HomeTeam'] == team) | 
                    (self.fbref_data['AwayTeam'] == team)
                ) & 
                (self.fbref_data['Date'] < before_date)
            ].tail(window)
            
            # Vérifier seuil minimal k≥3
            if len(team_matches) < min_threshold:
                from feature_fallback_tracker import track_insufficient_data
                track_insufficient_data(
                    f"J{self._estimate_matchday()}", 
                    f"{team}_{before_date}", 
                    f"fbref_stats_window_{window}",
                    len(team_matches), 
                    min_threshold
                )
                return None
            
            # Extraire stats selon position (home/away)
            stats = {
                'matches_count': len(team_matches),
                'xG': [],
                'xGA': [],
                'shots': [],
                'corners': [],
                'goals': []
            }
            
            for _, match in team_matches.iterrows():
                is_home = match['HomeTeam'] == team
                
                # xG stats
                if is_home:
                    if 'H_xG' in match and pd.notna(match['H_xG']):
                        stats['xG'].append(match['H_xG'])
                    if 'A_xG' in match and pd.notna(match['A_xG']):
                        stats['xGA'].append(match['A_xG'])
                    if 'H_Shots' in match and pd.notna(match['H_Shots']):
                        stats['shots'].append(match['H_Shots'])
                    if 'H_Corners' in match and pd.notna(match['H_Corners']):
                        stats['corners'].append(match['H_Corners'])
                    if 'FTHG' in match and pd.notna(match['FTHG']):
                        stats['goals'].append(match['FTHG'])
                else:  # away
                    if 'A_xG' in match and pd.notna(match['A_xG']):
                        stats['xG'].append(match['A_xG'])
                    if 'H_xG' in match and pd.notna(match['H_xG']):
                        stats['xGA'].append(match['H_xG'])
                    if 'A_Shots' in match and pd.notna(match['A_Shots']):
                        stats['shots'].append(match['A_Shots'])
                    if 'A_Corners' in match and pd.notna(match['A_Corners']):
                        stats['corners'].append(match['A_Corners'])
                    if 'FTAG' in match and pd.notna(match['FTAG']):
                        stats['goals'].append(match['FTAG'])
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Erreur stats FBref {team}: {e}")
            return None
    
    def calculate_all_features(self, match, historical_data):
        """Calcule toutes les 10 features (enhanced avec FBref si disponible)"""
        
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = pd.to_datetime('2025-10-03')  # Date J7
        
        features = {}
        
        print(f"   🔧 Calcul enhanced {home_team} vs {away_team}...")
        
        # 1-4. Features classiques (inchangées)
        features['form_diff_normalized'] = self._calculate_form_diff(home_team, away_team, historical_data)
        features['elo_diff_normalized'] = self._calculate_elo_diff(home_team, away_team, historical_data)
        features['h2h_score'] = self._calculate_h2h_score(home_team, away_team, historical_data)
        features['matchday_normalized'] = self._calculate_matchday_normalized(home_team, historical_data)
        
        # 5-6. Features tirs/corners (enhanced avec FBref)
        features['shots_diff_normalized'] = self._calculate_enhanced_shots_diff(home_team, away_team, historical_data, match_date)
        features['corners_diff_normalized'] = self._calculate_enhanced_corners_diff(home_team, away_team, historical_data, match_date)
        
        # 7. Market entropy (inchangée)
        features['market_entropy_norm'] = self._calculate_market_entropy(match)
        
        # 8-10. Features xG (enhanced avec FBref) 
        features['home_xg_eff_10'] = self._calculate_enhanced_xg_efficiency(home_team, historical_data, match_date, 'home')
        features['away_xg_eff_10'] = self._calculate_enhanced_xg_efficiency(away_team, historical_data, match_date, 'away')
        features['away_goals_sum_5'] = self._calculate_enhanced_away_goals_sum(away_team, historical_data, match_date)
        
        return features
    
    def _calculate_enhanced_shots_diff(self, home_team, away_team, historical_data, match_date):
        """Calcule shots_diff_normalized avec données FBref si disponibles"""
        
        if self.has_fbref_data():
            try:
                # Stats FBref
                home_stats = self.get_fbref_stats_for_team(home_team, match_date, window=5)
                away_stats = self.get_fbref_stats_for_team(away_team, match_date, window=5)
                
                if home_stats and away_stats and home_stats['shots'] and away_stats['shots']:
                    home_shots_avg = np.mean(home_stats['shots'])
                    away_shots_avg = np.mean(away_stats['shots'])
                    
                    shots_diff = (home_shots_avg - away_shots_avg) / 10  # Normaliser par 10 tirs
                    normalized = np.clip(0.5 + shots_diff/2, 0, 1)
                    
                    print(f"      shots_diff_normalized: {normalized:.4f} [FBref] (H:{home_shots_avg:.1f} A:{away_shots_avg:.1f})")
                    return normalized
                    
            except Exception as e:
                print(f"      ⚠️ Erreur shots FBref: {e}")
        
        # Fallback: méthode classique
        print(f"      shots_diff_normalized: 0.5000 [fallback]")
        return 0.5
    
    def _calculate_enhanced_corners_diff(self, home_team, away_team, historical_data, match_date):
        """Calcule corners_diff_normalized avec données FBref si disponibles"""
        
        if self.has_fbref_data():
            try:
                # Stats FBref
                home_stats = self.get_fbref_stats_for_team(home_team, match_date, window=5)
                away_stats = self.get_fbref_stats_for_team(away_team, match_date, window=5)
                
                if home_stats and away_stats and home_stats['corners'] and away_stats['corners']:
                    home_corners_avg = np.mean(home_stats['corners'])
                    away_corners_avg = np.mean(away_stats['corners'])
                    
                    corners_diff = (home_corners_avg - away_corners_avg) / 5  # Normaliser par 5 corners
                    normalized = np.clip(0.5 + corners_diff/2, 0, 1)
                    
                    print(f"      corners_diff_normalized: {normalized:.4f} [FBref] (H:{home_corners_avg:.1f} A:{away_corners_avg:.1f})")
                    return normalized
                    
            except Exception as e:
                print(f"      ⚠️ Erreur corners FBref: {e}")
        
        # Fallback: méthode classique
        print(f"      corners_diff_normalized: 0.5000 [fallback]")
        return 0.5
    
    def _calculate_enhanced_xg_efficiency(self, team, historical_data, match_date, side):
        """Calcule xG efficiency avec données FBref si disponibles"""
        
        if self.has_fbref_data():
            try:
                # Stats FBref
                team_stats = self.get_fbref_stats_for_team(team, match_date, window=10)
                
                if team_stats and team_stats['xG'] and team_stats['goals']:
                    total_xg = sum(team_stats['xG'])
                    total_goals = sum(team_stats['goals'])
                    
                    if total_xg > 0:
                        efficiency = min(1.0, total_goals / total_xg)
                        print(f"      {side}_xg_eff_10: {efficiency:.4f} [FBref] (G:{total_goals:.1f} xG:{total_xg:.1f})")
                        return efficiency
                    
            except Exception as e:
                print(f"      ⚠️ Erreur xG FBref {side}: {e}")
        
        # Fallback: approximation buts avec seuil k≥3
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | 
            (historical_data['AwayTeam'] == team)
        ].tail(10)
        
        # Vérifier seuil minimal k≥3
        if len(team_matches) < 3:
            from feature_fallback_tracker import track_insufficient_data
            track_insufficient_data(
                f"J{self._estimate_matchday()}", 
                f"{team}_xg_efficiency", 
                f"xg_efficiency_fallback_window_10",
                len(team_matches), 
                3
            )
            print(f"      {side}_xg_eff_10: NaN [insufficient data: {len(team_matches)}/3]")
            return np.nan
        
        goals = []
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                goals.append(match.get('FTHG', 1.5))
            else:
                goals.append(match.get('FTAG', 1.5))
        
        avg_goals = np.mean(goals) if goals else 1.5
        efficiency = min(1.0, avg_goals / 1.5)
        
        print(f"      {side}_xg_eff_10: {efficiency:.4f} [fallback] (avg goals: {avg_goals:.1f})")
        return efficiency
    
    def _calculate_enhanced_away_goals_sum(self, away_team, historical_data, match_date):
        """Calcule away_goals_sum_5 enhanced"""
        
        if self.has_fbref_data():
            try:
                # Stats FBref
                team_stats = self.get_fbref_stats_for_team(away_team, match_date, window=5)
                
                if team_stats and team_stats['goals']:
                    goals_sum = sum(team_stats['goals'])
                    print(f"      away_goals_sum_5: {goals_sum:.1f} [FBref]")
                    return float(goals_sum)
                    
            except Exception as e:
                print(f"      ⚠️ Erreur goals FBref away: {e}")
        
        # Fallback: méthode classique avec seuil k≥3
        away_matches = historical_data[
            (historical_data['HomeTeam'] == away_team) | 
            (historical_data['AwayTeam'] == away_team)
        ].tail(5)
        
        # Vérifier seuil minimal k≥3
        if len(away_matches) < 3:
            from feature_fallback_tracker import track_insufficient_data
            track_insufficient_data(
                f"J{self._estimate_matchday()}", 
                f"{away_team}_away_goals", 
                f"away_goals_sum_5",
                len(away_matches), 
                3
            )
            print(f"      away_goals_sum_5: NaN [insufficient data: {len(away_matches)}/3]")
            return np.nan
        
        goals_sum = 0
        for _, match in away_matches.iterrows():
            if match['HomeTeam'] == away_team:
                goals_sum += match.get('FTHG', 1)
            else:
                goals_sum += match.get('FTAG', 1)
        
        print(f"      away_goals_sum_5: {goals_sum:.1f} [fallback]")
        return float(goals_sum)
    
    # Méthodes inchangées (reprendre du calculateur précédent)
    def _calculate_form_diff(self, home_team, away_team, historical_data, window=5):
        """Form diff classique (inchangée)"""
        try:
            home_matches = historical_data[
                (historical_data['HomeTeam'] == home_team) | 
                (historical_data['AwayTeam'] == home_team)
            ].tail(window)
            
            home_points = sum([
                3 if ((row['HomeTeam'] == home_team and row['FullTimeResult'] == 'H') or 
                      (row['AwayTeam'] == home_team and row['FullTimeResult'] == 'A'))
                else 1 if row['FullTimeResult'] == 'D' else 0
                for _, row in home_matches.iterrows()
            ])
            
            away_matches = historical_data[
                (historical_data['HomeTeam'] == away_team) | 
                (historical_data['AwayTeam'] == away_team)
            ].tail(window)
            
            away_points = sum([
                3 if ((row['HomeTeam'] == away_team and row['FullTimeResult'] == 'H') or 
                      (row['AwayTeam'] == away_team and row['FullTimeResult'] == 'A'))
                else 1 if row['FullTimeResult'] == 'D' else 0
                for _, row in away_matches.iterrows()
            ])
            
            max_points = window * 3
            form_diff = (home_points - away_points) / max_points
            normalized = np.clip(0.5 + form_diff/2, 0, 1)
            
            print(f"      form_diff_normalized: {normalized:.4f} (H:{home_points} A:{away_points})")
            return normalized
            
        except Exception as e:
            print(f"      ⚠️ form_diff error: {e}")
            return 0.5
    
    def _calculate_elo_diff(self, home_team, away_team, historical_data, window=10):
        """ELO diff classique (inchangée)"""
        try:
            home_elo = self._estimate_team_elo(home_team, historical_data, window)
            away_elo = self._estimate_team_elo(away_team, historical_data, window)
            
            elo_diff = (home_elo - away_elo) / 400
            normalized = np.clip(0.5 + elo_diff/2, 0, 1)
            
            print(f"      elo_diff_normalized: {normalized:.4f} (H:{home_elo:.0f} A:{away_elo:.0f})")
            return normalized
            
        except Exception as e:
            print(f"      ⚠️ elo_diff error: {e}")
            return 0.5
    
    def _estimate_team_elo(self, team, historical_data, window=10):
        """Estime ELO équipe (inchangée)"""
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | 
            (historical_data['AwayTeam'] == team)
        ].tail(window)
        
        if len(team_matches) == 0:
            return 1500
        
        points = []
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                if match['FullTimeResult'] == 'H':
                    points.append(3)
                elif match['FullTimeResult'] == 'D':
                    points.append(1)
                else:
                    points.append(0)
            else:
                if match['FullTimeResult'] == 'A':
                    points.append(3)
                elif match['FullTimeResult'] == 'D':
                    points.append(1)
                else:
                    points.append(0)
        
        avg_points = np.mean(points)
        elo_estimate = 1500 + (avg_points - 1.5) * 200
        return elo_estimate
    
    def _calculate_h2h_score(self, home_team, away_team, historical_data, window=10):
        """H2H score classique (inchangée)"""
        try:
            h2h_matches = historical_data[
                ((historical_data['HomeTeam'] == home_team) & (historical_data['AwayTeam'] == away_team)) |
                ((historical_data['HomeTeam'] == away_team) & (historical_data['AwayTeam'] == home_team))
            ].tail(window)
            
            if len(h2h_matches) == 0:
                print(f"      h2h_score: 0.5000 (pas d'historique)")
                return 0.5
            
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
        """Matchday normalized (inchangée)"""
        try:
            team_matches = historical_data[
                (historical_data['HomeTeam'] == home_team) | 
                (historical_data['AwayTeam'] == home_team)
            ]
            
            current_season = team_matches[team_matches['Season'] == '2025-2026']
            matchday = len(current_season) + 1
            
            normalized = (matchday - 1) / (38 - 1)
            
            print(f"      matchday_normalized: {normalized:.4f} (J{matchday})")
            return normalized
            
        except Exception as e:
            print(f"      ⚠️ matchday error: {e}")
            return 0.18
    
    def _estimate_matchday(self):
        """Estime le numéro de journée actuel pour tracking"""
        # Estimation basique pour le tracker fallback
        return 7  # Default J7, peut être amélioré avec logique calendrier
    
    def _calculate_market_entropy(self, match):
        """Market entropy (inchangée)"""
        try:
            home_odds = match['B365H']
            draw_odds = match['B365D']
            away_odds = match['B365A']
            
            total_prob = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            home_prob = (1/home_odds) / total_prob
            draw_prob = (1/draw_odds) / total_prob
            away_prob = (1/away_odds) / total_prob
            
            entropy = -(home_prob * np.log(home_prob) + 
                       draw_prob * np.log(draw_prob) + 
                       away_prob * np.log(away_prob))
            
            normalized_entropy = entropy / np.log(3)
            
            print(f"      market_entropy_norm: {normalized_entropy:.4f}")
            return normalized_entropy
            
        except Exception as e:
            print(f"      ⚠️ market_entropy error: {e}")
            return 0.5
    
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