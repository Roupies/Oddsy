#!/usr/bin/env python3
"""
Dynamic Features Calculator - LA BASE ENFIN FAITE CORRECTEMENT
==============================================================

MISSION CRITIQUE: Calculer les features dynamiquement pour chaque match,
en utilisant SEULEMENT les données disponibles AVANT la date du match.

FEATURES DYNAMIQUES IMPLÉMENTÉES:
1. Form dynamique (5 derniers matchs par équipe)
2. Elo dynamique (ratings temps réel)
3. xG efficiency dynamique (10 derniers matchs)
4. H2H dynamique
5. Shots/Corners tendances récentes

OBJECTIF: Passer de 40% (features statiques) à 50-55% (features dynamiques)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DynamicFeaturesCalculator:
    """
    Calculateur de features dynamiques temps réel
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        
        # Paramètres par défaut
        self.form_window = 5      # 5 derniers matchs pour la forme
        self.xg_window = 10       # 10 derniers matchs pour xG efficiency
        self.elo_k_factor = 20    # K-factor pour mise à jour Elo
        self.initial_elo = 1500   # Elo initial pour nouvelles équipes
        
        # Cache pour optimisation
        self.elo_cache = {}
        self.form_cache = {}
        
    def load_historical_data(self):
        """Charger les données historiques"""
        
        print("📊 CHARGEMENT DONNÉES POUR FEATURES DYNAMIQUES")
        print("="*55)
        
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        
        # Trier par date pour calculs chronologiques
        self.dataset = self.dataset.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Dataset chargé: {len(self.dataset)} matches")
        print(f"   Période: {self.dataset['Date'].min().strftime('%d/%m/%Y')} → {self.dataset['Date'].max().strftime('%d/%m/%Y')}")
        
        # Vérifier colonnes nécessaires
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'FTHG', 'FTAG']
        missing_cols = [col for col in required_cols if col not in self.dataset.columns]
        
        if missing_cols:
            print(f"⚠️  Colonnes manquantes: {missing_cols}")
            # Essayer colonnes alternatives
            if 'FTR' in self.dataset.columns and 'FullTimeResult' not in self.dataset.columns:
                self.dataset['FullTimeResult'] = self.dataset['FTR']
                print("   ✅ Utilisé FTR → FullTimeResult")
                
        print(f"✅ Données prêtes pour calculs dynamiques")
        
        return True
        
    def get_team_matches_before_date(self, team, before_date, n_matches=None):
        """Récupérer les n derniers matchs d'une équipe avant une date"""
        
        # Filtrer tous les matchs de l'équipe avant la date
        team_matches = self.dataset[
            ((self.dataset['HomeTeam'] == team) | (self.dataset['AwayTeam'] == team)) &
            (self.dataset['Date'] < before_date)
        ].copy()
        
        # Trier par date décroissante
        team_matches = team_matches.sort_values('Date', ascending=False)
        
        # Limiter au nombre de matchs demandé
        if n_matches is not None:
            team_matches = team_matches.head(n_matches)
            
        return team_matches
        
    def calculate_dynamic_form(self, team, before_date, n_matches=5):
        """Calculer la forme dynamique d'une équipe (points par match)"""
        
        recent_matches = self.get_team_matches_before_date(team, before_date, n_matches)
        
        if len(recent_matches) == 0:
            return 0.5  # Forme neutre par défaut
            
        total_points = 0
        
        for _, match in recent_matches.iterrows():
            is_home = match['HomeTeam'] == team
            result = match['FullTimeResult']
            
            # Calculer points selon résultat
            if (is_home and result == 'H') or (not is_home and result == 'A'):
                points = 3  # Victoire
            elif result == 'D':
                points = 1  # Nul
            else:
                points = 0  # Défaite
                
            total_points += points
            
        # Points par match (sur 3 max)
        avg_points = total_points / len(recent_matches)
        form_normalized = avg_points / 3.0  # Normaliser entre 0 et 1
        
        return form_normalized
        
    def calculate_dynamic_elo(self, team, before_date):
        """Calculer l'Elo dynamique d'une équipe à une date donnée"""
        
        # Vérifier cache
        cache_key = f"{team}_{before_date.strftime('%Y%m%d')}"
        if cache_key in self.elo_cache:
            return self.elo_cache[cache_key]
            
        # Obtenir tous les matchs de l'équipe avant la date
        team_matches = self.get_team_matches_before_date(team, before_date)
        
        if len(team_matches) == 0:
            # Nouvelle équipe ou pas d'historique
            elo = self.initial_elo
        else:
            # Commencer avec Elo initial et simuler les updates
            elo = self.initial_elo
            
            # Traiter les matchs dans l'ordre chronologique
            team_matches_chrono = team_matches.sort_values('Date')
            
            for _, match in team_matches_chrono.iterrows():
                is_home = match['HomeTeam'] == team
                opponent = match['AwayTeam'] if is_home else match['HomeTeam']
                result = match['FullTimeResult']
                
                # Elo de l'adversaire (récursif si nécessaire, sinon initial)
                opponent_elo = self.initial_elo  # Simplification pour performance
                
                # Score réel du match pour cette équipe
                if (is_home and result == 'H') or (not is_home and result == 'A'):
                    score = 1.0  # Victoire
                elif result == 'D':
                    score = 0.5  # Nul
                else:
                    score = 0.0  # Défaite
                    
                # Avantage du terrain
                if is_home:
                    effective_elo = elo + 50  # Bonus domicile
                else:
                    effective_elo = elo
                    
                # Probabilité attendue
                expected = 1 / (1 + 10**((opponent_elo - effective_elo) / 400))
                
                # Mise à jour Elo
                elo = elo + self.elo_k_factor * (score - expected)
                
        # Sauvegarder en cache
        self.elo_cache[cache_key] = elo
        
        return elo
        
    def calculate_dynamic_xg_efficiency(self, team, before_date, n_matches=10):
        """Calculer l'efficacité xG dynamique"""
        
        recent_matches = self.get_team_matches_before_date(team, before_date, n_matches)
        
        if len(recent_matches) == 0:
            return 0.5  # Efficacité neutre
            
        total_goals = 0
        total_xg = 0
        
        for _, match in recent_matches.iterrows():
            is_home = match['HomeTeam'] == team
            
            # Goals marqués
            if is_home:
                goals = match.get('FTHG', 0)
                xg = match.get('home_xg', goals * 0.1)  # Fallback si pas de xG
            else:
                goals = match.get('FTAG', 0)  
                xg = match.get('away_xg', goals * 0.1)  # Fallback
                
            total_goals += goals
            total_xg += max(xg, 0.1)  # Éviter division par 0
            
        # Efficacité = Goals / xG attendus
        efficiency = total_goals / total_xg if total_xg > 0 else 1.0
        
        # Normaliser autour de 1.0 (100% efficiency)
        efficiency_normalized = min(efficiency, 2.0) / 2.0  # Cap à 200% efficiency
        
        return efficiency_normalized
        
    def calculate_dynamic_h2h(self, home_team, away_team, before_date, n_matches=5):
        """Calculer le score H2H dynamique"""
        
        # Matchs H2H entre ces équipes avant la date
        h2h_matches = self.dataset[
            (((self.dataset['HomeTeam'] == home_team) & (self.dataset['AwayTeam'] == away_team)) |
             ((self.dataset['HomeTeam'] == away_team) & (self.dataset['AwayTeam'] == home_team))) &
            (self.dataset['Date'] < before_date)
        ].copy()
        
        h2h_matches = h2h_matches.sort_values('Date', ascending=False).head(n_matches)
        
        if len(h2h_matches) == 0:
            return 0.5  # Score neutre
            
        home_team_points = 0
        total_matches = len(h2h_matches)
        
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team:
                # home_team jouait à domicile
                if match['FullTimeResult'] == 'H':
                    home_team_points += 3
                elif match['FullTimeResult'] == 'D':
                    home_team_points += 1
            else:
                # home_team jouait à l'extérieur
                if match['FullTimeResult'] == 'A':
                    home_team_points += 3
                elif match['FullTimeResult'] == 'D':
                    home_team_points += 1
                    
        # Score H2H normalisé
        h2h_score = home_team_points / (total_matches * 3)
        
        return h2h_score
        
    def calculate_dynamic_shots_corners_trend(self, team, before_date, n_matches=5):
        """Calculer les tendances récentes shots/corners"""
        
        recent_matches = self.get_team_matches_before_date(team, before_date, n_matches)
        
        if len(recent_matches) == 0:
            return 0.5, 0.5  # Tendances neutres
            
        total_shots_for = 0
        total_shots_against = 0
        total_corners_for = 0
        total_corners_against = 0
        
        for _, match in recent_matches.iterrows():
            is_home = match['HomeTeam'] == team
            
            # Shots
            shots_for = match.get('HS' if is_home else 'AS', 10)  # Fallback 10 shots
            shots_against = match.get('AS' if is_home else 'HS', 10)
            
            # Corners  
            corners_for = match.get('HC' if is_home else 'AC', 5)  # Fallback 5 corners
            corners_against = match.get('AC' if is_home else 'HC', 5)
            
            total_shots_for += shots_for
            total_shots_against += shots_against
            total_corners_for += corners_for
            total_corners_against += corners_against
            
        # Ratios normalisés
        shots_ratio = total_shots_for / (total_shots_for + total_shots_against) if (total_shots_for + total_shots_against) > 0 else 0.5
        corners_ratio = total_corners_for / (total_corners_for + total_corners_against) if (total_corners_for + total_corners_against) > 0 else 0.5
        
        return shots_ratio, corners_ratio
        
    def calculate_all_dynamic_features(self, home_team, away_team, match_date):
        """Calculer toutes les features dynamiques pour un match"""
        
        print(f"🔄 Calcul features dynamiques: {home_team} vs {away_team} ({match_date.strftime('%d/%m/%Y')})")
        
        # 1. FORM DYNAMIQUE
        home_form = self.calculate_dynamic_form(home_team, match_date, self.form_window)
        away_form = self.calculate_dynamic_form(away_team, match_date, self.form_window)
        form_diff_normalized = (home_form - away_form + 1) / 2  # Normaliser entre 0 et 1
        
        # 2. ELO DYNAMIQUE
        home_elo = self.calculate_dynamic_elo(home_team, match_date)
        away_elo = self.calculate_dynamic_elo(away_team, match_date)
        elo_diff_raw = home_elo - away_elo
        elo_diff_normalized = (elo_diff_raw + 400) / 800  # Normaliser entre 0 et 1 (diff ±400)
        
        # 3. xG EFFICIENCY DYNAMIQUE
        home_xg_eff = self.calculate_dynamic_xg_efficiency(home_team, match_date, self.xg_window)
        away_xg_eff = self.calculate_dynamic_xg_efficiency(away_team, match_date, self.xg_window)
        
        # 4. H2H DYNAMIQUE
        h2h_score = self.calculate_dynamic_h2h(home_team, away_team, match_date)
        
        # 5. SHOTS/CORNERS TENDANCES
        home_shots_ratio, home_corners_ratio = self.calculate_dynamic_shots_corners_trend(home_team, match_date)
        away_shots_ratio, away_corners_ratio = self.calculate_dynamic_shots_corners_trend(away_team, match_date)
        
        shots_diff_normalized = (home_shots_ratio - away_shots_ratio + 1) / 2
        corners_diff_normalized = (home_corners_ratio - away_corners_ratio + 1) / 2
        
        # 6. AUTRES FEATURES NÉCESSAIRES
        
        # Matchday normalisé (position dans la saison)
        season_start = pd.to_datetime(f"{match_date.year if match_date.month >= 8 else match_date.year-1}-08-01")
        days_since_start = (match_date - season_start).days
        matchday_normalized = min(days_since_start / 300, 1.0)  # Normaliser sur ~10 mois
        
        # Market entropy (proxy basé sur Elo diff)
        elo_uncertainty = 1 - abs(elo_diff_raw) / 400  # Plus l'écart est faible, plus l'incertitude est haute
        market_entropy_norm = max(0.5, elo_uncertainty)  # Minimum 50% entropy
        
        # Away goals sum (derniers 5 matchs away team)
        away_recent = self.get_team_matches_before_date(away_team, match_date, 5)
        away_goals_sum = 0
        for _, match in away_recent.iterrows():
            if match['AwayTeam'] == away_team:
                away_goals_sum += match.get('FTAG', 1)  # Fallback 1 goal
            else:
                away_goals_sum += match.get('FTHG', 1)
        away_goals_sum_5 = max(away_goals_sum, 1)  # Minimum 1 pour éviter 0
        
        # Assembler toutes les features dans l'ORDRE CORRECT v2.3
        dynamic_features = {
            'form_diff_normalized': form_diff_normalized,
            'elo_diff_normalized': elo_diff_normalized, 
            'h2h_score': h2h_score,
            'matchday_normalized': matchday_normalized,
            'shots_diff_normalized': shots_diff_normalized,
            'corners_diff_normalized': corners_diff_normalized,
            'market_entropy_norm': market_entropy_norm,
            'home_xg_eff_10': home_xg_eff,
            'away_goals_sum_5': away_goals_sum_5,
            'away_xg_eff_10': away_xg_eff
        }
        
        print(f"   ✅ Features calculées:")
        for feature, value in dynamic_features.items():
            print(f"      {feature}: {value:.3f}")
            
        return dynamic_features
        
    def calculate_features_for_matches(self, matches_df):
        """Calculer features dynamiques pour plusieurs matchs"""
        
        print(f"\n📊 CALCUL FEATURES DYNAMIQUES POUR {len(matches_df)} MATCHES")
        print("-" * 60)
        
        all_features = []
        
        for idx, match in matches_df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            match_date = pd.to_datetime(match['Date'])
            
            # Calculer features dynamiques pour ce match
            features = self.calculate_all_dynamic_features(home_team, away_team, match_date)
            
            # Ajouter métadonnées
            features['Match_ID'] = idx
            features['Date'] = match_date
            features['HomeTeam'] = home_team
            features['AwayTeam'] = away_team
            
            if 'FullTimeResult' in match:
                features['FullTimeResult'] = match['FullTimeResult']
            elif 'FTR' in match:
                features['FullTimeResult'] = match['FTR']
                
            all_features.append(features)
            
        # Convertir en DataFrame
        features_df = pd.DataFrame(all_features)
        
        print(f"✅ Features dynamiques calculées pour tous les matches")
        print(f"   Colonnes: {len(features_df.columns)}")
        print(f"   Features: {[col for col in features_df.columns if col not in ['Match_ID', 'Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult']]}")
        
        return features_df

def main():
    """Test du calculateur de features dynamiques"""
    
    # Configuration
    dataset_path = "data/processed/v15_final_enhanced.csv"
    
    # Initialiser calculateur
    calculator = DynamicFeaturesCalculator(dataset_path)
    
    # Charger données
    if not calculator.load_historical_data():
        return None
        
    # Test sur un match spécifique
    print(f"\n🧪 TEST SUR UN MATCH SPÉCIFIQUE")
    print("-" * 40)
    
    test_date = pd.to_datetime("2025-08-15")  # Premier match EPL 2025-26
    features = calculator.calculate_all_dynamic_features("Liverpool", "Bournemouth", test_date)
    
    print(f"\n🎯 Features dynamiques pour Liverpool vs Bournemouth (15/08/2025):")
    for feature, value in features.items():
        print(f"   {feature}: {value:.3f}")
        
    return calculator

if __name__ == "__main__":
    calculator = main()