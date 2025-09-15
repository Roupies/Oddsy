#!/usr/bin/env python3
"""
CASCADE DRAW DETECTOR - Amélioration v2.3 Dynamique
===================================================

STRATÉGIE: Modèle cascade en 2 étapes
1. ÉTAPE 1: Binary classifier Draw vs NotDraw 
2. ÉTAPE 2: Si NotDraw → v2.3 pour prédire H vs A

OBJECTIF: Récupérer les 6 draws ratés pour passer de 50% à 55%+ accuracy
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from datetime import datetime
import sys
import os

# Importer le calculateur dynamique
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.dynamic_features_calculator import DynamicFeaturesCalculator

import warnings
warnings.filterwarnings('ignore')

class CascadeDrawDetector:
    """Détecteur cascade pour améliorer prédiction draws"""
    
    def __init__(self):
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.v23_model_path = "models/v23_retrained_2025_09_11_154613.joblib"
        
        # Features spécialisées draw
        self.draw_features = [
            'teams_balance_ratio',      # Ratio équilibre forces
            'defensive_stability_both', # Moyenne solidité défensive
            'recent_draw_frequency',    # Fréquence draws récente
            'form_convergence',         # Convergence des formes
            'elo_equilibrium',          # Proximité Elo ratings
            'h2h_draw_tendency',        # Tendance draw H2H
            'goals_scored_balance',     # Équilibre attaques
            'market_uncertainty_high'   # Incertitude marché élevée
        ]
        
        # Objets
        self.features_calculator = None
        self.v23_model = None
        self.draw_detector = None
        
    def initialize(self):
        """Initialiser les composants"""
        
        print("🎯 CASCADE DRAW DETECTOR - INITIALISATION")
        print("=" * 55)
        
        # Calculateur features dynamiques
        print("📊 Chargement calculateur features dynamiques...")
        self.features_calculator = DynamicFeaturesCalculator(self.dataset_path)
        if not self.features_calculator.load_historical_data():
            return False
            
        # Modèle v2.3 original
        print("🤖 Chargement modèle v2.3...")
        self.v23_model = joblib.load(self.v23_model_path)
        
        return True
        
    def calculate_specialized_draw_features(self, home_team, away_team, match_date):
        """Calculer features spécialisées pour détection draw"""
        
        # Features v2.3 de base
        base_features = self.features_calculator.calculate_all_dynamic_features(
            home_team, away_team, match_date
        )
        
        # Calculer features spécialisées draw
        draw_features = {}
        
        # 1. Teams balance ratio
        elo_home = self.features_calculator.calculate_dynamic_elo(home_team, match_date)
        elo_away = self.features_calculator.calculate_dynamic_elo(away_team, match_date)
        elo_total = elo_home + elo_away
        teams_balance = abs(0.5 - (elo_home / elo_total)) if elo_total > 0 else 0.5
        draw_features['teams_balance_ratio'] = 1.0 - teams_balance
        
        # 2. Defensive stability
        home_form = self.features_calculator.calculate_dynamic_form(home_team, match_date, 5)
        away_form = self.features_calculator.calculate_dynamic_form(away_team, match_date, 5)
        defensive_stability = (home_form + away_form) / 2
        draw_features['defensive_stability_both'] = defensive_stability
        
        # 3. Recent draw frequency
        home_draws = self._calculate_recent_draws(home_team, match_date, 5)
        away_draws = self._calculate_recent_draws(away_team, match_date, 5)
        draw_features['recent_draw_frequency'] = (home_draws + away_draws) / 2
        
        # 4. Form convergence
        form_diff = abs(home_form - away_form)
        draw_features['form_convergence'] = 1.0 - form_diff
        
        # 5. Elo equilibrium
        elo_diff_abs = abs(elo_home - elo_away)
        elo_equilibrium = max(0, 1.0 - (elo_diff_abs / 200))
        draw_features['elo_equilibrium'] = elo_equilibrium
        
        # 6. H2H draw tendency
        h2h_draws = self._calculate_h2h_draws(home_team, away_team, match_date)
        draw_features['h2h_draw_tendency'] = h2h_draws
        
        # 7. Goals scored balance
        home_goals = self._calculate_recent_goals(home_team, match_date, 5)
        away_goals = self._calculate_recent_goals(away_team, match_date, 5)
        goal_balance = 1.0 - abs(home_goals - away_goals) / max(home_goals + away_goals, 1)
        draw_features['goals_scored_balance'] = goal_balance
        
        # 8. Market uncertainty
        market_entropy = base_features.get('market_entropy_norm', 0.5)
        draw_features['market_uncertainty_high'] = market_entropy
        
        return draw_features
        
    def _calculate_recent_draws(self, team, before_date, n_matches=5):
        """Calculer fréquence draws récente d'une équipe"""
        
        recent_matches = self.features_calculator.get_team_matches_before_date(
            team, before_date, n_matches
        )
        
        if len(recent_matches) == 0:
            return 0.2  # Baseline draw frequency
            
        draws = sum(1 for _, match in recent_matches.iterrows() 
                   if match['FullTimeResult'] == 'D')
        
        return draws / len(recent_matches)
        
    def _calculate_h2h_draws(self, home_team, away_team, before_date, n_matches=10):
        """Calculer tendance draw dans confrontations H2H"""
        
        h2h_matches = self.features_calculator.dataset[
            (((self.features_calculator.dataset['HomeTeam'] == home_team) & 
              (self.features_calculator.dataset['AwayTeam'] == away_team)) |
             ((self.features_calculator.dataset['HomeTeam'] == away_team) & 
              (self.features_calculator.dataset['AwayTeam'] == home_team))) &
            (self.features_calculator.dataset['Date'] < before_date)
        ].sort_values('Date', ascending=False).head(n_matches)
        
        if len(h2h_matches) == 0:
            return 0.2  # Baseline
            
        draws = sum(1 for _, match in h2h_matches.iterrows() 
                   if match['FullTimeResult'] == 'D')
        
        return draws / len(h2h_matches)
        
    def _calculate_recent_goals(self, team, before_date, n_matches=5):
        """Calculer moyenne buts récents d'une équipe"""
        
        recent_matches = self.features_calculator.get_team_matches_before_date(
            team, before_date, n_matches
        )
        
        if len(recent_matches) == 0:
            return 1.5  # Baseline goals per match
            
        total_goals = 0
        for _, match in recent_matches.iterrows():
            # Utiliser colonnes FTR comme proxy (pas FTHG/FTAG disponibles)
            if team == match['HomeTeam']:
                # Estimer goals based on result
                if match['FullTimeResult'] == 'H':
                    goals_for = 2  # Victoire moyenne
                elif match['FullTimeResult'] == 'D':
                    goals_for = 1  # Draw moyenne
                else:
                    goals_for = 0.5  # Défaite moyenne
            else:
                # Away team
                if match['FullTimeResult'] == 'A':
                    goals_for = 2
                elif match['FullTimeResult'] == 'D':
                    goals_for = 1
                else:
                    goals_for = 0.5
                    
            total_goals += goals_for
            
        return total_goals / len(recent_matches)
        
    def prepare_training_data(self, season_split='2024-2025'):
        """Préparer données d'entraînement pour le détecteur draw"""
        
        print(f"\n📊 PRÉPARATION DONNÉES ENTRAÎNEMENT")
        print("-" * 45)
        
        # Charger dataset
        dataset = pd.read_csv(self.dataset_path)
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        
        # Séparer train/test temporellement
        train_data = dataset[dataset['Season'] < season_split].copy()
        test_data = dataset[dataset['Season'] >= season_split].copy()
        
        print(f"📅 Split temporel:")
        print(f"   Train: {len(train_data)} matches (< {season_split})")
        print(f"   Test: {len(test_data)} matches (>= {season_split})")
        
        # Préparer features draw pour train
        print(f"\n🔧 Calcul features draw training...")
        X_train_draw = []
        y_train_draw = []
        
        # Traiter 200 premiers matches pour test
        for idx, match in train_data.head(200).iterrows():
            try:
                draw_features = self.calculate_specialized_draw_features(
                    match['HomeTeam'], match['AwayTeam'], match['Date']
                )
                
                X_train_draw.append(list(draw_features.values()))
                y_train_draw.append(1 if match['FullTimeResult'] == 'D' else 0)
                
                if (idx + 1) % 50 == 0:
                    print(f"   Processé: {idx+1}/200 matches")
                    
            except Exception as e:
                print(f"   ❌ Erreur match {idx}: {e}")
                continue
                
        X_train_draw = np.array(X_train_draw)
        y_train_draw = np.array(y_train_draw)
        
        print(f"\n✅ Training data préparé:")
        print(f"   Features shape: {X_train_draw.shape}")
        print(f"   Labels: {len(y_train_draw)} (Draws: {sum(y_train_draw)}, NotDraws: {len(y_train_draw) - sum(y_train_draw)})")
        
        return X_train_draw, y_train_draw, test_data
        
    def train_draw_detector(self, X_train, y_train):
        """Entraîner le détecteur binary draw"""
        
        print(f"\n🤖 ENTRAÎNEMENT DÉTECTEUR DRAW")
        print("-" * 40)
        
        # Modèle spécialisé draw
        base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Important pour équilibrer Draw vs NotDraw
            random_state=42
        )
        
        # Calibration des probabilités
        self.draw_detector = CalibratedClassifierCV(
            base_model, 
            method='isotonic', 
            cv=3
        )
        
        # Entraînement
        print(f"📊 Entraînement sur {len(X_train)} échantillons...")
        self.draw_detector.fit(X_train, y_train)
        
        # Validation cross-temporelle
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
            
            temp_model = CalibratedClassifierCV(base_model, method='isotonic', cv=2)
            temp_model.fit(X_train_cv, y_train_cv)
            
            val_score = temp_model.score(X_val_cv, y_val_cv)
            cv_scores.append(val_score)
            
        avg_cv_score = np.mean(cv_scores)
        
        print(f"✅ Détecteur draw entraîné")
        print(f"   CV Score: {avg_cv_score:.3f} ± {np.std(cv_scores):.3f}")
        
        return avg_cv_score
        
    def save_cascade_model(self):
        """Sauvegarder le modèle cascade"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path("models/cascade")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder détecteur draw
        draw_model_path = model_dir / f"cascade_draw_detector_{timestamp}.joblib"
        joblib.dump(self.draw_detector, draw_model_path)
        
        print(f"💾 Modèle cascade sauvegardé:")
        print(f"   Draw detector: {draw_model_path}")
        
        return draw_model_path

def main():
    """Test du cascade draw detector"""
    
    detector = CascadeDrawDetector()
    
    # Initialisation
    if not detector.initialize():
        print("❌ Échec initialisation")
        return
        
    # Préparer données
    X_train, y_train, test_data = detector.prepare_training_data()
    
    if len(X_train) == 0:
        print("❌ Pas de données d'entraînement")
        return
        
    # Entraîner détecteur
    cv_score = detector.train_draw_detector(X_train, y_train)
    
    # Sauvegarder
    model_path = detector.save_cascade_model()
    
    print(f"\n🏆 CASCADE DRAW DETECTOR CRÉÉ!")
    print(f"📊 Performance CV: {cv_score:.1%}")
    print(f"📄 Modèle: {model_path}")
    
    return detector

if __name__ == "__main__":
    detector = main()