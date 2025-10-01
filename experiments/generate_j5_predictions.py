#!/usr/bin/env python3
"""
🔮 Génération des prédictions réelles pour la J5 EPL 2025-26
Avec scores de confiance calculés par les modèles champions
"""

import pandas as pd
import joblib
import numpy as np
from datetime import datetime

def load_j5_matches():
    """Charge les matchs de la J5."""
    df = pd.read_csv('data/processed/v_auto_update_20250916_110247.csv')
    
    # Filtre J5 (13-14 septembre 2025)
    j5_matches = df[df['Date'].isin(['2025-09-13', '2025-09-14'])].copy()
    
    # Sélection des features de production
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    return j5_matches, features

def generate_real_predictions():
    """Génère les prédictions réelles avec les modèles champions."""
    
    print("🔮 Génération des prédictions J5 avec modèles champions...")
    
    # Chargement des données
    j5_matches, features = load_j5_matches()
    
    if j5_matches.empty:
        print("❌ Aucun match J5 trouvé")
        return []
    
    # Chargement du modèle Baseline Champion
    try:
        baseline_model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Baseline Champion chargé")
    except:
        baseline_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
        print("✅ Fallback model chargé")
    
    predictions = []
    
    for idx, match in j5_matches.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        date = match['Date']
        
        # Préparation des features
        X = np.array([match[features].values])
        
        # Prédiction avec probabilités
        try:
            proba = baseline_model.predict_proba(X)[0]
            pred_class = baseline_model.predict(X)[0]
            
            # Conversion des classes
            class_mapping = {0: 'H', 1: 'D', 2: 'A'}
            pred_label = class_mapping[pred_class]
            confidence = proba[pred_class]
            
            # Création du match
            match_info = {
                'Match': f"{home_team} vs {away_team}",
                'Date': date,
                'Model_Used': 'Baseline Champion',
                'Final_Pred': pred_label,
                'Final_Conf': confidence,
                'Prob_H': proba[0],
                'Prob_D': proba[1], 
                'Prob_A': proba[2]
            }
            
            predictions.append(match_info)
            print(f"✅ {home_team} vs {away_team}: {pred_label} ({confidence:.1%})")
            
        except Exception as e:
            print(f"❌ Erreur pour {home_team} vs {away_team}: {e}")
    
    return predictions[:5]  # Top 5 matchs

if __name__ == "__main__":
    predictions = generate_real_predictions()
    
    print(f"\n🎯 {len(predictions)} prédictions J5 générées:")
    for p in predictions:
        print(f"⚽ {p['Match']}: {p['Final_Pred']} ({p['Final_Conf']:.1%})")
        print(f"   📊 H:{p['Prob_H']:.1%} D:{p['Prob_D']:.1%} A:{p['Prob_A']:.1%}")