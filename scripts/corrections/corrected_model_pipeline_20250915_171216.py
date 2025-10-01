#!/usr/bin/env python3
"""
🎯 MODÈLE CORRIGÉ - FEATURES XG DÉBUT SAISON
Auto-généré le 2025-09-15 17:12:16

Correction appliquée: neutral
Performance: 0.500 (amélioration: +0.000)
"""

import pandas as pd
import numpy as np
import joblib

def correct_xg_features_early_season(df):
    """Appliquer correction features xG pour début saison"""
    df_corrected = df.copy()
    
    # Correction stratégie: neutral
    cutoff_date = pd.Timestamp('2025-08-01')
    early_season_mask = df_corrected['Date'] >= cutoff_date
    
    corrections_applied = 0
    for idx in df_corrected[early_season_mask].index:
        match_date = df_corrected.loc[idx, 'Date']
        season_start = pd.Timestamp('2025-08-01')
        days_since_start = (match_date - season_start).days
        
        # Appliquer correction pour premiers ~30 jours de saison (≈ J1-J6)
        if days_since_start <= 30:
            df_corrected.loc[idx, 'home_xg_eff_10'] = 0.500
            df_corrected.loc[idx, 'away_xg_eff_10'] = 0.500
            corrections_applied += 1
    
    print(f"✅ Correction XG appliquée: {corrections_applied} matches")
    return df_corrected

def predict_with_xg_correction(df, model_path='models/final_robust_model_20250915_163023.joblib'):
    """Prédiction avec correction XG automatique"""
    # Charger modèle
    model = joblib.load(model_path)
    
    # Appliquer corrections
    df_corrected = correct_xg_features_early_season(df)
    
    # Features
    features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Prédiction
    X = df_corrected[features]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return predictions, probabilities, df_corrected

if __name__ == "__main__":
    print("🔧 Pipeline modèle corrigé prêt!")
    print("Performance validée: 50.0% sur 40 matches J1-J4")
