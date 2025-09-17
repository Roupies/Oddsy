#!/usr/bin/env python3
"""
🚨 AUDIT D'URGENCE - HERMETIC DATA SPLIT

Créer split temporel absolument étanche pour éliminer tout data leakage potentiel.
Performance 71.8% suspecte - investigation nécessaire.

OBJECTIF: Prouver/refuter que les 71.8% sont dus à data leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_hermetic_split():
    """
    Créer split temporel hermétique sans aucune contamination possible
    """
    
    print("🚨 AUDIT D'URGENCE - HERMETIC DATA SPLIT")
    print("=" * 60)
    
    # Charger le dataset principal
    print("📊 Chargement dataset principal...")
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    print(f"Dataset total: {len(df)} matches")
    
    # Charger les 40 matches EPL 2025-26
    print("📊 Chargement 40 matches EPL 2025-26...")
    df_40 = pd.read_csv('data/processed/premier_league_2025_26_all_matches_played.csv')
    df_40['Date'] = pd.to_datetime(df_40['Date'])
    print(f"EPL 2025-26: {len(df_40)} matches")
    
    # DATE DE COUPURE STRICTE
    cutoff_date = pd.Timestamp('2025-08-01')
    print(f"📅 Date de coupure: {cutoff_date}")
    
    # SPLIT HERMÉTIQUE
    train_set = df[df['Date'] < cutoff_date].copy()
    print(f"🎓 TRAIN SET: {len(train_set)} matches (2019-{cutoff_date})")
    print(f"   Date range: {train_set['Date'].min()} à {train_set['Date'].max()}")
    
    # TEST SET = EPL 2025-26 uniquement
    test_set = df_40.copy()
    print(f"🧪 TEST SET: {len(test_set)} matches (EPL 2025-26)")
    print(f"   Date range: {test_set['Date'].min()} à {test_set['Date'].max()}")
    
    # VALIDATION: Aucun overlap
    overlap_check = train_set['Date'].max() >= test_set['Date'].min()
    if overlap_check:
        print("❌ ERREUR: Overlap détecté entre train/test!")
        return False
    else:
        print("✅ AUCUN OVERLAP - Split hermétique confirmé")
    
    # Créer répertoire de validation
    os.makedirs('data/validation', exist_ok=True)
    
    # Sauvegarder splits hermétiques
    train_path = 'data/validation/train_set_2019_2025_hermetic.csv'
    test_path = 'data/validation/test_set_epl_2025_26_hermetic.csv'
    
    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)
    
    print(f"💾 Train set sauvé: {train_path}")
    print(f"💾 Test set sauvé: {test_path}")
    
    # AUDIT FEATURES - vérifier disponibilité
    features_v23 = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    print()
    print("🔍 AUDIT FEATURES DISPONIBLES:")
    train_features = [f for f in features_v23 if f in train_set.columns]
    test_features = [f for f in features_v23 if f in test_set.columns]
    
    print(f"Train set: {len(train_features)}/10 features v2.3")
    print(f"Test set: {len(test_features)}/10 features v2.3")
    
    missing_train = set(features_v23) - set(train_features)
    missing_test = set(features_v23) - set(test_features)
    
    if missing_train:
        print(f"❌ Features manquantes train: {missing_train}")
    if missing_test:
        print(f"❌ Features manquantes test: {missing_test}")
        
    if len(train_features) == 10 and len(test_features) == 10:
        print("✅ Toutes les features v2.3 disponibles")
    
    # TARGETS DISPONIBLES
    print()
    print("🎯 AUDIT TARGETS:")
    if 'FullTimeResult' in train_set.columns and 'FullTimeResult' in test_set.columns:
        train_dist = train_set['FullTimeResult'].value_counts()
        test_dist = test_set['FullTimeResult'].value_counts()
        
        print("Distribution train set:")
        for result, count in train_dist.items():
            print(f"  {result}: {count} ({count/len(train_set)*100:.1f}%)")
            
        print("Distribution test set:")
        for result, count in test_dist.items():
            print(f"  {result}: {count} ({count/len(test_set)*100:.1f}%)")
        
        print("✅ Targets disponibles dans les deux sets")
    else:
        print("❌ ERREUR: FullTimeResult manquant!")
    
    print()
    print("🚨 SPLIT HERMÉTIQUE CRÉÉ - PRÊT POUR AUDIT")
    print("📋 Prochaine étape: Feature ablation test")
    
    return True

if __name__ == "__main__":
    create_hermetic_split()