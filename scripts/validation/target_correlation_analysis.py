#!/usr/bin/env python3
"""
🚨 TARGET CORRELATION ANALYSIS - DÉTECTION FINALE LEAKAGE

Analyse de corrélation entre features et target pour détecter data leakage.
Corrélation > 0.4 avec une feature = LEAKAGE QUASI CERTAIN

Objectif: Confirmer/infirmer les résultats de l'ablation test
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def analyze_target_correlations():
    """
    Analyser corrélations features-target pour détecter leakage
    """
    
    print("🚨 TARGET CORRELATION ANALYSIS - DÉTECTION FINALE")
    print("=" * 65)
    
    # Charger test set hermétique
    test_df = pd.read_csv('data/validation/test_set_epl_2025_26_hermetic.csv', parse_dates=['Date'])
    
    print(f"📊 Analyse sur {len(test_df)} matches EPL 2025-26")
    
    features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Encoder target numériquement
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    test_df['target_numeric'] = test_df['FullTimeResult'].map(target_mapping)
    
    print()
    print("🔍 CORRÉLATIONS FEATURES-TARGET:")
    print("-" * 65)
    
    correlations = {}
    suspicious_features = []
    
    for feature in features:
        if feature in test_df.columns:
            # Corrélation de Pearson
            corr, p_value = pearsonr(test_df[feature], test_df['target_numeric'])
            correlations[feature] = {
                'correlation': corr,
                'p_value': p_value,
                'abs_corr': abs(corr)
            }
            
            # Status
            if abs(corr) > 0.5:
                status = "🚨 TRÈS SUSPECT"
                suspicious_features.append(feature)
            elif abs(corr) > 0.3:
                status = "⚠️ SUSPECT"
                suspicious_features.append(feature)
            else:
                status = "✅ OK"
            
            print(f"{feature:25}: {corr:+.3f} (p={p_value:.3f}) {status}")
    
    print()
    print("📊 ANALYSE DÉTAILLÉE DES FEATURES SUSPECTES:")
    print("-" * 65)
    
    # Analyser distribution des features suspectes par résultat
    for feature in suspicious_features:
        print(f"\\n🔍 ANALYSE: {feature}")
        
        # Statistiques par classe
        for result_class in ['H', 'D', 'A']:
            subset = test_df[test_df['FullTimeResult'] == result_class]
            if len(subset) > 0:
                mean_val = subset[feature].mean()
                print(f"  {result_class}: {mean_val:.3f} (n={len(subset)})")
        
        # Vérifier si feature prédit parfaitement certains résultats
        feature_vals = test_df[feature]
        unique_vals = feature_vals.unique()
        
        perfect_prediction_count = 0
        for val in unique_vals:
            matches_with_val = test_df[test_df[feature] == val]
            if len(matches_with_val) > 1:  # Plus d'un match
                results = matches_with_val['FullTimeResult'].unique()
                if len(results) == 1:  # Tous le même résultat
                    perfect_prediction_count += len(matches_with_val)
        
        if perfect_prediction_count > 0:
            print(f"  🚨 PRÉDICTION PARFAITE: {perfect_prediction_count}/{len(test_df)} matches avec même feature → même résultat")
    
    # AUDIT SPÉCIAL: shots_diff et corners_diff
    print("\\n" + "=" * 65)
    print("🎯 AUDIT SPÉCIAL - FEATURES TRADITIONNELLES SUSPECTES")
    print("-" * 65)
    
    for feature in ['shots_diff_normalized', 'corners_diff_normalized']:
        if feature in suspicious_features:
            print(f"\\n🔍 INVESTIGATION: {feature}")
            
            # Regarder distribution détaillée
            feature_data = test_df[[feature, 'FullTimeResult', 'HomeTeam', 'AwayTeam']].copy()
            
            print("Top valeurs avec résultats:")
            for _, row in feature_data.head(10).iterrows():
                print(f"  {row['HomeTeam']} vs {row['AwayTeam']}: {feature}={row[feature]:.3f} → {row['FullTimeResult']}")
    
    # VERDICT FINAL
    print("\\n" + "=" * 65)
    print("⚖️ VERDICT CORRÉLATION ANALYSIS:")
    print("-" * 65)
    
    max_correlation = max([abs(c['correlation']) for c in correlations.values()])
    high_corr_features = [f for f, data in correlations.items() if data['abs_corr'] > 0.3]
    
    print(f"\\nMax corrélation absolue: {max_correlation:.3f}")
    print(f"Features avec corrélation > 0.3: {len(high_corr_features)}")
    
    if len(high_corr_features) > 0:
        print(f"Features suspectes: {high_corr_features}")
    
    if max_correlation > 0.5:
        print("\\n🚨 LEAKAGE DÉTECTÉ!")
        print("Corrélation très élevée détectée - investigation approfondie requise.")
    elif max_correlation > 0.3:
        print("\\n⚠️ LEAKAGE POSSIBLE")
        print("Corrélations modérées détectées - surveillance recommandée.")
    else:
        print("\\n✅ CORRÉLATIONS NORMALES")
        print("Aucune corrélation suspecte détectée.")
    
    # COMPARAISON AVEC ÉCHANTILLON HISTORIQUE
    print("\\n📊 COMPARAISON AVEC DONNÉES HISTORIQUES:")
    train_df = pd.read_csv('data/validation/train_set_2019_2025_hermetic.csv', parse_dates=['Date'])
    train_df['target_numeric'] = train_df['FullTimeResult'].map(target_mapping)
    
    # Prendre échantillon aléatoire de 39 matches historiques
    train_sample = train_df.sample(n=39, random_state=42)
    
    print("Corrélations échantillon historique vs EPL 2025-26:")
    for feature in high_corr_features:
        if feature in train_sample.columns:
            hist_corr, _ = pearsonr(train_sample[feature], train_sample['target_numeric'])
            epl_corr = correlations[feature]['correlation']
            diff = abs(epl_corr) - abs(hist_corr)
            
            print(f"{feature:25}: Hist={hist_corr:+.3f} vs EPL={epl_corr:+.3f} (diff={diff:+.3f})")
            
            if abs(diff) > 0.2:
                print(f"  🚨 DIFFÉRENCE MAJEURE! Corrélation anormalement différente.")
    
    return correlations, suspicious_features

if __name__ == "__main__":
    correlations, suspicious_features = analyze_target_correlations()