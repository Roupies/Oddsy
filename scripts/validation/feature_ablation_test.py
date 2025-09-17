#!/usr/bin/env python3
"""
🚨 FEATURE ABLATION TEST - DÉTECTION DATA LEAKAGE

Test d'ablation pour identifier quelles features causent la performance suspecte de 71.8%.
Hypothèse: Features complexes (market, xG) contiennent info du futur.

MÉTHODOLOGIE:
1. Test SANS features suspectes → Si performance chute = LEAKAGE DÉTECTÉ
2. Test chaque feature individuellement 
3. Corrélation avec target pour confirmation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def feature_ablation_test():
    """
    Test d'ablation pour détecter data leakage dans features
    """
    
    print("🚨 FEATURE ABLATION TEST - DÉTECTION LEAKAGE")
    print("=" * 70)
    
    # Charger splits hermétiques
    print("📊 Chargement splits hermétiques...")
    train_df = pd.read_csv('data/validation/train_set_2019_2025_hermetic.csv', parse_dates=['Date'])
    test_df = pd.read_csv('data/validation/test_set_epl_2025_26_hermetic.csv', parse_dates=['Date'])
    
    print(f"Train: {len(train_df)} matches (2019-2025)")
    print(f"Test: {len(test_df)} matches (EPL 2025-26)")
    
    # Features v2.3
    all_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Groupes de features suspects
    suspect_features = {
        'market_intelligence': ['market_entropy_norm'],
        'xg_features': ['home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'],
        'advanced_features': ['market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10'],
        'traditional_only': ['form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 
                            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized']
    }
    
    # Préparer données
    X_train_full = train_df[all_features]
    y_train = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    X_test_full = test_df[all_features]
    y_test = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print()
    print("🔬 TESTS D'ABLATION SYSTÉMATIQUES:")
    print("-" * 70)
    
    results = {}
    
    # TEST 1: Performance baseline avec TOUTES les features
    print("\\n🎯 TEST BASELINE - TOUTES FEATURES:")
    model_baseline = RandomForestClassifier(
        n_estimators=300, max_depth=20, max_features='sqrt',
        min_samples_split=5, class_weight='balanced', random_state=42
    )
    model_baseline = CalibratedClassifierCV(model_baseline, method='isotonic')
    model_baseline.fit(X_train_full, y_train)
    
    y_pred_baseline = model_baseline.predict(X_test_full)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    results['baseline_all_features'] = acc_baseline
    
    print(f"Accuracy BASELINE (toutes features): {acc_baseline:.1%} ({np.sum(y_pred_baseline == y_test)}/{len(y_test)})")
    
    # TEST 2: SANS features suspectes
    for group_name, features_to_exclude in suspect_features.items():
        if group_name == 'traditional_only':
            # Cas spécial: utiliser SEULEMENT ces features
            features_to_use = features_to_exclude
            test_name = f"SEULEMENT {group_name.upper()}"
        else:
            # Exclure ces features
            features_to_use = [f for f in all_features if f not in features_to_exclude]
            test_name = f"SANS {group_name.upper()}"
        
        print(f"\\n🔍 TEST {test_name}:")
        print(f"Features utilisées: {features_to_use}")
        
        X_train_subset = train_df[features_to_use]
        X_test_subset = test_df[features_to_use]
        
        # Entraîner modèle
        model_subset = RandomForestClassifier(
            n_estimators=300, max_depth=20, max_features='sqrt',
            min_samples_split=5, class_weight='balanced', random_state=42
        )
        model_subset = CalibratedClassifierCV(model_subset, method='isotonic')
        model_subset.fit(X_train_subset, y_train)
        
        # Test
        y_pred_subset = model_subset.predict(X_test_subset)
        acc_subset = accuracy_score(y_test, y_pred_subset)
        results[group_name] = acc_subset
        
        # Calcul impact
        impact = acc_baseline - acc_subset
        impact_pp = impact * 100
        
        print(f"Accuracy: {acc_subset:.1%} ({np.sum(y_pred_subset == y_test)}/{len(y_test)})")
        print(f"Impact vs baseline: {impact_pp:+.1f}pp")
        
        # DÉTECTION LEAKAGE
        if abs(impact_pp) > 15:  # Plus de 15pp de différence = suspect
            print(f"🚨 LEAKAGE SUSPECT DÉTECTÉ! Impact: {impact_pp:+.1f}pp")
            
    # TEST 3: Features individuelles
    print("\\n" + "=" * 70)
    print("🔍 TEST FEATURES INDIVIDUELLES:")
    print("-" * 70)
    
    individual_results = {}
    
    for feature in all_features:
        # Test avec SEULEMENT cette feature
        X_train_single = train_df[[feature]]
        X_test_single = test_df[[feature]]
        
        model_single = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        model_single.fit(X_train_single, y_train)
        
        y_pred_single = model_single.predict(X_test_single)
        acc_single = accuracy_score(y_test, y_pred_single)
        individual_results[feature] = acc_single
        
        print(f"{feature:25}: {acc_single:.1%}")
        
        # DÉTECTION LEAKAGE INDIVIDUEL
        if acc_single > 0.6:  # Plus de 60% avec UNE feature = très suspect
            print(f"  🚨 FEATURE SUSPECTE! {acc_single:.1%} avec une seule feature!")
    
    # RÉSUMÉ FINAL
    print("\\n" + "=" * 70)
    print("📊 RÉSUMÉ AUDIT LEAKAGE:")
    print("-" * 70)
    
    print(f"\\n🎯 PERFORMANCES PAR GROUPE:")
    for test_name, accuracy in results.items():
        impact = (results['baseline_all_features'] - accuracy) * 100
        status = "🚨 SUSPECT" if abs(impact) > 15 else "✅ OK"
        print(f"{test_name:25}: {accuracy:.1%} ({impact:+.1f}pp) {status}")
        
    print(f"\\n🔍 TOP FEATURES INDIVIDUELLES:")
    sorted_features = sorted(individual_results.items(), key=lambda x: x[1], reverse=True)
    for feature, acc in sorted_features[:5]:
        status = "🚨 SUSPECT" if acc > 0.6 else "✅ OK"
        print(f"{feature:25}: {acc:.1%} {status}")
    
    # VERDICT FINAL
    print("\\n" + "=" * 70)
    print("⚖️ VERDICT AUDIT LEAKAGE:")
    
    baseline_acc = results['baseline_all_features']
    traditional_acc = results['traditional_only']
    impact_advanced = (baseline_acc - traditional_acc) * 100
    
    print(f"\\nBaseline (toutes features): {baseline_acc:.1%}")
    print(f"Traditional only: {traditional_acc:.1%}")
    print(f"Impact features avancées: {impact_advanced:+.1f}pp")
    
    if abs(impact_advanced) > 20:
        print("\\n🚨 LEAKAGE CONFIRMÉ!")
        print("Les features avancées contiennent très probablement de l'information du futur.")
        print("La performance 71.8% est INVALIDE.")
    elif abs(impact_advanced) > 10:
        print("\\n⚠️ LEAKAGE POSSIBLE")
        print("Impact significatif détecté - investigation approfondie nécessaire.")
    else:
        print("\\n✅ PAS DE LEAKAGE MAJEUR DÉTECTÉ")
        print("Performance élevée semble légitime.")
        
    return results, individual_results

if __name__ == "__main__":
    results, individual_results = feature_ablation_test()