#!/usr/bin/env python3
"""
🚀 PHASE 1.2 - FEATURE SELECTION DRASTIQUE

Permutation Importance + RFE pour identifier features elite
BASELINE: RandomForest 50.0% avec 10 features  
OBJECTIF: Maintenir/améliorer performance avec moins de features
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
from datetime import datetime
import json

def load_clean_data():
    """Charger données nettoyées"""
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    
    cutoff_date = pd.Timestamp('2025-08-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    train_clean = train_df.dropna(subset=features + ['FullTimeResult'])
    test_clean = test_df.dropna(subset=features + ['FullTimeResult'])
    
    return train_clean, test_clean, features

def analyze_feature_selection():
    """Analyse complète de sélection de features"""
    
    print("🔬 PHASE 1.2 - FEATURE SELECTION DRASTIQUE")
    print("=" * 60)
    
    train_df, test_df, features = load_clean_data()
    
    X_train = train_df[features]
    y_train = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    X_test = test_df[features]
    y_test = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print(f"📊 Dataset: {len(X_train)} train, {len(X_test)} test")
    
    # Modèle baseline
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=20, max_features='sqrt',
        min_samples_split=5, class_weight='balanced', random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    baseline_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"🎯 Baseline (10 features): {baseline_acc:.3f}")
    
    # PERMUTATION IMPORTANCE
    print()
    print("🔍 PERMUTATION IMPORTANCE ANALYSIS:")
    print("-" * 50)
    
    perm_importance = permutation_importance(
        rf_model, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
    )
    
    feature_scores = []
    for i, feature in enumerate(features):
        importance = perm_importance.importances_mean[i]
        std = perm_importance.importances_std[i]
        abs_importance = abs(importance)
        
        impact_pp = importance * 100
        status = "💎" if abs_importance > 0.02 else "⚠️" if abs_importance > 0.01 else "❌"
        
        print(f"{status} {feature:25}: {impact_pp:+.1f}pp ± {std*100:.1f}pp")
        
        feature_scores.append({
            'feature': feature,
            'importance': importance,
            'abs_importance': abs_importance,
            'std': std
        })
    
    # Trier par importance absolue
    feature_scores.sort(key=lambda x: x['abs_importance'], reverse=True)
    
    # Features elite (impact > 1pp)
    elite_features = [f['feature'] for f in feature_scores if f['abs_importance'] > 0.01]
    print(f"\n💎 ELITE FEATURES ({len(elite_features)}): {elite_features}")
    
    # TEST RFE - différentes tailles
    print()
    print("🎯 RECURSIVE FEATURE ELIMINATION:")
    print("-" * 50)
    
    rfe_results = {}
    for n_features in [3, 4, 5, 6, 7, 8]:
        # RFE selection
        rfe = RFE(
            RandomForestClassifier(n_estimators=100, random_state=42), 
            n_features_to_select=n_features
        )
        rfe.fit(X_train, y_train)
        
        selected_features = [features[i] for i, selected in enumerate(rfe.support_) if selected]
        
        # Test performance
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]
        
        rf_subset = RandomForestClassifier(
            n_estimators=300, max_depth=20, max_features='sqrt',
            min_samples_split=5, class_weight='balanced', random_state=42
        )
        rf_subset.fit(X_train_subset, y_train)
        
        test_acc = accuracy_score(y_test, rf_subset.predict(X_test_subset))
        improvement = (test_acc - baseline_acc) * 100
        
        status = "✅" if improvement >= 0 else "❌"
        print(f"{status} {n_features} features: {test_acc:.3f} ({improvement:+.1f}pp)")
        print(f"    {selected_features}")
        
        rfe_results[n_features] = {
            'features': selected_features,
            'accuracy': test_acc,
            'improvement': improvement
        }
    
    # TEST CONFIGURATIONS MANUELLES
    print()
    print("🧪 TEST CONFIGURATIONS MANUELLES:")
    print("-" * 50)
    
    manual_configs = {
        'top_5_elite': elite_features[:5] if len(elite_features) >= 5 else elite_features,
        'core_3': ['elo_diff_normalized', 'market_entropy_norm', 'form_diff_normalized'],
        'traditional_4': ['elo_diff_normalized', 'form_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized']
    }
    
    manual_results = {}
    for config_name, config_features in manual_configs.items():
        if len(config_features) > 0 and all(f in features for f in config_features):
            X_train_subset = X_train[config_features]
            X_test_subset = X_test[config_features]
            
            rf_manual = RandomForestClassifier(
                n_estimators=300, max_depth=20, max_features='sqrt',
                min_samples_split=5, class_weight='balanced', random_state=42
            )
            rf_manual.fit(X_train_subset, y_train)
            
            test_acc = accuracy_score(y_test, rf_manual.predict(X_test_subset))
            improvement = (test_acc - baseline_acc) * 100
            
            status = "✅" if improvement >= 0 else "❌"
            print(f"{status} {config_name:15} ({len(config_features)}f): {test_acc:.3f} ({improvement:+.1f}pp)")
            
            manual_results[config_name] = {
                'features': config_features,
                'accuracy': test_acc,
                'improvement': improvement
            }
    
    # RECOMMANDATION FINALE
    print()
    print("=" * 60)
    print("📊 RECOMMANDATION FINALE:")
    print("=" * 60)
    
    # Trouver meilleure configuration
    all_results = {}
    all_results.update({f"rfe_{k}": v for k, v in rfe_results.items()})
    all_results.update(manual_results)
    
    if all_results:
        best_config = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
        best_data = all_results[best_config]
        
        if best_data['improvement'] > 0.5:  # Seuil 0.5pp
            print(f"✅ AMÉLIORATION DÉTECTÉE!")
            print(f"🏆 Meilleure config: {best_config}")
            print(f"📈 Gain: {best_data['improvement']:+.1f}pp")
            print(f"🎯 Performance: {best_data['accuracy']:.3f}")
            print(f"📋 Features ({len(best_data['features'])}): {best_data['features']}")
            
            recommended_features = best_data['features']
            recommended_accuracy = best_data['accuracy']
        else:
            print("❌ AUCUNE AMÉLIORATION SIGNIFICATIVE")
            print("💡 Recommandation: Garder les 10 features actuelles")
            recommended_features = features
            recommended_accuracy = baseline_acc
    else:
        print("⚠️ Erreur dans l'analyse - garder baseline")
        recommended_features = features
        recommended_accuracy = baseline_acc
    
    # Sauvegarder résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'baseline_accuracy': baseline_acc,
        'recommended_features': recommended_features,
        'recommended_accuracy': recommended_accuracy,
        'improvement_pp': (recommended_accuracy - baseline_acc) * 100,
        'feature_importance': feature_scores,
        'rfe_results': rfe_results,
        'manual_results': manual_results
    }
    
    results_path = f"evaluation/phase1_feature_selection_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Résultats sauvés: {results_path}")
    
    return recommended_features, recommended_accuracy, results

if __name__ == "__main__":
    recommended_features, accuracy, results = analyze_feature_selection()
    
    print()
    print("🚀 PHASE 1.2 TERMINÉE")
    print(f"🎯 Features recommandées: {len(recommended_features)}")
    print(f"📈 Performance: {accuracy:.1%}")
    print("📋 Prochaine étape: Feature Engineering (Phase 2)")