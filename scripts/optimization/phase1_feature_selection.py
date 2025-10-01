#!/usr/bin/env python3
"""
🚀 PHASE 1.2 - FEATURE SELECTION DRASTIQUE

Permutation Importance + RFE pour identifier features "elite" 
et éliminer celles qui apportent du bruit.

BASELINE: RandomForest 50.0% avec 10 features
OBJECTIF: Maintenir/améliorer performance avec moins de features
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import json

def load_clean_data():
    """
    Charger données nettoyées (même méthode que Phase 1.1)
    """
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    
    # Split temporel strict
    cutoff_date = pd.Timestamp('2025-08-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Nettoyer données
    train_clean = train_df.dropna(subset=features + ['FullTimeResult'])
    test_clean = test_df.dropna(subset=features + ['FullTimeResult'])
    
    return train_clean, test_clean, features

def permutation_importance_analysis():
    """
    Analyser importance réelle des features par permutation
    """
    print("🔬 PHASE 1.2 - FEATURE SELECTION DRASTIQUE")
    print("=" * 60)
    
    train_df, test_df, features = load_clean_data()
    
    X_train = train_df[features]
    y_train = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    X_test = test_df[features]
    y_test = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print(f"📊 Analyse sur {len(features)} features, {len(X_train)} samples train")
    
    # Modèle RandomForest baseline (gagnant Phase 1.1)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        max_features='sqrt',
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Performance baseline
    baseline_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"🎯 Baseline accuracy: {baseline_acc:.3f} (10 features)")
    
    print()
    print("🔍 PERMUTATION IMPORTANCE ANALYSIS:")
    print("-" * 60)
    
    # Calcul permutation importance
    perm_importance = permutation_importance(
        rf_model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring='accuracy'
    )
    
    # Créer dataframe des résultats
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'abs_importance': np.abs(perm_importance.importances_mean)
    }).sort_values('abs_importance', ascending=False)
    
    print("Feature Importance (Permutation):")
    for _, row in feature_importance.iterrows():
        impact_pp = row['importance_mean'] * 100
        status = "💎" if row['abs_importance'] > 0.02 else "⚠️" if row['abs_importance'] > 0.01 else "❌"
        print(f"{status} {row['feature']:25}: {impact_pp:+.1f}pp ± {row['importance_std']*100:.1f}pp")
    
    # Identifier features elite (seuil 1pp d'impact)
    elite_features = feature_importance[feature_importance['abs_importance'] > 0.01]['feature'].tolist()
    noise_features = feature_importance[feature_importance['abs_importance'] <= 0.01]['feature'].tolist()
    
    print(f"\n💎 FEATURES ELITE ({len(elite_features)}): {elite_features}")
    print(f"❌ FEATURES BRUIT ({len(noise_features)}): {noise_features}")
    
    return feature_importance, elite_features, noise_features

def recursive_feature_elimination():
    """
    RFE pour trouver nombre optimal de features
    """
    print()
    print("🎯 RECURSIVE FEATURE ELIMINATION:")
    print("-" * 60)
    
    train_df, test_df, features = load_clean_data()
    
    X_train = train_df[features]
    y_train = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    X_test = test_df[features]
    y_test = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    rf_model = RandomForestClassifier(
        n_estimators=200,  # Réduit pour vitesse RFE
        max_depth=20,
        max_features='sqrt',
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    # Test différentes tailles de features
    rfe_results = {}
    feature_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    
    for n_features in feature_sizes:
        print(f\"\\n🧪 Test avec {n_features} features:\")
        
        # RFE pour sélectionner top features
        rfe = RFE(rf_model, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        
        selected_features = [features[i] for i, selected in enumerate(rfe.support_) if selected]
        
        # Test performance avec subset
        X_train_subset = X_train[selected_features]
        X_test_subset = X_test[selected_features]
        
        rf_subset = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            max_features='sqrt',
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(rf_subset, X_train_subset, y_train, cv=tscv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Test final
        rf_subset.fit(X_train_subset, y_train)
        test_acc = accuracy_score(y_test, rf_subset.predict(X_test_subset))
        
        rfe_results[n_features] = {
            'selected_features': selected_features,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_acc
        }
        
        print(f\"   Features: {selected_features}\")
        print(f\"   CV: {cv_mean:.3f} ± {cv_std:.3f}\")
        print(f\"   Test Acc: {test_acc:.3f}\")
    
    return rfe_results

def test_minimal_models():
    \"\"\"
    Tester modèles ultra-minimalistes (3-4 features)
    \"\"\"
    print()
    print(\"🎪 TEST MODÈLES MINIMALISTES:\")
    print(\"-\" * 60)
    
    train_df, test_df, features = load_clean_data()
    
    X_train_full = train_df[features]
    y_train = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    X_test_full = test_df[features]
    y_test = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Configurations minimalistes à tester
    minimal_configs = {
        'core_3': ['elo_diff_normalized', 'market_entropy_norm', 'form_diff_normalized'],
        'core_4': ['elo_diff_normalized', 'market_entropy_norm', 'form_diff_normalized', 'home_xg_eff_10'],
        'traditional': ['elo_diff_normalized', 'form_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized'],
        'market_focused': ['elo_diff_normalized', 'market_entropy_norm', 'h2h_score']
    }
    
    minimal_results = {}
    
    for config_name, feature_subset in minimal_configs.items():
        print(f\"\\n🧪 {config_name.upper()} ({len(feature_subset)} features):\")
        print(f\"   Features: {feature_subset}\")
        
        X_train_subset = train_df[feature_subset]
        X_test_subset = test_df[feature_subset]
        
        rf_minimal = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            max_features='sqrt',
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(rf_minimal, X_train_subset, y_train, cv=tscv, scoring='accuracy')
        
        # Test final
        rf_minimal.fit(X_train_subset, y_train)
        y_pred = rf_minimal.predict(X_test_subset)
        test_acc = accuracy_score(y_test, y_pred)
        
        # Draw performance
        class_report = classification_report(y_test, y_pred, target_names=['H', 'D', 'A'], output_dict=True)
        draw_recall = class_report['D']['recall']
        
        minimal_results[config_name] = {
            'features': feature_subset,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(), 
            'test_accuracy': test_acc,
            'draw_recall': draw_recall
        }
        
        print(f\"   CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\")
        print(f\"   Test: {test_acc:.3f}\")
        print(f\"   Draw Recall: {draw_recall:.3f}\")
    
    return minimal_results

def analyze_results_and_recommend():
    \"\"\"
    Analyser tous les résultats et faire recommandation finale
    \"\"\"
    print()
    print(\"=\"*60)
    print(\"📊 ANALYSE FINALE - PHASE 1.2\")
    print(\"=\"*60)
    
    # Lancer toutes les analyses
    feature_importance, elite_features, noise_features = permutation_importance_analysis()
    rfe_results = recursive_feature_elimination()
    minimal_results = test_minimal_models()
    
    # Analyser RFE results
    print()
    print(\"🎯 RFE ANALYSIS - OPTIMAL FEATURE COUNT:\")
    baseline_10f = 0.500  # Baseline 10 features
    
    for n_features, data in rfe_results.items():
        improvement = (data['test_accuracy'] - baseline_10f) * 100
        status = \"✅\" if improvement >= 0 else \"❌\"
        print(f\"{status} {n_features} features: {data['test_accuracy']:.3f} ({improvement:+.1f}pp vs 10f)\")
    
    # Trouver meilleur nombre de features
    best_n_features = max(rfe_results.keys(), key=lambda k: rfe_results[k]['test_accuracy'])
    best_rfe = rfe_results[best_n_features]
    
    print()
    print(\"🏆 RECOMMANDATION FINALE:\")
    
    # Comparer approches
    approaches = {
        'baseline_10f': {'accuracy': baseline_10f, 'features': 10, 'description': 'Baseline 10 features'},
        'best_rfe': {'accuracy': best_rfe['test_accuracy'], 'features': best_n_features, 'description': f'RFE optimal ({best_n_features} features)'},
        'permutation_elite': {'accuracy': None, 'features': len(elite_features), 'description': f'Elite features ({len(elite_features)} features)'}
    }
    
    # Test elite features si différent du meilleur RFE
    if len(elite_features) != best_n_features:
        print(f\"\\n🧪 Test final - Elite features permutation ({len(elite_features)}):\")
        train_df, test_df, _ = load_clean_data()
        
        if len(elite_features) > 0:
            X_train_elite = train_df[elite_features]
            X_test_elite = test_df[elite_features]
            y_train = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
            y_test = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
            
            rf_elite = RandomForestClassifier(
                n_estimators=300, max_depth=20, max_features='sqrt',
                min_samples_split=5, class_weight='balanced', random_state=42
            )
            rf_elite.fit(X_train_elite, y_train)
            elite_acc = accuracy_score(y_test, rf_elite.predict(X_test_elite))
            approaches['permutation_elite']['accuracy'] = elite_acc
        
        print(f\"   Elite features accuracy: {elite_acc:.3f}\")
    
    # Recommandation finale
    print()
    print(\"💡 RECOMMANDATION STRATÉGIQUE:\")
    
    if best_rfe['test_accuracy'] > baseline_10f + 0.005:  # +0.5pp seuil
        print(f\"✅ RÉDUCTION RECOMMANDÉE: {best_n_features} features\")
        print(f\"📈 Gain: {(best_rfe['test_accuracy'] - baseline_10f)*100:+.1f}pp\")
        print(f\"🎯 Features sélectionnées: {best_rfe['selected_features']}\")
        recommended_features = best_rfe['selected_features']
        recommended_accuracy = best_rfe['test_accuracy']
    else:
        print(\"❌ RÉDUCTION NON BÉNÉFIQUE\")
        print(\"💡 Recommandation: Garder les 10 features actuelles\")
        recommended_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
            'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
            'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        recommended_accuracy = baseline_10f
    
    # Sauvegarder résultats
    timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")
    
    results_summary = {
        'timestamp': timestamp,
        'baseline_accuracy': baseline_10f,
        'recommended_features': recommended_features,
        'recommended_accuracy': recommended_accuracy,
        'improvement_pp': (recommended_accuracy - baseline_10f) * 100,
        'permutation_analysis': feature_importance.to_dict('records'),
        'rfe_results': rfe_results,
        'minimal_results': minimal_results
    }
    
    results_path = f\"evaluation/phase1_feature_selection_{timestamp}.json\"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f\"\\n📋 Résultats sauvés: {results_path}\")
    
    return recommended_features, recommended_accuracy, results_summary

if __name__ == \"__main__\":
    recommended_features, accuracy, results = analyze_results_and_recommend()
    
    print()
    print(\"🚀 PHASE 1.2 TERMINÉE\")
    print(f\"🎯 Features optimales: {len(recommended_features)} features\") 
    print(f\"📈 Performance: {accuracy:.1%}\")
    print(\"📋 Prochaine étape: Feature Engineering (Phase 2)\")