#!/usr/bin/env python3
"""
Analyse des résultats d'optimisation v2.3
=========================================

Analyse des résultats d'optimisation et test manuel des 
meilleurs paramètres identifiés.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, classification_report
import json
import os

def log(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ANALYSIS - {msg}")

def test_specific_params(X, y, params, name):
    """Test des paramètres spécifiques avec validation temporelle."""
    log(f"Test de configuration: {name}")
    log(f"Paramètres: {params}")
    
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    
    # Cross-validation temporelle
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    balanced_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        cv_scores.append(accuracy_score(y_val, y_pred))
        balanced_scores.append(balanced_accuracy_score(y_val, y_pred))
    
    return {
        'accuracy_mean': np.mean(cv_scores),
        'accuracy_std': np.std(cv_scores),
        'balanced_accuracy_mean': np.mean(balanced_scores),
        'balanced_accuracy_std': np.std(balanced_scores),
        'cv_scores': cv_scores
    }

def main():
    log("🔍 ANALYSE D'OPTIMISATION v2.3")
    
    # Charger données
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv')
    v23_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    X = df[v23_features]
    y = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    log(f"Dataset: {X.shape[0]} matches, {X.shape[1]} features")
    
    # Configuration originale (du modèle v2.3 retrained)
    original_params = {
        'n_estimators': 100,
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': None
    }
    
    # Meilleurs paramètres trouvés par optimisation ultra-rapide
    best_ultra_quick = {
        'n_estimators': 200,
        'max_depth': None,
        'max_features': 0.8,
        'class_weight': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    # Configuration conservative (légères améliorations)
    conservative_params = {
        'n_estimators': 150,
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'class_weight': None
    }
    
    # Configuration aggressive
    aggressive_params = {
        'n_estimators': 300,
        'max_depth': 25,
        'max_features': 0.8,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    }
    
    # Tests des différentes configurations
    configurations = [
        (original_params, "Original v2.3"),
        (best_ultra_quick, "Ultra-Quick Best"),
        (conservative_params, "Conservative"),
        (aggressive_params, "Aggressive")
    ]
    
    results = {}
    
    for params, name in configurations:
        results[name] = test_specific_params(X, y, params, name)
    
    # Analyse comparative
    log("\n" + "="*60)
    log("📊 RÉSULTATS COMPARATIFS")
    log("="*60)
    
    baseline_accuracy = results["Original v2.3"]["accuracy_mean"]
    
    for name, result in results.items():
        accuracy = result["accuracy_mean"]
        std = result["accuracy_std"]
        improvement = (accuracy - baseline_accuracy) * 100
        
        log(f"{name:20} | {accuracy:.4f} ± {std:.4f} | {improvement:+.2f}pp")
    
    # Identification du meilleur
    best_config = max(results.items(), key=lambda x: x[1]["accuracy_mean"])
    best_name, best_result = best_config
    
    log(f"\n🏆 MEILLEURE CONFIGURATION: {best_name}")
    log(f"Performance: {best_result['accuracy_mean']:.4f} ± {best_result['accuracy_std']:.4f}")
    
    # Sauvegarde des résultats
    os.makedirs('results', exist_ok=True)
    
    # Trouver les paramètres de la meilleure configuration
    best_params = None
    for params, name in configurations:
        if name == best_name:
            best_params = params
            break
    
    analysis_report = {
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'method': 'manual_configuration_testing',
            'baseline_model': 'v23_retrained'
        },
        'configurations_tested': {
            name: {
                'params': params,
                'results': result
            } for (params, name), result in zip(configurations, results.values())
        },
        'best_configuration': {
            'name': best_name,
            'params': best_params,
            'performance': best_result,
            'improvement_vs_original': (best_result['accuracy_mean'] - baseline_accuracy) * 100
        },
        'summary': {
            'original_performance': baseline_accuracy,
            'best_performance': best_result['accuracy_mean'],
            'total_improvement': (best_result['accuracy_mean'] - baseline_accuracy) * 100,
            'is_improvement_significant': abs((best_result['accuracy_mean'] - baseline_accuracy) * 100) > best_result['accuracy_std'] * 100
        }
    }
    
    with open('results/v23_optimization_analysis.json', 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    log("💾 Analyse complète sauvée dans results/v23_optimization_analysis.json")
    
    # Recommandation finale
    improvement = (best_result['accuracy_mean'] - baseline_accuracy) * 100
    
    if improvement > 0.5:
        log(f"✅ RECOMMANDATION: Utiliser configuration {best_name} (+{improvement:.2f}pp)")
    elif improvement > -0.5:
        log("📊 RECOMMANDATION: Paramètres originaux restent optimaux (performance équivalente)")
    else:
        log("❌ RECOMMANDATION: Conserver paramètres originaux (dégradation détectée)")
    
    return analysis_report

if __name__ == "__main__":
    main()