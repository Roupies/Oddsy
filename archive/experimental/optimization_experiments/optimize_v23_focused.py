#!/usr/bin/env python3
"""
Optimisation FOCALISÉE des hyperparamètres v2.3
===============================================

Optimisation ciblée sur les paramètres les plus prometteurs
basée sur les résultats de l'optimisation ultra-rapide.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, balanced_accuracy_score
import json
import os

def log(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - FOCUSED_OPT - {msg}")

def main():
    log("🚀 OPTIMISATION FOCALISÉE v2.3")
    log("Basée sur les meilleurs paramètres détectés")
    
    # Charger données
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv')
    v23_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    X = df[v23_features]
    y = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Grille focalisée autour des meilleurs paramètres
    param_grid = {
        'n_estimators': [150, 200, 250, 300],  # Autour de 200
        'max_depth': [20, 25, None],  # Autour de None
        'max_features': [0.6, 0.7, 0.8, 0.9],  # Autour de 0.8
        'min_samples_split': [2, 3, 4],  # Paramètre critique
        'class_weight': [None]  # Déjà optimal
    }
    
    log(f"Grille focalisée: {4*3*4*3*1} = 144 combinaisons")
    
    # GridSearch avec validation temporelle appropriée
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    log("Lancement GridSearch focalisé...")
    grid_search.fit(X, y)
    
    log(f"✅ Optimisation terminée!")
    log(f"Meilleur score CV: {grid_search.best_score_:.4f}")
    log(f"Meilleurs params: {grid_search.best_params_}")
    
    # Comparaison rigoureuse avec le modèle original
    log("🔍 Comparaison rigoureuse avec modèle original...")
    
    original_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
    
    # Évaluation en cross-validation temporelle (comme dans l'audit)
    tscv_eval = TimeSeriesSplit(n_splits=5)
    original_scores = []
    optimized_scores = []
    
    rf_opt = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    
    for train_idx, val_idx in tscv_eval.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Original
        original_model.fit(X_train, y_train)
        original_scores.append(accuracy_score(y_val, original_model.predict(X_val)))
        
        # Optimized
        rf_opt.fit(X_train, y_train)
        optimized_scores.append(accuracy_score(y_val, rf_opt.predict(X_val)))
    
    orig_mean = np.mean(original_scores)
    orig_std = np.std(original_scores)
    opt_mean = np.mean(optimized_scores)
    opt_std = np.std(optimized_scores)
    improvement = (opt_mean - orig_mean) * 100
    
    log("📊 RÉSULTATS FOCALISÉS (Cross-Validation Temporelle):")
    log(f"Original:  {orig_mean:.4f} ± {orig_std:.4f}")
    log(f"Optimisé:  {opt_mean:.4f} ± {opt_std:.4f}")
    log(f"Amélioration: {improvement:+.2f}pp")
    
    # Évaluation complète
    rf_final = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    model_final = CalibratedClassifierCV(rf_final, method='isotonic', cv=3)
    model_final.fit(X, y)
    
    # Sauvegarder résultats détaillés
    os.makedirs('results', exist_ok=True)
    results = {
        'optimization_metadata': {
            'timestamp': datetime.now().isoformat(),
            'method': 'focused_optimization',
            'combinations_tested': len(grid_search.cv_results_['params'])
        },
        'best_hyperparameters': grid_search.best_params_,
        'grid_search_results': {
            'best_cv_score': grid_search.best_score_,
            'std_cv_score': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        },
        'comparison_results': {
            'original_cv_mean': orig_mean,
            'original_cv_std': orig_std,
            'original_cv_scores': original_scores,
            'optimized_cv_mean': opt_mean,
            'optimized_cv_std': opt_std,
            'optimized_cv_scores': optimized_scores,
            'improvement_pp': improvement
        },
        'statistical_significance': {
            'improvement_larger_than_std': abs(improvement) > orig_std * 100,
            'cv_stability': opt_std < 0.05
        }
    }
    
    with open('results/v23_focused_optimization.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    log("💾 Résultats détaillés sauvés dans results/v23_focused_optimization.json")
    
    if improvement > 0.5:
        log("🎉 AMÉLIORATION SIGNIFICATIVE DÉTECTÉE!")
        model_path = f'models/v23_focused_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(model_final, model_path)
        log(f"💾 Modèle optimisé sauvegardé: {model_path}")
    elif improvement > -0.5:
        log("📊 PERFORMANCE ÉQUIVALENTE - Paramètres par défaut restent optimaux")
    else:
        log("📉 DÉGRADATION DÉTECTÉE - Modèle original reste supérieur")
    
    return results

if __name__ == "__main__":
    main()