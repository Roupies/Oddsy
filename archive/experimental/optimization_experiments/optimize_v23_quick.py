#!/usr/bin/env python3
"""
Optimisation RAPIDE des hyperparamètres v2.3
=============================================

Version allégée avec grille de recherche réduite pour tests rapides.
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
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - QUICK_OPT - {msg}")

def main():
    log("🚀 OPTIMISATION RAPIDE v2.3")
    
    # Charger données
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv')
    v23_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    X = df[v23_features]
    y = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Grille réduite (plus rapide)
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [15, 20, None],
        'max_features': ['sqrt', 0.8],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    }
    
    log(f"Grille réduite: {3*3*2*3*2} = 108 combinaisons")
    
    # GridSearch rapide
    tscv = TimeSeriesSplit(n_splits=3)  # Réduit à 3 folds
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=2
    )
    
    log("Lancement GridSearch rapide...")
    grid_search.fit(X, y)
    
    log(f"✅ Terminé! Meilleur score: {grid_search.best_score_:.4f}")
    log(f"Meilleurs params: {grid_search.best_params_}")
    
    # Comparer avec original
    original_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
    
    # Test simple
    tscv_test = TimeSeriesSplit(n_splits=5)
    original_scores = []
    optimized_scores = []
    
    rf_opt = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    
    for train_idx, val_idx in tscv_test.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Original
        original_model.fit(X_train, y_train)
        original_scores.append(accuracy_score(y_val, original_model.predict(X_val)))
        
        # Optimized
        rf_opt.fit(X_train, y_train)
        optimized_scores.append(accuracy_score(y_val, rf_opt.predict(X_val)))
    
    orig_mean = np.mean(original_scores)
    opt_mean = np.mean(optimized_scores)
    improvement = (opt_mean - orig_mean) * 100
    
    log("📊 RÉSULTATS RAPIDES:")
    log(f"Original:  {orig_mean:.4f} ± {np.std(original_scores):.4f}")
    log(f"Optimisé:  {opt_mean:.4f} ± {np.std(optimized_scores):.4f}")
    log(f"Amélioration: {improvement:+.2f}pp")
    
    # Sauvegarder résultats
    os.makedirs('results', exist_ok=True)
    results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'quick_optimization',
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'original_cv': orig_mean,
        'optimized_cv': opt_mean,
        'improvement_pp': improvement,
        'combinations_tested': len(grid_search.cv_results_['params'])
    }
    
    with open('results/v23_quick_optimization.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    log("💾 Résultats sauvés dans results/v23_quick_optimization.json")
    
    if improvement > 0.5:
        log("🎉 AMÉLIORATION SIGNIFICATIVE!")
        # Entraîner et sauver le modèle optimisé
        model_opt = CalibratedClassifierCV(rf_opt, method='isotonic', cv=3)
        model_opt.fit(X, y)
        joblib.dump(model_opt, f'models/v23_quick_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
        log("💾 Modèle optimisé sauvegardé")
    
    return results

if __name__ == "__main__":
    main()