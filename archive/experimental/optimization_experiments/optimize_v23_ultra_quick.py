#!/usr/bin/env python3
"""
Optimisation ULTRA RAPIDE des hyperparamètres v2.3
===================================================

Version minimale avec seulement les paramètres essentiels.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import json
import os

def log(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ULTRA_QUICK - {msg}")

def main():
    log("🚀 OPTIMISATION ULTRA RAPIDE v2.3")
    
    # Charger données
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv')
    v23_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    X = df[v23_features]
    y = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Grille ultra réduite (paramètres clés uniquement)
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [15, None],
        'max_features': ['sqrt', 0.8],
        'class_weight': [None, 'balanced']
    }
    
    log(f"Grille ultra réduite: {2*2*2*2} = 16 combinaisons")
    
    # GridSearch ultra rapide
    tscv = TimeSeriesSplit(n_splits=3)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=tscv,
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    log("Lancement GridSearch ultra rapide...")
    grid_search.fit(X, y)
    
    log(f"✅ Terminé! Meilleur score: {grid_search.best_score_:.4f}")
    log(f"Meilleurs params: {grid_search.best_params_}")
    
    # Évaluation simple
    original_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
    
    # Test rapide avec un fold
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Original
    original_model.fit(X_train, y_train)
    orig_score = accuracy_score(y_test, original_model.predict(X_test))
    
    # Optimisé
    rf_opt = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    rf_opt.fit(X_train, y_train)
    opt_score = accuracy_score(y_test, rf_opt.predict(X_test))
    
    improvement = (opt_score - orig_score) * 100
    
    log("📊 RÉSULTATS ULTRA RAPIDES:")
    log(f"Original:  {orig_score:.4f}")
    log(f"Optimisé:  {opt_score:.4f}")
    log(f"Amélioration: {improvement:+.2f}pp")
    
    # Sauvegarder résultats
    os.makedirs('results', exist_ok=True)
    results = {
        'timestamp': datetime.now().isoformat(),
        'method': 'ultra_quick_optimization',
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'original_test': orig_score,
        'optimized_test': opt_score,
        'improvement_pp': improvement,
        'combinations_tested': len(grid_search.cv_results_['params'])
    }
    
    with open('results/v23_ultra_quick_optimization.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    log("💾 Résultats sauvés dans results/v23_ultra_quick_optimization.json")
    
    if improvement > 0.5:
        log("🎉 AMÉLIORATION DÉTECTÉE!")
        # Entraîner et sauver le modèle optimisé
        model_opt = CalibratedClassifierCV(rf_opt, method='isotonic', cv=3)
        model_opt.fit(X, y)
        model_path = f'models/v23_ultra_quick_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(model_opt, model_path)
        log(f"💾 Modèle optimisé sauvegardé: {model_path}")
    else:
        log("📊 Pas d'amélioration significative détectée")
    
    return results

if __name__ == "__main__":
    main()