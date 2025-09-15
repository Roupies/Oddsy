#!/usr/bin/env python3
"""
Optimisation des hyperparamètres pour le modèle v2.3 Retrained
================================================================

Ce script optimise les hyperparamètres du modèle v2.3 champion avec :
1. GridSearchCV avec validation temporelle
2. Recherche exhaustive sur les paramètres clés
3. Comparaison avec le modèle original
4. Audit complet du modèle optimisé

Author: Oddsy Project
Date: 2025-09-12
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
    """Logger avec timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - OPTIMIZATION - {msg}")

def load_v23_data():
    """Charger les données avec les mêmes features que v2.3."""
    log("Chargement des données v2.3...")
    
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv')
    
    # Features v2.3 exactes
    v23_features = [
        'form_diff_normalized',
        'elo_diff_normalized', 
        'h2h_score',
        'matchday_normalized',
        'shots_diff_normalized',
        'corners_diff_normalized',
        'market_entropy_norm',
        'home_xg_eff_10',
        'away_goals_sum_5',
        'away_xg_eff_10'
    ]
    
    X = df[v23_features]
    y = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    log(f"Dataset: {X.shape[0]} matches, {X.shape[1]} features")
    log(f"Distribution: {y.value_counts().to_dict()}")
    
    return X, y, v23_features

def optimize_hyperparameters(X, y):
    """Optimiser les hyperparamètres avec GridSearchCV."""
    log("🔧 Démarrage optimisation hyperparamètres...")
    
    # Grille de recherche étendue
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, None],
        'max_features': ['sqrt', 'log2', 0.5, 0.8],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    log(f"Grille de recherche: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['max_features']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['class_weight'])} combinaisons")
    
    # Validation temporelle (comme dans l'audit)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # GridSearch
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    log("Lancement GridSearchCV (cela peut prendre 10-20 minutes)...")
    grid_search.fit(X, y)
    
    log("✅ Optimisation terminée!")
    log(f"Meilleur score CV: {grid_search.best_score_:.4f}")
    log(f"Meilleurs paramètres: {grid_search.best_params_}")
    
    return grid_search

def train_optimized_model(X, y, best_params):
    """Entraîner le modèle optimisé avec calibration."""
    log("🚀 Entraînement du modèle optimisé...")
    
    # Modèle avec meilleurs hyperparamètres
    rf_optimized = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    # Calibration (comme le modèle original)
    model_optimized = CalibratedClassifierCV(rf_optimized, method='isotonic', cv=3)
    model_optimized.fit(X, y)
    
    log("✅ Modèle optimisé entraîné avec calibration")
    return model_optimized

def evaluate_comparison(X, y, original_model, optimized_model):
    """Comparer performance original vs optimisé."""
    log("📊 Comparaison des performances...")
    
    # Validation temporelle pour les deux modèles
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {
        'original': {'cv_scores': [], 'cv_mean': 0, 'cv_std': 0},
        'optimized': {'cv_scores': [], 'cv_mean': 0, 'cv_std': 0}
    }
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Original
        original_model.fit(X_train, y_train)
        y_pred_orig = original_model.predict(X_val)
        acc_orig = accuracy_score(y_val, y_pred_orig)
        results['original']['cv_scores'].append(acc_orig)
        
        # Optimized
        optimized_model.fit(X_train, y_train)
        y_pred_opt = optimized_model.predict(X_val)
        acc_opt = accuracy_score(y_val, y_pred_opt)
        results['optimized']['cv_scores'].append(acc_opt)
    
    # Calculs finaux
    results['original']['cv_mean'] = np.mean(results['original']['cv_scores'])
    results['original']['cv_std'] = np.std(results['original']['cv_scores'])
    results['optimized']['cv_mean'] = np.mean(results['optimized']['cv_scores'])
    results['optimized']['cv_std'] = np.std(results['optimized']['cv_scores'])
    
    log("📈 RÉSULTATS COMPARATIFS:")
    log(f"Original  - CV: {results['original']['cv_mean']:.4f} ± {results['original']['cv_std']:.4f}")
    log(f"Optimisé  - CV: {results['optimized']['cv_mean']:.4f} ± {results['optimized']['cv_std']:.4f}")
    
    improvement = (results['optimized']['cv_mean'] - results['original']['cv_mean']) * 100
    log(f"Amélioration: {improvement:+.2f} points de pourcentage")
    
    return results

def save_optimization_report(grid_search, results, best_params, features):
    """Sauvegarder le rapport d'optimisation."""
    log("💾 Sauvegarde du rapport d'optimisation...")
    
    report = {
        'optimization_metadata': {
            'timestamp': datetime.now().isoformat(),
            'base_model': 'v2.3_retrained',
            'optimization_method': 'GridSearchCV_TimeSeriesSplit',
            'total_combinations_tested': len(grid_search.cv_results_['params'])
        },
        'original_performance': {
            'cv_accuracy': results['original']['cv_mean'],
            'cv_std': results['original']['cv_std'],
            'cv_scores': results['original']['cv_scores']
        },
        'optimized_performance': {
            'cv_accuracy': results['optimized']['cv_mean'], 
            'cv_std': results['optimized']['cv_std'],
            'cv_scores': results['optimized']['cv_scores'],
            'improvement_pp': (results['optimized']['cv_mean'] - results['original']['cv_mean']) * 100
        },
        'best_hyperparameters': best_params,
        'features_used': features,
        'grid_search_results': {
            'best_score': grid_search.best_score_,
            'best_index': grid_search.best_index_,
            'n_combinations': len(grid_search.cv_results_['params'])
        }
    }
    
    # Sauvegarder JSON
    with open('results/v23_optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Rapport markdown
    with open('results/v23_optimization_summary.md', 'w') as f:
        f.write(f"""# OPTIMISATION v2.3 - RAPPORT COMPLET

## 🎯 RÉSULTATS PRINCIPAUX

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Performance Comparative
- **Original v2.3:** {results['original']['cv_mean']:.4f} ± {results['original']['cv_std']:.4f}
- **Optimisé v2.3:** {results['optimized']['cv_mean']:.4f} ± {results['optimized']['cv_std']:.4f}
- **Amélioration:** {(results['optimized']['cv_mean'] - results['original']['cv_mean']) * 100:+.2f} points de pourcentage

### Meilleurs Hyperparamètres
```json
{json.dumps(best_params, indent=2)}
```

### Recherche GridSearch
- **Combinaisons testées:** {len(grid_search.cv_results_['params'])}
- **Meilleur score CV:** {grid_search.best_score_:.4f}
- **Validation:** TimeSeriesSplit (5 folds)

## 📊 Scores CV Détaillés

**Original:** {results['original']['cv_scores']}
**Optimisé:** {results['optimized']['cv_scores']}

---
*Rapport généré automatiquement - Oddsy Optimization Pipeline*
""")
    
    log("✅ Rapports sauvegardés:")
    log("  - results/v23_optimization_report.json")
    log("  - results/v23_optimization_summary.md")

def main():
    """Pipeline principal d'optimisation."""
    log("🚀 DÉMARRAGE OPTIMISATION v2.3")
    log("="*60)
    
    # Créer dossier results
    os.makedirs('results', exist_ok=True)
    
    try:
        # 1. Charger données
        X, y, features = load_v23_data()
        
        # 2. Charger modèle original
        log("Chargement du modèle original...")
        original_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
        
        # 3. Optimiser hyperparamètres
        grid_search = optimize_hyperparameters(X, y)
        
        # 4. Entraîner modèle optimisé
        optimized_model = train_optimized_model(X, y, grid_search.best_params_)
        
        # 5. Comparer performances
        results = evaluate_comparison(X, y, original_model, optimized_model)
        
        # 6. Sauvegarder modèle optimisé
        model_path = f'models/v23_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(optimized_model, model_path)
        log(f"💾 Modèle optimisé sauvegardé: {model_path}")
        
        # 7. Sauvegarder rapports
        save_optimization_report(grid_search, results, grid_search.best_params_, features)
        
        log("🎉 OPTIMISATION TERMINÉE AVEC SUCCÈS!")
        log("="*60)
        
        improvement = (results['optimized']['cv_mean'] - results['original']['cv_mean']) * 100
        if improvement > 0:
            log(f"🏆 AMÉLIORATION: +{improvement:.2f}pp - Nouveau champion!")
        elif improvement > -0.5:
            log(f"📊 RÉSULTAT: {improvement:+.2f}pp - Performance équivalente")
        else:
            log(f"📉 RÉSULTAT: {improvement:+.2f}pp - Original reste meilleur")
            
        return optimized_model, results
        
    except Exception as e:
        log(f"❌ ERREUR: {str(e)}")
        raise

if __name__ == "__main__":
    main()