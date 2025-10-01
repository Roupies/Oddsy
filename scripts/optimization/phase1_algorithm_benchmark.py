#!/usr/bin/env python3
"""
🚀 PHASE 1.1 - ALGORITHM BENCHMARK

Tester 4 algorithmes sur les 10 features actuelles pour identifier 
le meilleur baseline avant feature engineering.

OBJECTIF: Trouver algorithme naturellement plus robuste que RandomForest (43.3%)
TARGET: Potentiel +2-4pp juste par changement d'algorithme
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime

def load_clean_data():
    """
    Charger données avec validation stricte - dataset propre uniquement
    """
    print("📊 CHARGEMENT DONNÉES PROPRES (validation anti-leakage)")
    
    # Utiliser le dataset v15 validé (pas de leakage)
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    
    # Split temporel strict: Train 2019-2025, Test EPL 2025-26
    cutoff_date = pd.Timestamp('2025-08-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    print(f"✅ Train set: {len(train_df)} matches (2019-2025)")
    print(f"✅ Test set: {len(test_df)} matches (EPL 2025-26)")
    
    return train_df, test_df

def benchmark_algorithms():
    """
    Benchmark 4 algorithmes sur features actuelles v2.3
    """
    print()
    print("🎯 PHASE 1.1 - ALGORITHM BENCHMARK")
    print("=" * 60)
    
    # Charger données
    train_df, test_df = load_clean_data()
    
    # Features v2.3 actuelles
    features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Nettoyer données (supprimer NaN)
    train_clean = train_df.dropna(subset=features + ['FullTimeResult'])
    test_clean = test_df.dropna(subset=features + ['FullTimeResult'])
    
    # Préparer données
    X_train = train_clean[features]
    y_train = train_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    X_test = test_clean[features]
    y_test = test_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print(f"🧹 Nettoyage: Train {len(train_df)}→{len(train_clean)}, Test {len(test_df)}→{len(test_clean)}")
    
    print(f"📋 Features testées: {len(features)}")
    print(f"🎓 Training: {len(X_train)} samples")
    print(f"🧪 Testing: {len(X_test)} samples")
    
    # Configuration des 4 algorithmes
    algorithms = {
        'RandomForest': {
            'model': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                max_features='sqrt',
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            ),
            'description': 'Baseline actuel v2.3'
        },
        
        'XGBoost_Regularized': {
            'model': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,          # Réduit vs défaut 6
                learning_rate=0.05,   # Réduit vs défaut 0.1  
                reg_alpha=5,          # L1 regularization
                reg_lambda=5,         # L2 regularization
                min_child_weight=10,  # Contre overfitting
                subsample=0.8,        # Bagging
                colsample_bytree=0.8, # Feature sampling
                objective='multi:softprob',
                random_state=42
            ),
            'description': 'XGBoost ultra-régularisé'
        },
        
        'LightGBM_Regularized': {
            'model': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                reg_alpha=5,
                reg_lambda=5,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multiclass',
                random_state=42,
                verbose=-1
            ),
            'description': 'LightGBM ultra-régularisé'
        },
        
        'LogisticRegression': {
            'model': LogisticRegression(
                penalty='elasticnet',
                C=0.1,               # Forte régularisation
                l1_ratio=0.5,        # Mix L1+L2
                solver='saga',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            'description': 'Régression logistique régularisée'
        }
    }
    
    print()
    print("🔬 BENCHMARK ALGORITHMES:")
    print("-" * 60)
    
    results = {}
    
    for name, config in algorithms.items():
        print(f"\\n🧪 TEST: {name}")
        print(f"📝 {config['description']}")
        
        model = config['model']
        
        # Cross-validation temporelle
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"📊 Cross-Validation: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Entraînement et test final
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Probabilités pour log-loss
        y_proba = model.predict_proba(X_test)
        test_logloss = log_loss(y_test, y_proba)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.3f} ({np.sum(y_pred == y_test)}/{len(y_test)})")
        print(f"📈 Test Log-Loss: {test_logloss:.3f}")
        
        # Analyse par classe
        class_report = classification_report(y_test, y_pred, 
                                           target_names=['H', 'D', 'A'],
                                           output_dict=True)
        
        print(f"🏠 Home Recall: {class_report['H']['recall']:.3f}")
        print(f"🤝 Draw Recall: {class_report['D']['recall']:.3f}")  
        print(f"✈️ Away Recall: {class_report['A']['recall']:.3f}")
        
        # Stockage résultats
        results[name] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_accuracy,
            'test_logloss': test_logloss,
            'class_report': class_report,
            'model': model
        }
    
    # ANALYSE COMPARATIVE
    print()
    print("=" * 60)
    print("📊 ANALYSE COMPARATIVE FINALE:")
    print("-" * 60)
    
    # Trier par test accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    baseline_acc = results['RandomForest']['test_accuracy']
    
    print(f"\\n🏆 CLASSEMENT FINAL:")
    for i, (name, data) in enumerate(sorted_results):
        rank = i + 1
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📍"
        
        improvement = (data['test_accuracy'] - baseline_acc) * 100
        cv_stability = "STABLE" if data['cv_std'] < 0.03 else "INSTABLE"
        
        print(f"{emoji} {rank}. {name:20}: {data['test_accuracy']:.1%} ({improvement:+.1f}pp vs RF) | CV: {cv_stability}")
    
    # RECOMMANDATION
    best_model_name = sorted_results[0][0]
    best_model_data = sorted_results[0][1]
    improvement = (best_model_data['test_accuracy'] - baseline_acc) * 100
    
    print()
    print("💡 RECOMMANDATION FINALE:")
    print(f"🏆 Meilleur algorithme: {best_model_name}")
    print(f"📈 Amélioration vs RandomForest: {improvement:+.1f}pp")
    print(f"🎯 Performance: {best_model_data['test_accuracy']:.1%}")
    print(f"📊 Stabilité CV: {best_model_data['cv_mean']:.3f} ± {best_model_data['cv_std']:.3f}")
    
    if improvement > 1.0:
        print("✅ GAIN SIGNIFICATIF détecté! Changer d'algorithme recommandé.")
    elif improvement > 0.5:
        print("⚠️ GAIN MARGINAL détecté. Changement optionnel.")  
    else:
        print("❌ PAS DE GAIN détecté. RandomForest reste optimal.")
    
    # ANALYSE DES DRAWS
    print()
    print("🤝 ANALYSE SPÉCIALE - DÉTECTION DRAWS:")
    for name, data in sorted_results:
        draw_recall = data['class_report']['D']['recall']
        draw_precision = data['class_report']['D']['precision']
        draw_f1 = data['class_report']['D']['f1-score']
        
        print(f"{name:20}: Recall={draw_recall:.3f} | Precision={draw_precision:.3f} | F1={draw_f1:.3f}")
    
    # Sauvegarder meilleur modèle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = f"models/phase1_best_algorithm_{timestamp}.joblib"
    
    joblib.dump(best_model_data['model'], best_model_path)
    
    print(f"\\n💾 Meilleur modèle sauvé: {best_model_path}")
    
    # Sauvegarder résultats
    results_summary = {
        'timestamp': timestamp,
        'best_algorithm': best_model_name,
        'improvement_pp': improvement,
        'baseline_accuracy': baseline_acc,
        'best_accuracy': best_model_data['test_accuracy'],
        'detailed_results': {k: {
            'cv_mean': v['cv_mean'],
            'cv_std': v['cv_std'], 
            'test_accuracy': v['test_accuracy'],
            'test_logloss': v['test_logloss']
        } for k, v in results.items()}
    }
    
    import json
    results_path = f"evaluation/phase1_algorithm_benchmark_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"📋 Résultats détaillés: {results_path}")
    
    return best_model_name, best_model_data, results

if __name__ == "__main__":
    best_algorithm, best_data, all_results = benchmark_algorithms()
    
    print()
    print("🚀 PHASE 1.1 TERMINÉE")
    print("📋 Prochaine étape: Feature Selection (Phase 1.2)")