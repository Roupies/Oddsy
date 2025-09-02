#!/usr/bin/env python3
"""
test_xgboost_lightgbm_clean.py

TEST XGBoost & LightGBM sur 10 features nettoyées
Suite de l'analyse de redondance - Phase 2

OBJECTIF:
- Tester XGBoost vs LightGBM vs RandomForest
- Sur les 10 features optimales (vs 27 redondantes)
- Hyperparameter tuning pour chaque algorithme
- Objectif: Dépasser RandomForest 53.3% → 55%+

FEATURES OPTIMALES (10):
1. elo_diff_normalized ⭐
2. market_entropy_norm ⭐  
3. home_xg_eff_10
4. shots_diff_normalized
5. away_xg_eff_10
6. corners_diff_normalized
7. matchday_normalized ⭐
8. form_diff_normalized ⭐
9. away_goals_sum_5
10. h2h_score ⭐
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Tuple, Any

# ML imports
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class AlgorithmBenchmark:
    """
    Benchmark XGBoost vs LightGBM vs RandomForest sur features nettoyées
    """
    
    def __init__(self):
        self.logger = setup_logging()
        
        # 10 features optimales (post-analyse redondance)
        self.optimal_features = [
            'form_diff_normalized',      # ⭐ v2.1 core
            'elo_diff_normalized',       # ⭐ v2.1 core  
            'h2h_score',                 # ⭐ v2.1 core
            'matchday_normalized',       # ⭐ v2.1 core
            'shots_diff_normalized',
            'corners_diff_normalized',
            'market_entropy_norm',       # ⭐ v2.1 core
            'home_xg_eff_10',
            'away_goals_sum_5',
            'away_xg_eff_10'
        ]
        
        self.results = {}
        
    def load_clean_data(self, filepath='data/processed/v13_xg_safe_features.csv'):
        """
        Charger données avec features optimales uniquement
        """
        self.logger.info("📊 CHARGEMENT DES DONNÉES NETTOYÉES")
        self.logger.info("="*70)
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        self.logger.info(f"✅ Données brutes: {len(df)} matches")
        
        # Vérifier que toutes les features optimales existent
        missing_features = [f for f in self.optimal_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features manquantes: {missing_features}")
        
        self.logger.info(f"✅ 10 features optimales confirmées")
        
        # Filtrer données complètes
        valid_data = df.dropna(subset=self.optimal_features)
        self.logger.info(f"📊 Matches avec toutes features: {len(valid_data)}")
        
        # Split temporel (même que analyse redondance)
        train_end = pd.to_datetime('2024-05-19')
        test_start = pd.to_datetime('2024-08-16')
        
        train_data = valid_data[valid_data['Date'] <= train_end].copy()
        test_data = valid_data[valid_data['Date'] >= test_start].copy()
        
        self.logger.info(f"📊 Train: {len(train_data)} matches (jusqu'à {train_end.strftime('%Y-%m-%d')})")
        self.logger.info(f"📊 Test: {len(test_data)} matches (depuis {test_start.strftime('%Y-%m-%d')})")
        
        # Préparer matrices
        self.X_train = train_data[self.optimal_features].values
        self.X_test = test_data[self.optimal_features].values
        
        # Target encoding: H->0, D->1, A->2
        self.y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        self.y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        
        # Handle NaN avec valeur neutre
        self.X_train = np.nan_to_num(self.X_train, nan=0.5)
        self.X_test = np.nan_to_num(self.X_test, nan=0.5)
        
        self.logger.info(f"✅ Données prêtes:")
        self.logger.info(f"  X_train: {self.X_train.shape}")
        self.logger.info(f"  X_test: {self.X_test.shape}")
        self.logger.info(f"  Features: {self.optimal_features}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def train_randomforest(self):
        """
        RandomForest baseline optimisé (référence)
        """
        self.logger.info("\n🌲 RANDOMFOREST OPTIMISÉ (BASELINE)")
        self.logger.info("="*70)
        
        start_time = time.time()
        
        # Hyperparameter grid optimisé
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2', 0.8],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search avec TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=4)
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info(f"🔍 Grid search: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['max_features'])} combinaisons")
        
        # Training
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        
        # Calibration
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_model.fit(self.X_train, self.y_train)
        
        # Evaluation
        train_score = calibrated_model.score(self.X_train, self.y_train)
        test_score = calibrated_model.score(self.X_test, self.y_test)
        
        y_pred = calibrated_model.predict(self.X_test)
        y_proba = calibrated_model.predict_proba(self.X_test)
        
        training_time = time.time() - start_time
        
        self.logger.info(f"✅ RandomForest Results:")
        self.logger.info(f"  Best params: {grid_search.best_params_}")
        self.logger.info(f"  Best CV score: {grid_search.best_score_:.4f}")
        self.logger.info(f"  Train accuracy: {train_score:.4f}")
        self.logger.info(f"  Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        self.logger.info(f"  Training time: {training_time:.1f}s")
        
        # Feature importance
        feature_importance = dict(zip(self.optimal_features, best_model.feature_importances_))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"🏆 Top 5 features:")
        for i, (feat, imp) in enumerate(sorted_importance[:5], 1):
            self.logger.info(f"  {i}. {feat}: {imp:.3f}")
        
        self.results['randomforest'] = {
            'algorithm': 'RandomForest',
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'test_log_loss': float(log_loss(self.y_test, y_proba)),
            'training_time_seconds': training_time,
            'feature_importance': sorted_importance,
            'model': calibrated_model
        }
        
        return calibrated_model, test_score
    
    def train_xgboost(self):
        """
        XGBoost optimisé pour notre problème multiclass
        """
        self.logger.info("\n⚡ XGBOOST OPTIMISÉ")
        self.logger.info("="*70)
        
        start_time = time.time()
        
        # Hyperparameter grid pour XGBoost
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0.1, 1.0, 10.0]
        }
        
        # Paramètres XGBoost spécifiques
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # Grid search réduit pour efficacité
        reduced_param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0.1, 1.0]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)  # Réduit pour vitesse
        grid_search = GridSearchCV(
            xgb_model, reduced_param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info(f"🔍 XGBoost grid search: {len(reduced_param_grid['n_estimators']) * len(reduced_param_grid['max_depth']) * len(reduced_param_grid['learning_rate'])} combinaisons")
        
        # Training
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        
        # Calibration
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_model.fit(self.X_train, self.y_train)
        
        # Evaluation
        train_score = calibrated_model.score(self.X_train, self.y_train)
        test_score = calibrated_model.score(self.X_test, self.y_test)
        
        y_pred = calibrated_model.predict(self.X_test)
        y_proba = calibrated_model.predict_proba(self.X_test)
        
        training_time = time.time() - start_time
        
        self.logger.info(f"✅ XGBoost Results:")
        self.logger.info(f"  Best params: {grid_search.best_params_}")
        self.logger.info(f"  Best CV score: {grid_search.best_score_:.4f}")
        self.logger.info(f"  Train accuracy: {train_score:.4f}")
        self.logger.info(f"  Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        self.logger.info(f"  Training time: {training_time:.1f}s")
        
        # Feature importance XGBoost
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(self.optimal_features, best_model.feature_importances_))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"🏆 Top 5 features:")
            for i, (feat, imp) in enumerate(sorted_importance[:5], 1):
                self.logger.info(f"  {i}. {feat}: {imp:.3f}")
        else:
            sorted_importance = []
        
        self.results['xgboost'] = {
            'algorithm': 'XGBoost',
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'test_log_loss': float(log_loss(self.y_test, y_proba)),
            'training_time_seconds': training_time,
            'feature_importance': sorted_importance,
            'model': calibrated_model
        }
        
        return calibrated_model, test_score
    
    def train_lightgbm(self):
        """
        LightGBM optimisé
        """
        self.logger.info("\n💡 LIGHTGBM OPTIMISÉ")
        self.logger.info("="*70)
        
        start_time = time.time()
        
        # Hyperparameter grid pour LightGBM
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [6, 8, 10, -1],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0.1, 1.0, 10.0],
            'num_leaves': [31, 50, 100]
        }
        
        # Paramètres LightGBM spécifiques
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbosity=-1  # Silencer warnings
        )
        
        # Grid search réduit
        reduced_param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [8, 10, -1],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 1.0],
            'num_leaves': [31, 50]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            lgb_model, reduced_param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info(f"🔍 LightGBM grid search: {len(reduced_param_grid['n_estimators']) * len(reduced_param_grid['max_depth']) * len(reduced_param_grid['learning_rate'])} combinaisons")
        
        # Training
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        
        # Calibration
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_model.fit(self.X_train, self.y_train)
        
        # Evaluation
        train_score = calibrated_model.score(self.X_train, self.y_train)
        test_score = calibrated_model.score(self.X_test, self.y_test)
        
        y_pred = calibrated_model.predict(self.X_test)
        y_proba = calibrated_model.predict_proba(self.X_test)
        
        training_time = time.time() - start_time
        
        self.logger.info(f"✅ LightGBM Results:")
        self.logger.info(f"  Best params: {grid_search.best_params_}")
        self.logger.info(f"  Best CV score: {grid_search.best_score_:.4f}")
        self.logger.info(f"  Train accuracy: {train_score:.4f}")
        self.logger.info(f"  Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        self.logger.info(f"  Training time: {training_time:.1f}s")
        
        # Feature importance LightGBM
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(self.optimal_features, best_model.feature_importances_))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"🏆 Top 5 features:")
            for i, (feat, imp) in enumerate(sorted_importance[:5], 1):
                self.logger.info(f"  {i}. {feat}: {imp:.3f}")
        else:
            sorted_importance = []
        
        self.results['lightgbm'] = {
            'algorithm': 'LightGBM',
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'test_log_loss': float(log_loss(self.y_test, y_proba)),
            'training_time_seconds': training_time,
            'feature_importance': sorted_importance,
            'model': calibrated_model
        }
        
        return calibrated_model, test_score
    
    def compare_algorithms(self):
        """
        Comparaison finale et sélection du meilleur
        """
        self.logger.info("\n🏆 COMPARAISON FINALE DES ALGORITHMES")
        self.logger.info("="*80)
        
        # Trier par performance test
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['test_accuracy'], 
                               reverse=True)
        
        self.logger.info(f"\n📊 CLASSEMENT PAR PERFORMANCE:")
        self.logger.info("-"*80)
        for i, (algo, result) in enumerate(sorted_results, 1):
            self.logger.info(f"{i}. {result['algorithm']:15} : {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%) "
                           f"[CV: {result['best_cv_score']:.4f}] "
                           f"({result['training_time_seconds']:.0f}s)")
        
        # Meilleur algorithme
        best_algo, best_result = sorted_results[0]
        
        self.logger.info(f"\n🥇 MEILLEUR ALGORITHME: {best_result['algorithm']}")
        self.logger.info(f"  Performance: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
        self.logger.info(f"  Amélioration vs RandomForest baseline: {(best_result['test_accuracy'] - 0.533)*100:+.2f}pp")
        
        # Benchmark vs objectifs
        self.logger.info(f"\n📈 vs OBJECTIFS:")
        if best_result['test_accuracy'] >= 0.55:
            self.logger.info(f"  ✅ EXCELLENT (>55%): OBJECTIF ATTEINT!")
        elif best_result['test_accuracy'] >= 0.542:
            self.logger.info(f"  ✅ DÉPASSE v2.1 (54.2%): +{(best_result['test_accuracy']-0.542)*100:.1f}pp")
        else:
            self.logger.info(f"  ⚠️ Sous v2.1 (54.2%): {(best_result['test_accuracy']-0.542)*100:.1f}pp")
        
        # Analyse détaillée du meilleur
        y_pred = best_result['model'].predict(self.X_test)
        report = classification_report(self.y_test, y_pred, 
                                     target_names=['Home', 'Draw', 'Away'],
                                     output_dict=True)
        
        self.logger.info(f"\n📋 ANALYSE DÉTAILLÉE ({best_result['algorithm']}):")
        self.logger.info(f"  Home (precision/recall/f1): {report['0']['precision']:.3f}/{report['0']['recall']:.3f}/{report['0']['f1-score']:.3f}")
        self.logger.info(f"  Draw (precision/recall/f1): {report['1']['precision']:.3f}/{report['1']['recall']:.3f}/{report['1']['f1-score']:.3f}")
        self.logger.info(f"  Away (precision/recall/f1): {report['2']['precision']:.3f}/{report['2']['recall']:.3f}/{report['2']['f1-score']:.3f}")
        
        return best_algo, best_result
    
    def save_results(self):
        """
        Sauvegarder résultats complets
        """
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        
        # Supprimer modèles des résultats (non sérialisable)
        clean_results = {}
        for algo, result in self.results.items():
            clean_result = result.copy()
            if 'model' in clean_result:
                del clean_result['model']
            clean_results[algo] = clean_result
        
        report = {
            'timestamp': timestamp,
            'experiment': 'algorithm_benchmark_clean_features',
            'features_used': self.optimal_features,
            'n_features': len(self.optimal_features),
            'algorithms_tested': list(clean_results.keys()),
            'results': clean_results,
            'best_algorithm': max(clean_results.items(), key=lambda x: x[1]['test_accuracy']),
            'summary': {
                'best_performance': max(result['test_accuracy'] for result in clean_results.values()),
                'improvement_vs_baseline': max(result['test_accuracy'] for result in clean_results.values()) - 0.533,
                'training_times': {algo: result['training_time_seconds'] for algo, result in clean_results.items()},
                'beats_v21': max(result['test_accuracy'] for result in clean_results.values()) > 0.542,
                'achieves_excellent': max(result['test_accuracy'] for result in clean_results.values()) >= 0.55
            }
        }
        
        output_file = f'reports/algorithm_benchmark_{timestamp}.json'
        os.makedirs('reports', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\n✅ RÉSULTATS SAUVEGARDÉS: {output_file}")
        
        return report

def main():
    """
    Script principal de benchmark des algorithmes
    """
    print("🚀 BENCHMARK ALGORITHMES SUR FEATURES NETTOYÉES")
    print("="*80)
    print("Test: XGBoost vs LightGBM vs RandomForest")
    print("Features: 10 optimales (post-analyse redondance)")
    print("Objectif: Dépasser 55% (excellent)")
    print("="*80)
    
    benchmark = AlgorithmBenchmark()
    
    try:
        # 1. Charger données nettoyées
        X_train, y_train, X_test, y_test = benchmark.load_clean_data()
        
        # 2. Tester tous les algorithmes
        rf_model, rf_score = benchmark.train_randomforest()
        xgb_model, xgb_score = benchmark.train_xgboost()
        lgb_model, lgb_score = benchmark.train_lightgbm()
        
        # 3. Comparaison finale
        best_algo, best_result = benchmark.compare_algorithms()
        
        # 4. Sauvegarder
        report = benchmark.save_results()
        
        # 5. Status final
        print("\n" + "="*80)
        print("🎯 BENCHMARK TERMINÉ!")
        print(f"🥇 Meilleur: {best_result['algorithm']}")
        print(f"📊 Performance: {best_result['test_accuracy']:.1%}")
        print(f"🎯 vs Objectif 55%: {'✅ ATTEINT' if best_result['test_accuracy'] >= 0.55 else '❌ Pas encore'}")
        print(f"📈 vs v2.1 (54.2%): {(best_result['test_accuracy']-0.542)*100:+.1f}pp")
        print("="*80)
        
        return 0 if best_result['test_accuracy'] >= 0.55 else 1
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())