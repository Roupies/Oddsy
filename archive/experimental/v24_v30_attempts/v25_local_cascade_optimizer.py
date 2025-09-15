#!/usr/bin/env python3
"""
v25_local_cascade_optimizer.py

LOCAL EXECUTION - Optimisation cascade v2.5 simplifiée
Grids réduits pour exécution locale rapide (5-8 minutes)

ARCHITECTURE CASCADE v2.4:
- Stage 1: Draw vs Non-Draw (SMOTE + RandomForest + Calibration)
- Stage 2: Home vs Away (RandomForest + Calibration)
- Threshold optimization: Seuil dynamique

FEATURES: 10 features v2.4 validées uniquement
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    log_loss, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class LocalCascadeOptimizer:
    """
    v2.5 Local Cascade Optimizer - Grids simplifiés pour exécution rapide
    """
    
    def __init__(self):
        self.logger = setup_logging()
        
        # Features v2.4 validées (EXACTEMENT ces 10)
        self.features = [
            'elo_diff_normalized',       # 15.5% importance
            'market_entropy_norm',       # 12.5% importance  
            'home_xg_eff_10',           # 11.4% importance
            'away_xg_eff_10',           # 10.8% importance
            'shots_diff_normalized',     # 10.5% importance
            'corners_diff_normalized',   # 9.4% importance
            'matchday_normalized',       # 8.2% importance
            'form_diff_normalized',      # 7.7% importance
            'h2h_score',                 # 7.4% importance
            'away_goals_sum_5'           # 6.5% importance
        ]
        
        self.draw_classifier = None      
        self.home_away_classifier = None 
        self.scaler = StandardScaler()
        
        self.optimal_threshold = None
        self.optimization_results = {}
        
    def load_data(self, filepath='data/processed/v13_xg_corrected_features_latest.csv'):
        """
        Charger données v2.4 avec validation features
        """
        self.logger.info("🎯 LOCAL CASCADE v2.5 - CHARGEMENT DONNÉES")
        self.logger.info("="*60)
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        self.logger.info(f"📊 Dataset brut: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Vérifier disponibilité des 10 features
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            self.logger.error(f"❌ Features manquantes: {missing_features}")
            self.logger.info(f"📋 Features disponibles: {sorted(df.columns.tolist())}")
            raise ValueError(f"Features manquantes: {missing_features}")
        
        self.logger.info(f"✅ 10 features v2.4 trouvées: {self.features}")
        
        # Données complètes
        required_cols = self.features + ['FullTimeResult']
        valid_data = df.dropna(subset=required_cols)
        self.logger.info(f"📊 Données valides: {len(valid_data)} matches")
        
        # Split temporel v2.4 (IDENTIQUE)
        train_end = pd.to_datetime('2024-05-19')
        test_start = pd.to_datetime('2024-08-16')
        
        train_data = valid_data[valid_data['Date'] <= train_end].copy()
        test_data = valid_data[valid_data['Date'] >= test_start].copy()
        
        self.logger.info(f"📊 Split temporel: Train {len(train_data)}, Test {len(test_data)}")
        self.logger.info(f"📊 Gap train/test: {(test_start - train_end).days} jours")
        
        # Matrices features (10 features uniquement)
        self.X_train = train_data[self.features].values
        self.X_test = test_data[self.features].values
        
        # Targets cascade
        train_results = train_data['FullTimeResult'].values
        test_results = test_data['FullTimeResult'].values
        
        # Stage 1: Draw vs Non-Draw
        self.y_train_draw = (train_results == 'D').astype(int)
        self.y_test_draw = (test_results == 'D').astype(int)
        
        # Stage 2: Home vs Away (non-draws uniquement)
        non_draw_train_mask = train_results != 'D'
        non_draw_test_mask = test_results != 'D'
        
        self.X_train_ha = self.X_train[non_draw_train_mask]
        self.X_test_ha = self.X_test[non_draw_test_mask]
        
        self.y_train_ha = (train_results[non_draw_train_mask] == 'H').astype(int)
        self.y_test_ha = (test_results[non_draw_test_mask] == 'H').astype(int)
        
        # Target globale (H=0, D=1, A=2)
        self.y_train_global = pd.Series(train_results).map({'H': 0, 'D': 1, 'A': 2}).values
        self.y_test_global = pd.Series(test_results).map({'H': 0, 'D': 1, 'A': 2}).values
        
        # Standardisation
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.X_train_ha_scaled = self.scaler.transform(self.X_train_ha)
        self.X_test_ha_scaled = self.scaler.transform(self.X_test_ha)
        
        # Distribution
        draw_pct = np.mean(self.y_train_draw) * 100
        home_pct = np.mean(self.y_train_global == 0) * 100
        away_pct = np.mean(self.y_train_global == 2) * 100
        
        self.logger.info(f"📊 Distribution train: H {home_pct:.1f}%, D {draw_pct:.1f}%, A {away_pct:.1f}%")
        
        return True
        
    def optimize_stage1(self):
        """
        Stage 1: Draw Detector optimisé (grid réduit pour local)
        """
        self.logger.info("\n🎯 STAGE 1 OPTIMIZATION - DRAW DETECTOR")
        self.logger.info("="*60)
        
        # SMOTE pour équilibrer draws
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train_scaled, self.y_train_draw)
        
        draws_before = np.sum(self.y_train_draw)
        draws_after = np.sum(y_train_smote)
        total_after = len(y_train_smote)
        
        self.logger.info(f"📈 SMOTE: {draws_before} → {draws_after} draws ({draws_after/total_after:.1%})")
        
        # Grid search SIMPLIFIÉ (local execution)
        param_grid = {
            'n_estimators': [300, 500],              # 2 valeurs
            'max_depth': [15, None],                 # 2 valeurs  
            'min_samples_split': [5, 10],            # 2 valeurs
            'min_samples_leaf': [2, 4],              # 2 valeurs
            'max_features': ['sqrt'],                # 1 valeur
            'class_weight': ['balanced']             # 1 valeur
        }
        
        # Total: 2×2×2×2×1×1 = 16 combinaisons (très rapide)
        
        rf_draw = RandomForestClassifier(random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        self.logger.info("🔍 Grid Search Stage 1 (16 combinaisons)...")
        grid_search = GridSearchCV(
            rf_draw, param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_smote, y_train_smote)
        
        self.logger.info(f"✅ Meilleurs params: {grid_search.best_params_}")
        self.logger.info(f"✅ Score CV F1: {grid_search.best_score_:.4f}")
        
        # Calibration isotonic
        best_model = grid_search.best_estimator_
        calibrated_draw = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_draw.fit(self.X_train_scaled, self.y_train_draw)  # Original data
        
        self.draw_classifier = calibrated_draw
        
        # Évaluation Stage 1
        y_pred_draw = self.draw_classifier.predict(self.X_test_scaled)
        y_proba_draw = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test_draw, y_pred_draw, average='binary'
        )
        
        roc_auc = roc_auc_score(self.y_test_draw, y_proba_draw)
        
        self.logger.info(f"\n✅ STAGE 1 RESULTS:")
        self.logger.info(f"   F1-score: {f1:.4f}")
        self.logger.info(f"   Precision: {precision:.4f}")
        self.logger.info(f"   Recall: {recall:.4f}")
        self.logger.info(f"   ROC-AUC: {roc_auc:.4f}")
        
        self.optimization_results['stage1'] = {
            'best_params': grid_search.best_params_,
            'cv_score': float(grid_search.best_score_),
            'test_f1': float(f1),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'roc_auc': float(roc_auc)
        }
        
        return True
        
    def optimize_stage2(self):
        """
        Stage 2: Home vs Away optimisé (grid réduit)
        """
        self.logger.info("\n🏠 STAGE 2 OPTIMIZATION - HOME vs AWAY")
        self.logger.info("="*60)
        
        # Grid search SIMPLIFIÉ
        param_grid = {
            'n_estimators': [300, 500],              # 2 valeurs
            'max_depth': [20, None],                 # 2 valeurs
            'min_samples_split': [5, 10],            # 2 valeurs  
            'min_samples_leaf': [2, 4],              # 2 valeurs
            'max_features': ['sqrt'],                # 1 valeur
            'class_weight': [None, 'balanced']       # 2 valeurs
        }
        
        # Total: 2×2×2×2×1×2 = 32 combinaisons
        
        rf_ha = RandomForestClassifier(random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        self.logger.info("🔍 Grid Search Stage 2 (32 combinaisons)...")
        grid_search = GridSearchCV(
            rf_ha, param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train_ha_scaled, self.y_train_ha)
        
        self.logger.info(f"✅ Meilleurs params: {grid_search.best_params_}")
        self.logger.info(f"✅ Score CV accuracy: {grid_search.best_score_:.4f}")
        
        # Calibration
        best_model = grid_search.best_estimator_
        calibrated_ha = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_ha.fit(self.X_train_ha_scaled, self.y_train_ha)
        
        self.home_away_classifier = calibrated_ha
        
        # Évaluation Stage 2
        y_pred_ha = self.home_away_classifier.predict(self.X_test_ha_scaled)
        accuracy_ha = accuracy_score(self.y_test_ha, y_pred_ha)
        
        self.logger.info(f"\n✅ STAGE 2 RESULTS:")
        self.logger.info(f"   Accuracy H vs A: {accuracy_ha:.4f} ({accuracy_ha*100:.2f}%)")
        
        self.optimization_results['stage2'] = {
            'best_params': grid_search.best_params_,
            'cv_score': float(grid_search.best_score_),
            'test_accuracy': float(accuracy_ha)
        }
        
        return True
        
    def optimize_threshold(self, metric='f1_macro'):
        """
        Optimisation seuil cascade (simplifié pour local)
        """
        self.logger.info(f"\n🎯 THRESHOLD OPTIMIZATION ({metric})")
        self.logger.info("="*60)
        
        # Probabilités draw
        draw_probas = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        # Test seuils (simplifié)
        threshold_range = np.arange(0.2, 0.7, 0.05)  # 10 seuils
        best_score = -1
        best_threshold = 0.4
        
        results_analysis = []
        
        self.logger.info(f"🔍 Test {len(threshold_range)} seuils...")
        
        for threshold in threshold_range:
            # Cascade prediction avec ce seuil
            draw_pred = (draw_probas >= threshold).astype(int)
            
            # Prédictions finales
            final_preds = np.zeros(len(self.X_test), dtype=int)
            
            # Matches prédits draws
            draw_mask = draw_pred == 1
            final_preds[draw_mask] = 1  # D = 1
            
            # Matches non-draws → Stage 2
            non_draw_mask = draw_pred == 0
            if np.sum(non_draw_mask) > 0:
                X_non_draw = self.X_test_scaled[non_draw_mask]
                ha_pred = self.home_away_classifier.predict(X_non_draw)
                final_preds[non_draw_mask] = np.where(ha_pred == 1, 0, 2)  # H=0, A=2
            
            # Calcul métriques
            if metric == 'f1_macro':
                score = f1_score(self.y_test_global, final_preds, average='macro')
            else:
                score = accuracy_score(self.y_test_global, final_preds)
                
            # Métriques détaillées
            accuracy = accuracy_score(self.y_test_global, final_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test_global, final_preds, average=None, zero_division=0
            )
            
            draw_recall = recall[1] if len(recall) > 1 else 0.0
            f1_macro = f1_score(self.y_test_global, final_preds, average='macro')
            
            results_analysis.append({
                'threshold': float(threshold),
                'metric_score': float(score),
                'accuracy': float(accuracy),
                'draw_recall': float(draw_recall),
                'f1_macro': float(f1_macro)
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        self.logger.info(f"✅ Seuil optimal: {best_threshold:.3f}")
        self.logger.info(f"✅ Score ({metric}): {best_score:.4f}")
        
        # Top 5 seuils
        top_thresholds = sorted(results_analysis, 
                              key=lambda x: x['metric_score'], reverse=True)[:5]
        
        self.logger.info(f"\n📊 Top 5 seuils:")
        for i, result in enumerate(top_thresholds):
            self.logger.info(f"   {i+1}. {result['threshold']:.3f}: "
                           f"F1-macro {result['f1_macro']:.4f}, "
                           f"Acc {result['accuracy']:.4f}, "
                           f"Draw-Recall {result['draw_recall']:.4f}")
        
        self.optimization_results['threshold_optimization'] = {
            'optimal_threshold': float(best_threshold),
            'best_score': float(best_score),
            'metric_optimized': metric,
            'top_thresholds': top_thresholds
        }
        
        return best_threshold
        
    def predict_cascade(self, X, threshold=None):
        """
        Prédiction cascade avec seuil optimisé
        """
        if threshold is None:
            threshold = self.optimal_threshold or 0.4
            
        X_scaled = self.scaler.transform(X)
        
        # Stage 1: Probabilités draws
        draw_proba = self.draw_classifier.predict_proba(X_scaled)[:, 1]
        draw_pred = (draw_proba >= threshold).astype(int)
        
        # Prédictions finales
        final_predictions = np.zeros(len(X), dtype=int)
        
        # Draws
        draw_mask = draw_pred == 1
        final_predictions[draw_mask] = 1
        
        # Non-draws
        non_draw_mask = draw_pred == 0
        if np.sum(non_draw_mask) > 0:
            X_non_draw = X_scaled[non_draw_mask]
            ha_pred = self.home_away_classifier.predict(X_non_draw)
            final_predictions[non_draw_mask] = np.where(ha_pred == 1, 0, 2)
        
        return final_predictions
        
    def evaluate_final(self):
        """
        Évaluation finale complète
        """
        self.logger.info(f"\n🏆 ÉVALUATION FINALE v2.5 (Seuil: {self.optimal_threshold:.3f})")
        self.logger.info("="*60)
        
        # Prédictions avec seuil optimisé
        y_pred_final = self.predict_cascade(self.X_test)
        
        # Métriques globales
        accuracy = accuracy_score(self.y_test_global, y_pred_final)
        
        # Rapport par classe
        report = classification_report(
            self.y_test_global, 
            y_pred_final,
            target_names=['Home', 'Draw', 'Away'],
            output_dict=True,
            zero_division=0
        )
        
        # Métriques clés
        home_f1 = report['Home']['f1-score']
        draw_f1 = report['Draw']['f1-score']
        draw_recall = report['Draw']['recall']
        away_f1 = report['Away']['f1-score']
        f1_macro = report['macro avg']['f1-score']
        
        # Comparaison v2.4 baseline
        v24_accuracy = 0.530
        v24_draw_recall = 0.344
        v24_f1_macro = 0.507
        
        self.logger.info(f"🎯 RÉSULTATS FINAUX:")
        self.logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"  F1-macro: {f1_macro:.4f}")
        self.logger.info(f"  Draw Recall: {draw_recall:.4f} ({draw_recall*100:.1f}%)")
        
        self.logger.info(f"\n📊 Performance par classe:")
        self.logger.info(f"  Home F1: {home_f1:.4f}")
        self.logger.info(f"  Draw F1: {draw_f1:.4f}")
        self.logger.info(f"  Away F1: {away_f1:.4f}")
        
        self.logger.info(f"\n📈 vs v2.4 BASELINE:")
        accuracy_gain = accuracy - v24_accuracy
        draw_gain = draw_recall - v24_draw_recall  
        f1_gain = f1_macro - v24_f1_macro
        
        self.logger.info(f"  Accuracy: {accuracy_gain:+.3f}pp ({accuracy_gain/v24_accuracy*100:+.1f}%)")
        self.logger.info(f"  Draw Recall: {draw_gain:+.3f}pp ({draw_gain/v24_draw_recall*100:+.1f}%)")
        self.logger.info(f"  F1-macro: {f1_gain:+.3f}pp ({f1_gain/v24_f1_macro*100:+.1f}%)")
        
        # Matrice confusion
        cm = confusion_matrix(self.y_test_global, y_pred_final)
        self.logger.info(f"\n📊 Matrice Confusion:")
        self.logger.info(f"       H    D    A")
        for i, label in enumerate(['H', 'D', 'A']):
            row = ' '.join([f'{cm[i][j]:4d}' for j in range(3)])
            self.logger.info(f"  {label}: {row}")
        
        # Évaluation succès
        success_criteria = {
            'maintain_draw_recall': draw_recall >= 0.25,  # 25% minimum
            'improve_vs_v24': accuracy >= v24_accuracy,
            'balanced_performance': f1_macro >= 0.45
        }
        
        success = all(success_criteria.values())
        
        status = "✅ SUCCÈS" if success else "⚠️ PARTIEL"
        self.logger.info(f"\n{status}: v2.5 Local Optimization")
        
        if not success:
            failed = [k for k, v in success_criteria.items() if not v]
            self.logger.info(f"   Critères manqués: {failed}")
        
        # Résultats finaux
        final_results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'draw_recall': float(draw_recall),
            'draw_f1': float(draw_f1),
            'home_f1': float(home_f1),
            'away_f1': float(away_f1),
            'optimal_threshold': float(self.optimal_threshold),
            'vs_v24_baseline': {
                'accuracy_gain': float(accuracy_gain),
                'draw_recall_gain': float(draw_gain),
                'f1_macro_gain': float(f1_gain)
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'success_criteria': success_criteria,
            'overall_success': success
        }
        
        self.optimization_results['final_evaluation'] = final_results
        
        return success, accuracy, draw_recall, f1_macro

def main():
    """
    Pipeline principal local v2.5
    """
    print("🎯 LOCAL CASCADE OPTIMIZATION v2.5")
    print("="*60)
    print("FEATURES: 10 v2.4 validées uniquement")
    print("GRIDS: Réduits pour exécution locale (5-8 mins)")
    print("ARCHITECTURE: 2-stage cascade + threshold optimization")
    print("="*60)
    
    start_time = datetime.now()
    
    optimizer = LocalCascadeOptimizer()
    
    try:
        # 1. Load data
        print("\n📊 Étape 1/4: Chargement données...")
        step_start = datetime.now()
        optimizer.load_data()
        step_duration = (datetime.now() - step_start).total_seconds()
        print(f"✅ Données chargées en {step_duration:.1f}s")
        
        # 2. Stage 1 optimization  
        print("\n🎯 Étape 2/4: Optimisation Stage 1...")
        step_start = datetime.now()
        optimizer.optimize_stage1()
        step_duration = (datetime.now() - step_start).total_seconds()
        print(f"✅ Stage 1 optimisé en {step_duration:.1f}s")
        
        # 3. Stage 2 optimization
        print("\n🏠 Étape 3/4: Optimisation Stage 2...")
        step_start = datetime.now()
        optimizer.optimize_stage2()
        step_duration = (datetime.now() - step_start).total_seconds()
        print(f"✅ Stage 2 optimisé en {step_duration:.1f}s")
        
        # 4. Threshold + final evaluation
        print("\n🎯 Étape 4/4: Optimisation seuil + évaluation...")
        step_start = datetime.now()
        optimal_threshold = optimizer.optimize_threshold(metric='f1_macro')
        success, accuracy, draw_recall, f1_macro = optimizer.evaluate_final()
        step_duration = (datetime.now() - step_start).total_seconds()
        print(f"✅ Optimisation finale en {step_duration:.1f}s")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Sauvegarder modèle
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        model_path = f'models/v25_local_optimized_cascade_{timestamp}.joblib'
        
        optimized_model = {
            'draw_classifier': optimizer.draw_classifier,
            'home_away_classifier': optimizer.home_away_classifier,
            'scaler': optimizer.scaler,
            'features': optimizer.features,
            'optimal_threshold': optimizer.optimal_threshold,
            'optimization_results': optimizer.optimization_results
        }
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(optimized_model, model_path)
        
        # Rapport final
        report_path = f'evaluation/reports/v25_local_optimization_report_{timestamp}.json'
        os.makedirs('evaluation/reports', exist_ok=True)
        
        final_report = {
            'timestamp': timestamp,
            'version': 'v2.5_local_cascade_optimization',
            'execution_time_seconds': total_time,
            'features_count': len(optimizer.features),
            'features_used': optimizer.features,
            'optimization_results': optimizer.optimization_results,
            'model_path': model_path
        }
        
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # STATUS FINAL
        print("\n" + "="*60)
        print("🏆 LOCAL CASCADE OPTIMIZATION v2.5 COMPLETE!")
        print("="*60)
        print(f"⏱️  Temps total: {total_time/60:.1f} minutes")
        print(f"🎯  Accuracy: {accuracy:.1%} (vs v2.4: {accuracy-0.530:+.1%})")
        print(f"📊  Draw Recall: {draw_recall:.1%} (vs v2.4: {draw_recall-0.344:+.1%})")
        print(f"⚖️  F1-macro: {f1_macro:.1%}")
        print(f"🔧  Seuil optimal: {optimal_threshold:.3f}")
        print(f"✅  Succès: {'OUI' if success else 'PARTIEL'}")
        print(f"💾  Modèle: {model_path}")
        print(f"📋  Rapport: {report_path}")
        print("="*60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ ERREUR LOCAL OPTIMIZATION: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())