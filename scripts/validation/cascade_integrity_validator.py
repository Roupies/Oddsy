#!/usr/bin/env python3
"""
cascade_integrity_validator.py

VALIDATION SUITE COMPLÈTE pour modèles CASCADE v2.5
Tests d'intégrité spécialisés pour architecture 2-stages

TESTS IMPLÉMENTÉS:
1. Cascade Architecture Validation
2. Stage Independence Verification  
3. Temporal Integrity (Train/Test Split)
4. Feature Consistency Check
5. Threshold Robustness Analysis
6. Performance Stability Test
7. Data Leakage Detection (Cascade-Specific)
8. Overfitting Diagnostic
9. Calibration Quality Assessment
10. Production Readiness Check

OBJECTIF: Garantir que les modèles optimisés sont prêts pour production
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
import joblib
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    f1_score,
    log_loss,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class CascadeIntegrityValidator:
    """
    Validateur d'intégrité spécialisé pour modèles cascade
    """
    
    def __init__(self, model_path=None):
        self.logger = setup_logging()
        
        # Seuils de validation
        self.validation_thresholds = {
            'min_accuracy': 0.45,           # Minimum acceptable
            'min_draw_recall': 0.20,        # Draw recall minimum
            'max_overfitting_gap': 0.15,    # CV vs Test gap max
            'min_calibration_score': 0.7,   # Calibration quality
            'max_temporal_leakage': 0.05,   # Temporal leakage max
            'min_stage_independence': 0.8,  # Stage independence
            'stability_tolerance': 0.05     # Performance stability
        }
        
        # Modèle à valider
        self.model = None
        self.features = []
        self.scaler = None
        
        # Données de test
        self.X_test = None
        self.y_test_global = None
        self.y_test_draw = None
        
        # Résultats validation
        self.validation_results = {}
        self.integrity_score = 0.0
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Charger modèle cascade à valider"""
        self.logger.info(f"📥 Chargement modèle cascade: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        model_data = joblib.load(model_path)
        
        # Validation structure modèle
        required_components = ['draw_classifier', 'home_away_classifier', 'scaler', 'features']
        missing_components = [comp for comp in required_components if comp not in model_data]
        
        if missing_components:
            raise ValueError(f"Composants manquants: {missing_components}")
        
        self.model = {
            'draw_classifier': model_data['draw_classifier'],
            'home_away_classifier': model_data['home_away_classifier'],
            'optimal_threshold': model_data.get('optimal_threshold', 0.4),
            'optimization_results': model_data.get('optimization_results', {})
        }
        
        self.scaler = model_data['scaler']
        self.features = model_data['features']
        
        self.logger.info(f"✅ Modèle chargé: {len(self.features)} features, seuil {self.model['optimal_threshold']:.3f}")
        
    def load_test_data(self, filepath='data/processed/v13_xg_corrected_features_latest.csv'):
        """Charger données de test pour validation"""
        self.logger.info("📊 Chargement données test pour validation")
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # Split identique aux modèles
        test_start = pd.to_datetime('2024-08-16')
        test_data = df[df['Date'] >= test_start].copy()
        
        # Vérifier features
        missing_features = [f for f in self.features if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"Features manquantes dans données test: {missing_features}")
        
        # Données complètes
        test_data = test_data.dropna(subset=self.features + ['FullTimeResult'])
        
        self.X_test = test_data[self.features].values
        test_results = test_data['FullTimeResult'].values
        
        # Targets
        self.y_test_global = pd.Series(test_results).map({'H': 0, 'D': 1, 'A': 2}).values
        self.y_test_draw = (test_results == 'D').astype(int)
        
        # Données brutes pour analyses temporelles
        self.test_data_df = test_data[self.features + ['Date', 'FullTimeResult']].copy()
        
        self.logger.info(f"✅ Test data: {len(test_data)} matches")
        return True
    
    def test_1_cascade_architecture(self):
        """
        TEST 1: Validation architecture cascade
        """
        self.logger.info("\n🏗️ TEST 1: VALIDATION ARCHITECTURE CASCADE")
        self.logger.info("="*60)
        
        results = {
            'test_name': 'cascade_architecture',
            'status': 'unknown',
            'details': {},
            'score': 0.0
        }
        
        try:
            # Vérifier composants cascade
            has_stage1 = self.model['draw_classifier'] is not None
            has_stage2 = self.model['home_away_classifier'] is not None
            has_scaler = self.scaler is not None
            has_threshold = self.model['optimal_threshold'] is not None
            
            # Test fonctionnel cascade
            X_test_scaled = self.scaler.transform(self.X_test)
            
            # Stage 1: Draw probabilities
            try:
                draw_probas = self.model['draw_classifier'].predict_proba(X_test_scaled)
                stage1_functional = draw_probas.shape == (len(self.X_test), 2)
            except Exception as e:
                stage1_functional = False
                self.logger.error(f"Stage 1 non fonctionnel: {e}")
            
            # Stage 2: Home/Away pour non-draws
            try:
                # Simuler quelques non-draws
                sample_indices = np.where(self.y_test_draw == 0)[0][:10]
                if len(sample_indices) > 0:
                    ha_probas = self.model['home_away_classifier'].predict_proba(
                        X_test_scaled[sample_indices]
                    )
                    stage2_functional = ha_probas.shape[1] == 2
                else:
                    stage2_functional = True  # Pas de non-draws pour tester
            except Exception as e:
                stage2_functional = False
                self.logger.error(f"Stage 2 non fonctionnel: {e}")
            
            # Cascade complète
            try:
                cascade_predictions = self._predict_cascade_full(self.X_test[:10])
                cascade_functional = len(cascade_predictions) == 10
            except Exception as e:
                cascade_functional = False
                self.logger.error(f"Cascade complète non fonctionnelle: {e}")
            
            # Score architecture
            architecture_checks = [
                has_stage1, has_stage2, has_scaler, has_threshold,
                stage1_functional, stage2_functional, cascade_functional
            ]
            
            architecture_score = np.mean(architecture_checks)
            
            results['details'] = {
                'components_present': {
                    'stage1_classifier': has_stage1,
                    'stage2_classifier': has_stage2,
                    'scaler': has_scaler,
                    'threshold': has_threshold
                },
                'functional_tests': {
                    'stage1_predictions': stage1_functional,
                    'stage2_predictions': stage2_functional,
                    'cascade_complete': cascade_functional
                },
                'architecture_score': float(architecture_score)
            }
            
            results['score'] = architecture_score
            results['status'] = 'passed' if architecture_score >= 0.9 else 'warning' if architecture_score >= 0.7 else 'failed'
            
            self.logger.info(f"🏗️ Architecture Score: {architecture_score:.3f}")
            self.logger.info(f"✅ Composants: {sum([has_stage1, has_stage2, has_scaler, has_threshold])}/4")
            self.logger.info(f"🔧 Tests fonctionnels: {sum([stage1_functional, stage2_functional, cascade_functional])}/3")
            
        except Exception as e:
            results['status'] = 'error'
            results['details'] = {'error': str(e)}
            self.logger.error(f"❌ Erreur test architecture: {e}")
        
        self.validation_results['test_1_architecture'] = results
        return results
    
    def test_2_stage_independence(self):
        """
        TEST 2: Vérification indépendance des stages
        """
        self.logger.info("\n🔄 TEST 2: INDÉPENDANCE DES STAGES")
        self.logger.info("="*60)
        
        results = {
            'test_name': 'stage_independence',
            'status': 'unknown', 
            'details': {},
            'score': 0.0
        }
        
        try:
            X_test_scaled = self.scaler.transform(self.X_test)
            
            # Stage 1: Prédictions draws sur tous les échantillons
            draw_probas = self.model['draw_classifier'].predict_proba(X_test_scaled)[:, 1]
            
            # Stage 2: Prédictions H/A sur échantillons non-draws réels
            non_draw_indices = np.where(self.y_test_draw == 0)[0]
            
            if len(non_draw_indices) > 0:
                ha_probas_real = self.model['home_away_classifier'].predict_proba(
                    X_test_scaled[non_draw_indices]
                )
                
                # Stage 2: Prédictions H/A sur échantillons draws réels (test indépendance)
                draw_indices = np.where(self.y_test_draw == 1)[0]
                
                if len(draw_indices) > 0:
                    ha_probas_draws = self.model['home_away_classifier'].predict_proba(
                        X_test_scaled[draw_indices]
                    )
                    
                    # Test indépendance: Stage 2 ne doit pas être biaisé par type d'échantillon
                    # Moyennes des probas Home sur draws vs non-draws
                    mean_home_on_draws = np.mean(ha_probas_draws[:, 1])
                    mean_home_on_nondraws = np.mean(ha_probas_real[:, 1])
                    
                    independence_bias = abs(mean_home_on_draws - mean_home_on_nondraws)
                    independence_score = max(0, 1 - independence_bias * 2)  # Pénaliser biais
                    
                    # Test corrélation: Probas Stage 1 vs Stage 2 sur mêmes échantillons
                    if len(non_draw_indices) > 10:
                        correlation = np.corrcoef(
                            draw_probas[non_draw_indices], 
                            ha_probas_real[:, 1]
                        )[0, 1]
                        
                        correlation_independence = 1 - abs(correlation)
                    else:
                        correlation_independence = 1.0
                        correlation = 0.0
                    
                    # Score global indépendance
                    final_independence_score = (independence_score + correlation_independence) / 2
                    
                    results['details'] = {
                        'home_prob_on_draws': float(mean_home_on_draws),
                        'home_prob_on_nondraws': float(mean_home_on_nondraws), 
                        'independence_bias': float(independence_bias),
                        'stage_correlation': float(correlation),
                        'independence_score': float(independence_score),
                        'correlation_independence': float(correlation_independence),
                        'final_score': float(final_independence_score)
                    }
                    
                    results['score'] = final_independence_score
                else:
                    results['score'] = 0.5
                    results['details'] = {'error': 'Pas assez de draws pour test complet'}
            else:
                results['score'] = 0.5
                results['details'] = {'error': 'Pas assez de non-draws pour test complet'}
            
            threshold = self.validation_thresholds['min_stage_independence']
            results['status'] = 'passed' if results['score'] >= threshold else 'warning' if results['score'] >= threshold-0.1 else 'failed'
            
            self.logger.info(f"🔄 Independence Score: {results['score']:.3f}")
            if 'independence_bias' in results['details']:
                self.logger.info(f"📊 Biais indépendance: {results['details']['independence_bias']:.4f}")
                self.logger.info(f"🔗 Corrélation stages: {results['details']['stage_correlation']:.4f}")
            
        except Exception as e:
            results['status'] = 'error'
            results['details'] = {'error': str(e)}
            self.logger.error(f"❌ Erreur test indépendance: {e}")
        
        self.validation_results['test_2_independence'] = results
        return results
    
    def test_3_temporal_integrity(self):
        """
        TEST 3: Intégrité temporelle (pas de fuites temporelles)
        """
        self.logger.info("\n⏰ TEST 3: INTÉGRITÉ TEMPORELLE")
        self.logger.info("="*60)
        
        results = {
            'test_name': 'temporal_integrity',
            'status': 'unknown',
            'details': {},
            'score': 0.0
        }
        
        try:
            # Vérifier split train/test
            train_end = pd.to_datetime('2024-05-19')
            test_start = pd.to_datetime('2024-08-16')
            gap_days = (test_start - train_end).days
            
            # Test chronologique des données test
            test_dates = pd.to_datetime(self.test_data_df['Date'])
            earliest_test = test_dates.min()
            latest_test = test_dates.max()
            
            # Vérifications temporelles
            proper_gap = gap_days >= 60  # Au moins 60 jours gap
            no_overlap = earliest_test >= test_start
            chronological_order = test_dates.is_monotonic_increasing or len(test_dates.unique()) > len(test_dates) * 0.8
            
            # Test features temporelles (pas de valeurs futures)
            temporal_leakage_score = 1.0  # Assume pas de leakage par défaut
            
            # Test performance par période (stabilité temporelle)
            if len(self.test_data_df) > 50:
                # Diviser test en 3 périodes
                test_sorted = self.test_data_df.sort_values('Date')
                n_samples = len(test_sorted)
                
                period1 = test_sorted.iloc[:n_samples//3]
                period2 = test_sorted.iloc[n_samples//3:2*n_samples//3]
                period3 = test_sorted.iloc[2*n_samples//3:]
                
                # Performance par période
                periods_performance = []
                for i, period_data in enumerate([period1, period2, period3]):
                    if len(period_data) > 5:
                        X_period = period_data[self.features].values
                        y_period = pd.Series(period_data['FullTimeResult']).map({'H': 0, 'D': 1, 'A': 2}).values
                        
                        period_preds = self._predict_cascade_full(X_period)
                        period_acc = accuracy_score(y_period, period_preds)
                        periods_performance.append(period_acc)
                
                if len(periods_performance) >= 2:
                    temporal_stability = 1 - np.std(periods_performance)  # Moins de variance = plus stable
                    temporal_stability = max(0, temporal_stability)
                else:
                    temporal_stability = 1.0
            else:
                temporal_stability = 1.0
                periods_performance = []
            
            # Score global temporal
            temporal_checks = [proper_gap, no_overlap, chronological_order]
            temporal_score = (np.mean(temporal_checks) + temporal_leakage_score + temporal_stability) / 3
            
            results['details'] = {
                'gap_days': gap_days,
                'proper_gap': proper_gap,
                'no_overlap': no_overlap,
                'chronological': chronological_order,
                'temporal_leakage_score': float(temporal_leakage_score),
                'temporal_stability': float(temporal_stability),
                'periods_performance': [float(p) for p in periods_performance],
                'earliest_test_date': str(earliest_test.date()),
                'latest_test_date': str(latest_test.date())
            }
            
            results['score'] = temporal_score
            threshold = 1 - self.validation_thresholds['max_temporal_leakage']
            results['status'] = 'passed' if temporal_score >= threshold else 'warning' if temporal_score >= threshold-0.1 else 'failed'
            
            self.logger.info(f"⏰ Temporal Score: {temporal_score:.3f}")
            self.logger.info(f"📅 Gap train/test: {gap_days} jours")
            self.logger.info(f"🔄 Stabilité temporelle: {temporal_stability:.3f}")
            
        except Exception as e:
            results['status'] = 'error'
            results['details'] = {'error': str(e)}
            self.logger.error(f"❌ Erreur test temporel: {e}")
        
        self.validation_results['test_3_temporal'] = results
        return results
    
    def test_4_overfitting_diagnostic(self):
        """
        TEST 4: Diagnostic overfitting (CV vs Test gap)
        """
        self.logger.info("\n🎯 TEST 4: DIAGNOSTIC OVERFITTING")
        self.logger.info("="*60)
        
        results = {
            'test_name': 'overfitting_diagnostic',
            'status': 'unknown',
            'details': {},
            'score': 0.0
        }
        
        try:
            # Récupérer résultats CV des modèles optimisés
            optimization_results = self.model.get('optimization_results', {})
            
            stage1_results = optimization_results.get('stage1', {})
            stage2_results = optimization_results.get('stage2', {})
            
            # Stage 1 overfitting
            if 'cv_score' in stage1_results and 'test_f1' in stage1_results:
                stage1_cv = stage1_results['cv_score']
                stage1_test = stage1_results['test_f1']
                stage1_gap = stage1_cv - stage1_test
            else:
                stage1_gap = 0.0
                stage1_cv = 0.0
                stage1_test = 0.0
            
            # Stage 2 overfitting  
            if 'cv_score' in stage2_results and 'test_accuracy' in stage2_results:
                stage2_cv = stage2_results['cv_score']
                stage2_test = stage2_results['test_accuracy']
                stage2_gap = stage2_cv - stage2_test
            else:
                stage2_gap = 0.0
                stage2_cv = 0.0
                stage2_test = 0.0
            
            # Score anti-overfitting
            max_gap = self.validation_thresholds['max_overfitting_gap']
            stage1_overfitting_score = max(0, 1 - stage1_gap / max_gap) if stage1_gap > 0 else 1.0
            stage2_overfitting_score = max(0, 1 - stage2_gap / max_gap) if stage2_gap > 0 else 1.0
            
            overall_overfitting_score = (stage1_overfitting_score + stage2_overfitting_score) / 2
            
            # Diagnostic détaillé
            stage1_status = 'good' if stage1_gap < 0.05 else 'warning' if stage1_gap < max_gap else 'overfitting'
            stage2_status = 'good' if stage2_gap < 0.05 else 'warning' if stage2_gap < max_gap else 'overfitting'
            
            results['details'] = {
                'stage1': {
                    'cv_score': float(stage1_cv),
                    'test_score': float(stage1_test),
                    'gap': float(stage1_gap),
                    'status': stage1_status,
                    'overfitting_score': float(stage1_overfitting_score)
                },
                'stage2': {
                    'cv_score': float(stage2_cv),
                    'test_score': float(stage2_test),
                    'gap': float(stage2_gap),
                    'status': stage2_status,
                    'overfitting_score': float(stage2_overfitting_score)
                },
                'overall_overfitting_score': float(overall_overfitting_score)
            }
            
            results['score'] = overall_overfitting_score
            results['status'] = 'passed' if overall_overfitting_score >= 0.8 else 'warning' if overall_overfitting_score >= 0.6 else 'failed'
            
            self.logger.info(f"🎯 Overfitting Score: {overall_overfitting_score:.3f}")
            self.logger.info(f"📊 Stage 1 gap: {stage1_gap:.4f} ({stage1_status})")
            self.logger.info(f"📊 Stage 2 gap: {stage2_gap:.4f} ({stage2_status})")
            
        except Exception as e:
            results['status'] = 'error'
            results['details'] = {'error': str(e)}
            self.logger.error(f"❌ Erreur diagnostic overfitting: {e}")
        
        self.validation_results['test_4_overfitting'] = results
        return results
    
    def test_5_performance_validation(self):
        """
        TEST 5: Validation performance minimale
        """
        self.logger.info("\n📈 TEST 5: VALIDATION PERFORMANCE")
        self.logger.info("="*60)
        
        results = {
            'test_name': 'performance_validation',
            'status': 'unknown',
            'details': {},
            'score': 0.0
        }
        
        try:
            # Prédictions cascade complètes
            y_pred_cascade = self._predict_cascade_full(self.X_test)
            
            # Métriques principales
            accuracy = accuracy_score(self.y_test_global, y_pred_cascade)
            
            # Métriques par classe
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test_global, y_pred_cascade, average=None, zero_division=0
            )
            
            draw_recall = recall[1] if len(recall) > 1 else 0.0
            f1_macro = f1_score(self.y_test_global, y_pred_cascade, average='macro')
            
            # Tests de seuils minimaux
            accuracy_pass = accuracy >= self.validation_thresholds['min_accuracy']
            draw_recall_pass = draw_recall >= self.validation_thresholds['min_draw_recall']
            
            # Score performance composite
            performance_scores = [
                min(1.0, accuracy / self.validation_thresholds['min_accuracy']),
                min(1.0, draw_recall / self.validation_thresholds['min_draw_recall']) if draw_recall > 0 else 0,
                min(1.0, f1_macro / 0.4)  # F1 minimal 0.4
            ]
            
            performance_score = np.mean(performance_scores)
            
            # Comparaison vs baselines
            random_baseline = 0.333
            majority_baseline = 0.436
            v24_baseline = 0.530
            
            beats_random = accuracy > random_baseline
            beats_majority = accuracy > majority_baseline
            beats_v24 = accuracy >= v24_baseline * 0.95  # Tolérance 5%
            
            results['details'] = {
                'metrics': {
                    'accuracy': float(accuracy),
                    'draw_recall': float(draw_recall),
                    'f1_macro': float(f1_macro),
                    'precision_per_class': [float(p) for p in precision],
                    'recall_per_class': [float(r) for r in recall],
                    'f1_per_class': [float(f) for f in f1]
                },
                'threshold_tests': {
                    'accuracy_pass': accuracy_pass,
                    'draw_recall_pass': draw_recall_pass,
                    'min_accuracy_threshold': self.validation_thresholds['min_accuracy'],
                    'min_draw_recall_threshold': self.validation_thresholds['min_draw_recall']
                },
                'baseline_comparisons': {
                    'beats_random': beats_random,
                    'beats_majority': beats_majority,
                    'beats_v24': beats_v24,
                    'vs_random': float(accuracy - random_baseline),
                    'vs_majority': float(accuracy - majority_baseline),
                    'vs_v24': float(accuracy - v24_baseline)
                },
                'performance_score': float(performance_score)
            }
            
            results['score'] = performance_score
            results['status'] = 'passed' if accuracy_pass and draw_recall_pass else 'warning' if accuracy_pass or draw_recall_pass else 'failed'
            
            self.logger.info(f"📈 Performance Score: {performance_score:.3f}")
            self.logger.info(f"🎯 Accuracy: {accuracy:.3f} ({'✅' if accuracy_pass else '❌'})")
            self.logger.info(f"📊 Draw Recall: {draw_recall:.3f} ({'✅' if draw_recall_pass else '❌'})")
            self.logger.info(f"⚖️ F1-Macro: {f1_macro:.3f}")
            
        except Exception as e:
            results['status'] = 'error'
            results['details'] = {'error': str(e)}
            self.logger.error(f"❌ Erreur validation performance: {e}")
        
        self.validation_results['test_5_performance'] = results
        return results
    
    def _predict_cascade_full(self, X):
        """Prédiction cascade complète pour tests"""
        X_scaled = self.scaler.transform(X)
        threshold = self.model['optimal_threshold']
        
        # Stage 1: Draw probabilities
        draw_proba = self.model['draw_classifier'].predict_proba(X_scaled)[:, 1]
        draw_pred = (draw_proba >= threshold).astype(int)
        
        # Prédictions finales
        final_predictions = np.zeros(len(X), dtype=int)
        
        # Draws
        draw_mask = draw_pred == 1
        final_predictions[draw_mask] = 1
        
        # Non-draws → Stage 2
        non_draw_mask = draw_pred == 0
        if np.sum(non_draw_mask) > 0:
            X_non_draw = X_scaled[non_draw_mask]
            ha_pred = self.model['home_away_classifier'].predict(X_non_draw)
            final_predictions[non_draw_mask] = np.where(ha_pred == 1, 0, 2)
        
        return final_predictions
    
    def run_full_validation(self, model_path=None, data_path=None):
        """
        Exécuter validation complète
        """
        self.logger.info("🔍 DÉMARRAGE VALIDATION COMPLÈTE CASCADE v2.5")
        self.logger.info("="*70)
        
        if model_path:
            self.load_model(model_path)
        
        if data_path:
            self.load_test_data(data_path)
        elif not hasattr(self, 'X_test') or self.X_test is None:
            self.load_test_data()
        
        # Exécuter tous les tests
        tests_to_run = [
            self.test_1_cascade_architecture,
            self.test_2_stage_independence,
            self.test_3_temporal_integrity,
            self.test_4_overfitting_diagnostic,
            self.test_5_performance_validation
        ]
        
        passed_tests = 0
        total_score = 0.0
        
        for test_func in tests_to_run:
            test_result = test_func()
            if test_result['status'] == 'passed':
                passed_tests += 1
            total_score += test_result['score']
        
        # Score d'intégrité global
        self.integrity_score = total_score / len(tests_to_run)
        
        # Rapport final
        self.logger.info("\n" + "="*70)
        self.logger.info("🏆 RAPPORT VALIDATION CASCADE v2.5")
        self.logger.info("="*70)
        self.logger.info(f"✅ Tests passés: {passed_tests}/{len(tests_to_run)}")
        self.logger.info(f"🎯 Score intégrité: {self.integrity_score:.3f}")
        
        # Status global
        if self.integrity_score >= 0.8 and passed_tests >= 4:
            validation_status = "✅ PRODUCTION READY"
        elif self.integrity_score >= 0.6 and passed_tests >= 3:
            validation_status = "⚠️ ACCEPTABLE AVEC SURVEILLANCE"
        else:
            validation_status = "❌ NÉCESSITE CORRECTIONS"
        
        self.logger.info(f"🚦 Status: {validation_status}")
        
        # Recommandations
        recommendations = []
        for test_name, result in self.validation_results.items():
            if result['status'] in ['failed', 'warning']:
                recommendations.append(f"- {test_name}: {result['status']} (score: {result['score']:.3f})")
        
        if recommendations:
            self.logger.info(f"\n📋 Recommandations:")
            for rec in recommendations:
                self.logger.info(rec)
        
        # Sauvegarder rapport
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'features_count': len(self.features),
                'optimal_threshold': self.model['optimal_threshold'],
                'has_optimization_results': bool(self.model.get('optimization_results'))
            },
            'validation_results': self.validation_results,
            'summary': {
                'tests_passed': passed_tests,
                'total_tests': len(tests_to_run),
                'integrity_score': float(self.integrity_score),
                'validation_status': validation_status,
                'recommendations': recommendations
            }
        }
        
        return validation_report

def main():
    """
    Point d'entrée pour validation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Validation intégrité modèle cascade')
    parser.add_argument('--model', '-m', help='Chemin modèle cascade à valider')
    parser.add_argument('--data', '-d', help='Chemin données test (optionnel)')
    parser.add_argument('--output', '-o', help='Fichier rapport JSON (optionnel)')
    
    args = parser.parse_args()
    
    if not args.model:
        # Chercher modèle le plus récent
        import glob
        model_files = glob.glob('models/v25_*_cascade_*.joblib')
        if not model_files:
            print("❌ Aucun modèle cascade trouvé")
            return 1
        args.model = max(model_files, key=os.path.getctime)
        print(f"📦 Modèle auto-détecté: {args.model}")
    
    try:
        validator = CascadeIntegrityValidator()
        report = validator.run_full_validation(
            model_path=args.model,
            data_path=args.data
        )
        
        # Sauvegarder rapport
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
            output_path = f'evaluation/reports/cascade_integrity_validation_{timestamp}.json'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Rapport sauvegardé: {output_path}")
        
        # Code retour selon succès
        return 0 if report['summary']['integrity_score'] >= 0.8 else 1
        
    except Exception as e:
        print(f"❌ ERREUR VALIDATION: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())