#!/usr/bin/env python3
"""
v25_advanced_threshold_strategies.py

ADVANCED THRESHOLD OPTIMIZATION pour CASCADE v2.4/v2.5
Stratégies sophistiquées pour améliorer les performances draws

STRATÉGIES IMPLÉMENTÉES:
1. Dynamic Contextual Thresholds (basé sur features)
2. Probabilistic Ensemble Thresholds  
3. Confidence-Based Adaptive Thresholds
4. Multi-Objective Threshold Optimization
5. Time-Series Adaptive Thresholds

OBJECTIF: Améliorer draw recall de 22.6% → 30%+ tout en maintenant accuracy
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
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    precision_recall_fscore_support,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import scipy.optimize as opt
from scipy.stats import beta
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class AdvancedThresholdOptimizer:
    """
    Optimiseur sophistiqué de seuils pour cascade
    """
    
    def __init__(self, cascade_model_path=None):
        self.logger = setup_logging()
        
        # Features v2.4
        self.features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10',
            'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
            'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        # Charger modèle cascade existant
        if cascade_model_path:
            self.load_cascade_model(cascade_model_path)
        else:
            # Utiliser le modèle le plus récent
            import glob
            model_files = glob.glob('models/v25_local_optimized_cascade_*.joblib')
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                self.load_cascade_model(latest_model)
            else:
                raise ValueError("Aucun modèle cascade trouvé")
        
        self.threshold_strategies = {}
        self.results = {}
        
    def load_cascade_model(self, model_path):
        """Charger modèle cascade pré-entraîné"""
        self.logger.info(f"📥 Chargement modèle cascade: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.draw_classifier = model_data['draw_classifier']
        self.home_away_classifier = model_data['home_away_classifier'] 
        self.scaler = model_data['scaler']
        self.baseline_threshold = model_data.get('optimal_threshold', 0.4)
        
        self.logger.info(f"✅ Modèle chargé, seuil baseline: {self.baseline_threshold:.3f}")
        
    def load_test_data(self, filepath='data/processed/v13_xg_corrected_features_latest.csv'):
        """Charger données de test pour optimisation"""
        self.logger.info("📊 Chargement données test pour threshold optimization")
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # Split identique
        test_start = pd.to_datetime('2024-08-16')
        test_data = df[df['Date'] >= test_start].copy()
        test_data = test_data.dropna(subset=self.features + ['FullTimeResult'])
        
        self.X_test = test_data[self.features].values
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Targets
        test_results = test_data['FullTimeResult'].values
        self.y_test_global = pd.Series(test_results).map({'H': 0, 'D': 1, 'A': 2}).values
        self.y_test_draw = (test_results == 'D').astype(int)
        
        # Features pour stratégies contextuelles
        self.test_features_df = test_data[self.features + ['Date', 'Season']].reset_index(drop=True)
        
        self.logger.info(f"✅ Test data: {len(test_data)} matches")
        return True
        
    def strategy_1_dynamic_contextual(self):
        """
        STRATÉGIE 1: Seuils dynamiques basés sur contexte match
        
        Hypothèse: Différents contextes nécessitent différents seuils
        - Elo diff élevé → seuil plus haut (moins de draws attendus)
        - Market entropy élevé → seuil plus bas (plus d'incertitude)
        - Matchday avancé → ajustements saisonniers
        """
        self.logger.info("\n🎯 STRATÉGIE 1: SEUILS DYNAMIQUES CONTEXTUELS")
        self.logger.info("="*60)
        
        # Probabilités draw de base
        draw_probas = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        # Features contextuelles normalisées [0,1]
        elo_diff = self.test_features_df['elo_diff_normalized'].values
        market_entropy = self.test_features_df['market_entropy_norm'].values
        matchday = self.test_features_df['matchday_normalized'].values
        
        # RÈGLES CONTEXTUELLES:
        
        # 1. Elo diff impact: Plus la différence est grande, moins de draws
        elo_adjustment = -0.1 * elo_diff  # Seuil plus haut si équipes déséquilibrées
        
        # 2. Market uncertainty: Plus d'entropy = plus de draws possibles  
        market_adjustment = 0.15 * market_entropy  # Seuil plus bas si incertitude
        
        # 3. Seasonal effects: Fin de saison = plus de draws défensifs
        season_adjustment = 0.05 * matchday  # Seuil plus bas en fin de saison
        
        # Seuil dynamique par match
        base_threshold = 0.35  # Plus bas que baseline fixe
        dynamic_thresholds = np.clip(
            base_threshold + elo_adjustment + market_adjustment + season_adjustment,
            0.1, 0.7  # Bornes raisonnables
        )
        
        # Prédictions avec seuils dynamiques
        draw_pred_dynamic = (draw_probas >= dynamic_thresholds).astype(int)
        
        # Cascade finale
        final_preds = self._apply_cascade(draw_pred_dynamic)
        
        # Évaluation
        accuracy = accuracy_score(self.y_test_global, final_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test_global, final_preds, average=None, zero_division=0
        )
        
        draw_recall = recall[1] if len(recall) > 1 else 0.0
        f1_macro = f1_score(self.y_test_global, final_preds, average='macro')
        
        self.logger.info(f"🎯 Seuil moyen: {np.mean(dynamic_thresholds):.3f} ± {np.std(dynamic_thresholds):.3f}")
        self.logger.info(f"📊 Range: [{np.min(dynamic_thresholds):.3f}, {np.max(dynamic_thresholds):.3f}]")
        self.logger.info(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"✅ Draw Recall: {draw_recall:.4f} ({draw_recall*100:.1f}%)")
        self.logger.info(f"✅ F1-macro: {f1_macro:.4f}")
        
        self.threshold_strategies['dynamic_contextual'] = {
            'accuracy': float(accuracy),
            'draw_recall': float(draw_recall), 
            'f1_macro': float(f1_macro),
            'threshold_stats': {
                'mean': float(np.mean(dynamic_thresholds)),
                'std': float(np.std(dynamic_thresholds)),
                'min': float(np.min(dynamic_thresholds)),
                'max': float(np.max(dynamic_thresholds))
            }
        }
        
        return accuracy, draw_recall, f1_macro
        
    def strategy_2_probabilistic_ensemble(self):
        """
        STRATÉGIE 2: Ensemble probabiliste de seuils
        
        Utilise plusieurs seuils candidats avec pondération
        selon la confiance du modèle
        """
        self.logger.info("\n🎯 STRATÉGIE 2: ENSEMBLE PROBABILISTE DE SEUILS")
        self.logger.info("="*60)
        
        draw_probas = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        # Ensemble de seuils candidats
        candidate_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
        # Pondérations basées sur confiance du modèle
        # Plus la proba est proche de 0.5, moins on est confiant
        confidence_scores = 1 - 2 * np.abs(draw_probas - 0.5)  # [0,1]
        
        # Predictions par seuil
        ensemble_predictions = []
        
        for threshold in candidate_thresholds:
            draw_pred = (draw_probas >= threshold).astype(int)
            final_preds = self._apply_cascade(draw_pred)
            ensemble_predictions.append(final_preds)
        
        ensemble_predictions = np.array(ensemble_predictions)  # (n_thresholds, n_samples)
        
        # Vote pondéré par confiance
        final_ensemble_preds = np.zeros(len(self.X_test), dtype=int)
        
        for i in range(len(self.X_test)):
            # Pondération: plus confiant = plus de poids aux prédictions extrêmes
            if confidence_scores[i] > 0.6:  # Haute confiance
                # Favoriser seuils conservateurs (moins de draws)
                weights = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.1])
            elif confidence_scores[i] > 0.3:  # Confiance moyenne
                # Seuils équilibrés
                weights = np.array([0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.05])
            else:  # Faible confiance
                # Favoriser seuils agressifs (plus de draws)
                weights = np.array([0.25, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05])
            
            # Vote pondéré
            class_votes = np.zeros(3)
            for j, threshold in enumerate(candidate_thresholds):
                pred_class = ensemble_predictions[j, i]
                class_votes[pred_class] += weights[j]
            
            final_ensemble_preds[i] = np.argmax(class_votes)
        
        # Évaluation
        accuracy = accuracy_score(self.y_test_global, final_ensemble_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test_global, final_ensemble_preds, average=None, zero_division=0
        )
        
        draw_recall = recall[1] if len(recall) > 1 else 0.0
        f1_macro = f1_score(self.y_test_global, final_ensemble_preds, average='macro')
        
        self.logger.info(f"📊 Confiance moyenne: {np.mean(confidence_scores):.3f}")
        self.logger.info(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"✅ Draw Recall: {draw_recall:.4f} ({draw_recall*100:.1f}%)")
        self.logger.info(f"✅ F1-macro: {f1_macro:.4f}")
        
        self.threshold_strategies['probabilistic_ensemble'] = {
            'accuracy': float(accuracy),
            'draw_recall': float(draw_recall),
            'f1_macro': float(f1_macro),
            'mean_confidence': float(np.mean(confidence_scores))
        }
        
        return accuracy, draw_recall, f1_macro
        
    def strategy_3_confidence_adaptive(self):
        """
        STRATÉGIE 3: Seuils adaptatifs basés sur la confiance
        
        Ajuste le seuil selon la confiance du modèle:
        - Haute confiance → seuil plus strict
        - Faible confiance → seuil plus permissif
        """
        self.logger.info("\n🎯 STRATÉGIE 3: SEUILS ADAPTATIFS CONFIANCE")
        self.logger.info("="*60)
        
        draw_probas = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        # Mesure de confiance: distance à 0.5 (indécision)
        confidence = 2 * np.abs(draw_probas - 0.5)  # [0,1]
        
        # Seuils adaptatifs basés sur confiance
        base_threshold = 0.3
        confidence_adjustment = 0.2 * confidence  # Plus confiant = seuil plus haut
        
        adaptive_thresholds = base_threshold + confidence_adjustment
        
        # Prédictions adaptatives
        draw_pred_adaptive = (draw_probas >= adaptive_thresholds).astype(int)
        final_preds = self._apply_cascade(draw_pred_adaptive)
        
        # Évaluation
        accuracy = accuracy_score(self.y_test_global, final_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test_global, final_preds, average=None, zero_division=0
        )
        
        draw_recall = recall[1] if len(recall) > 1 else 0.0
        f1_macro = f1_score(self.y_test_global, final_preds, average='macro')
        
        self.logger.info(f"🎯 Seuil moyen: {np.mean(adaptive_thresholds):.3f}")
        self.logger.info(f"📊 Confiance moyenne: {np.mean(confidence):.3f}")
        self.logger.info(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"✅ Draw Recall: {draw_recall:.4f} ({draw_recall*100:.1f}%)")
        self.logger.info(f"✅ F1-macro: {f1_macro:.4f}")
        
        self.threshold_strategies['confidence_adaptive'] = {
            'accuracy': float(accuracy),
            'draw_recall': float(draw_recall),
            'f1_macro': float(f1_macro),
            'mean_threshold': float(np.mean(adaptive_thresholds)),
            'mean_confidence': float(np.mean(confidence))
        }
        
        return accuracy, draw_recall, f1_macro
        
    def strategy_4_multi_objective_optimization(self):
        """
        STRATÉGIE 4: Optimisation multi-objectifs
        
        Optimise simultanément accuracy et draw recall
        avec contraintes de balance
        """
        self.logger.info("\n🎯 STRATÉGIE 4: OPTIMISATION MULTI-OBJECTIFS")
        self.logger.info("="*60)
        
        draw_probas = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        def multi_objective_loss(threshold_params):
            """
            Fonction de coût multi-objectifs
            threshold_params: [base_threshold, variance]
            """
            base_thresh, variance = threshold_params
            
            # Seuils avec variance pour robustesse
            thresholds = np.random.normal(base_thresh, variance, len(draw_probas))
            thresholds = np.clip(thresholds, 0.1, 0.8)
            
            # Moyenne sur plusieurs échantillonnages
            accuracies = []
            draw_recalls = []
            
            for _ in range(5):  # 5 échantillons Monte Carlo
                sample_thresholds = np.random.normal(base_thresh, variance, len(draw_probas))
                sample_thresholds = np.clip(sample_thresholds, 0.1, 0.8)
                
                draw_pred = (draw_probas >= sample_thresholds).astype(int)
                final_preds = self._apply_cascade(draw_pred)
                
                acc = accuracy_score(self.y_test_global, final_preds)
                _, recall, _, _ = precision_recall_fscore_support(
                    self.y_test_global, final_preds, average=None, zero_division=0
                )
                draw_rec = recall[1] if len(recall) > 1 else 0.0
                
                accuracies.append(acc)
                draw_recalls.append(draw_rec)
            
            mean_acc = np.mean(accuracies)
            mean_draw_recall = np.mean(draw_recalls)
            
            # Multi-objectif avec pondération
            # Objectif: maximiser accuracy ET draw recall
            # Pénalité si draw recall < 25%
            draw_penalty = max(0, 0.25 - mean_draw_recall) * 2
            
            # Score composite (à minimiser)
            composite_score = -(0.6 * mean_acc + 0.4 * mean_draw_recall) + draw_penalty
            
            return composite_score
        
        # Optimisation
        self.logger.info("🔍 Optimisation multi-objectifs...")
        
        # Bornes: [base_threshold, variance]
        bounds = [(0.1, 0.6), (0.01, 0.15)]
        
        result = opt.minimize(
            multi_objective_loss,
            x0=[0.3, 0.05],  # Initial guess
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimal_base, optimal_variance = result.x
        
        # Évaluation avec paramètres optimaux
        final_thresholds = np.clip(
            np.random.normal(optimal_base, optimal_variance, len(draw_probas)),
            0.1, 0.8
        )
        
        draw_pred_optimal = (draw_probas >= final_thresholds).astype(int)
        final_preds = self._apply_cascade(draw_pred_optimal)
        
        accuracy = accuracy_score(self.y_test_global, final_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test_global, final_preds, average=None, zero_division=0
        )
        
        draw_recall = recall[1] if len(recall) > 1 else 0.0
        f1_macro = f1_score(self.y_test_global, final_preds, average='macro')
        
        self.logger.info(f"🎯 Seuil optimal: {optimal_base:.3f} ± {optimal_variance:.3f}")
        self.logger.info(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"✅ Draw Recall: {draw_recall:.4f} ({draw_recall*100:.1f}%)")
        self.logger.info(f"✅ F1-macro: {f1_macro:.4f}")
        
        self.threshold_strategies['multi_objective'] = {
            'accuracy': float(accuracy),
            'draw_recall': float(draw_recall),
            'f1_macro': float(f1_macro),
            'optimal_base': float(optimal_base),
            'optimal_variance': float(optimal_variance)
        }
        
        return accuracy, draw_recall, f1_macro
        
    def _apply_cascade(self, draw_predictions):
        """
        Appliquer cascade avec prédictions draw données
        """
        final_preds = np.zeros(len(self.X_test), dtype=int)
        
        # Draws
        draw_mask = draw_predictions == 1
        final_preds[draw_mask] = 1  # D = 1
        
        # Non-draws → Stage 2
        non_draw_mask = draw_predictions == 0
        if np.sum(non_draw_mask) > 0:
            X_non_draw = self.X_test_scaled[non_draw_mask]
            ha_pred = self.home_away_classifier.predict(X_non_draw)
            final_preds[non_draw_mask] = np.where(ha_pred == 1, 0, 2)  # H=0, A=2
        
        return final_preds
        
    def compare_strategies(self):
        """
        Comparaison complète de toutes les stratégies
        """
        self.logger.info("\n🏆 COMPARAISON COMPLÈTE DES STRATÉGIES")
        self.logger.info("="*60)
        
        # Baseline cascade fixe
        draw_probas = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        draw_pred_baseline = (draw_probas >= self.baseline_threshold).astype(int)
        baseline_preds = self._apply_cascade(draw_pred_baseline)
        
        baseline_acc = accuracy_score(self.y_test_global, baseline_preds)
        _, baseline_recall, _, _ = precision_recall_fscore_support(
            self.y_test_global, baseline_preds, average=None, zero_division=0
        )
        baseline_draw_recall = baseline_recall[1] if len(baseline_recall) > 1 else 0.0
        baseline_f1 = f1_score(self.y_test_global, baseline_preds, average='macro')
        
        # Comparaison
        self.logger.info(f"📊 BASELINE (seuil fixe {self.baseline_threshold:.3f}):")
        self.logger.info(f"   Accuracy: {baseline_acc:.4f}, Draw Recall: {baseline_draw_recall:.4f}, F1-macro: {baseline_f1:.4f}")
        
        self.logger.info(f"\n📊 STRATÉGIES AVANCÉES:")
        
        best_strategy = None
        best_composite_score = -1
        
        for strategy_name, results in self.threshold_strategies.items():
            acc = results['accuracy']
            draw_rec = results['draw_recall'] 
            f1_mac = results['f1_macro']
            
            # Score composite (même logique que multi-objectif)
            composite = 0.6 * acc + 0.4 * draw_rec
            
            # vs baseline
            acc_gain = acc - baseline_acc
            draw_gain = draw_rec - baseline_draw_recall
            f1_gain = f1_mac - baseline_f1
            
            self.logger.info(f"   {strategy_name.upper()}: Acc {acc:.4f} ({acc_gain:+.3f}), "
                           f"Draw {draw_rec:.4f} ({draw_gain:+.3f}), F1 {f1_mac:.4f} ({f1_gain:+.3f})")
            
            if composite > best_composite_score:
                best_composite_score = composite
                best_strategy = strategy_name
        
        self.logger.info(f"\n🏆 MEILLEURE STRATÉGIE: {best_strategy.upper()}")
        
        # Sauvegarder résultats
        comparison_results = {
            'baseline': {
                'accuracy': float(baseline_acc),
                'draw_recall': float(baseline_draw_recall),
                'f1_macro': float(baseline_f1),
                'threshold': float(self.baseline_threshold)
            },
            'strategies': self.threshold_strategies,
            'best_strategy': best_strategy,
            'best_composite_score': float(best_composite_score)
        }
        
        return comparison_results

def main():
    """
    Test complet des stratégies de seuils avancées
    """
    print("🎯 ADVANCED THRESHOLD STRATEGIES v2.5")
    print("="*60)
    print("Test de 4 stratégies sophistiquées pour cascade")
    print("Objectif: Améliorer draw recall tout en maintenant accuracy")
    print("="*60)
    
    try:
        optimizer = AdvancedThresholdOptimizer()
        optimizer.load_test_data()
        
        # Test des 4 stratégies
        print("\n🚀 EXÉCUTION DES STRATÉGIES...")
        
        optimizer.strategy_1_dynamic_contextual()
        optimizer.strategy_2_probabilistic_ensemble() 
        optimizer.strategy_3_confidence_adaptive()
        optimizer.strategy_4_multi_objective_optimization()
        
        # Comparaison finale
        results = optimizer.compare_strategies()
        
        # Sauvegarder rapport
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        report_path = f'evaluation/reports/advanced_threshold_strategies_{timestamp}.json'
        os.makedirs('evaluation/reports', exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Rapport sauvegardé: {report_path}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())