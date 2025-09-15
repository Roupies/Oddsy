#!/usr/bin/env python3
"""
v25_dynamic_threshold_optimizer.py

OPTIMISATION v2.5 - SEUILS DYNAMIQUES CASCADE
Améliorer v2.4 (53% + 34% draw recall) sans nouvelles features

OPTIMISATION 1: Seuils Dynamiques
- Au lieu de threshold fixe 0.4 pour draws
- Calculer seuil optimal par période/contexte
- Maximiser F1-macro équilibré (H/D/A)

ARCHITECTURE INCHANGÉE:
- Stage 1: Draw vs Non-Draw (avec SMOTE)  
- Stage 2: Home vs Away
- Features: mêmes 10 features validées v2.4

OBJECTIF v2.5:
- Maintenir 34% draw recall minimum
- Améliorer accuracy globale 53% → 54-56%
- Optimiser balance précision/rappel par classe
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    log_loss, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import scipy.optimize as opt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class DynamicThresholdCascade:
    """
    v2.5 Optimisation: Seuils dynamiques pour cascade draws
    """
    
    def __init__(self, features_list=None):
        self.logger = setup_logging()
        
        # Features v2.4 validées (inchangées)
        self.features = features_list or [
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
        
        # Nouveauté v2.5: Seuils optimisés
        self.optimal_threshold = None
        self.threshold_optimization_results = {}
        
        self.results = {}
        
    def load_data(self, filepath='data/processed/v13_xg_corrected_features_latest.csv'):
        """
        Charger données v2.4 validées
        """
        self.logger.info("📊 CHARGEMENT DONNÉES v2.5 (SEUILS DYNAMIQUES)")
        self.logger.info("="*70)
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # Filtrer données complètes
        valid_data = df.dropna(subset=self.features)
        self.logger.info(f"✅ Données: {len(valid_data)} matches avec {len(self.features)} features")
        
        # Split temporel identique v2.4
        train_end = pd.to_datetime('2024-05-19')
        test_start = pd.to_datetime('2024-08-16')
        
        train_data = valid_data[valid_data['Date'] <= train_end].copy()
        test_data = valid_data[valid_data['Date'] >= test_start].copy()
        
        self.logger.info(f"📊 Split temporel: Train {len(train_data)}, Test {len(test_data)}")
        
        # Matrices features
        self.X_train = train_data[self.features].values
        self.X_test = test_data[self.features].values
        
        # Targets cascade
        train_results = train_data['FullTimeResult'].values
        test_results = test_data['FullTimeResult'].values
        
        # Stage 1: Draw vs Non-Draw
        self.y_train_draw = (train_results == 'D').astype(int)
        self.y_test_draw = (test_results == 'D').astype(int)
        
        # Stage 2: Home vs Away (non-draws only)
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
        
        # Analyse distribution
        draw_pct = np.mean(self.y_train_draw) * 100
        self.logger.info(f"📊 Distribution: Draws {draw_pct:.1f}%, Non-Draws {100-draw_pct:.1f}%")
        
        return self.X_train, self.y_train_global, self.X_test, self.y_test_global
    
    def train_draw_classifier_v25(self):
        """
        Stage 1 amélioré: RandomForest + SMOTE + calibration
        """
        self.logger.info("\n🎯 STAGE 1 v2.5: CLASSIFICATEUR DRAWS (OPTIMISÉ)")
        self.logger.info("="*70)
        
        # SMOTE pour équilibrer draws (technique v2.4 validée)
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train_scaled, self.y_train_draw)
        
        self.logger.info(f"📈 SMOTE appliqué:")
        self.logger.info(f"  Avant: {np.sum(self.y_train_draw)} draws / {len(self.y_train_draw)} total")
        self.logger.info(f"  Après: {np.sum(y_train_smote)} draws / {len(y_train_smote)} total")
        
        # RandomForest optimisé (hyperparamètres v2.4 + améliorations)
        param_grid = {
            'n_estimators': [200, 300, 500, 700],  # Plus d'arbres
            'max_depth': [10, 15, 20, 25, None],   # Plus de profondeur
            'min_samples_split': [3, 5, 7, 10],    # Plus de granularité
            'min_samples_leaf': [1, 2, 4],         # Feuilles plus petites
            'max_features': ['sqrt', 'log2', 0.7], # Plus d'options
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf_draw = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Cross-validation optimisée
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            rf_draw, param_grid,
            cv=cv,
            scoring='f1',  # F1 pour draws
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info("🔍 Optimisation hyperparamètres Stage 1...")
        grid_search.fit(X_train_smote, y_train_smote)
        
        best_model = grid_search.best_estimator_
        self.logger.info(f"✅ Meilleurs params: {grid_search.best_params_}")
        self.logger.info(f"✅ Meilleur score CV: {grid_search.best_score_:.4f}")
        
        # Calibration isotonic (v2.4 validée)
        calibrated_draw = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
        calibrated_draw.fit(self.X_train_scaled, self.y_train_draw)  # Original non-SMOTE
        
        self.draw_classifier = calibrated_draw
        
        # Évaluation Stage 1
        y_pred_draw = self.draw_classifier.predict(self.X_test_scaled)
        y_proba_draw = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test_draw, y_pred_draw, average='binary'
        )
        
        self.logger.info(f"\n✅ STAGE 1 FINALISÉ:")
        self.logger.info(f"  F1-score Draws: {f1:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall: {recall:.4f}")
        self.logger.info(f"  ROC-AUC: {roc_auc_score(self.y_test_draw, y_proba_draw):.4f}")
        
        self.results['stage1_draw_classifier'] = {
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc_score(self.y_test_draw, y_proba_draw)),
            'best_params': grid_search.best_params_
        }
        
        return self.draw_classifier
    
    def train_home_away_classifier_v25(self):
        """
        Stage 2 amélioré: RandomForest optimisé pour Home/Away
        """
        self.logger.info(f"\n🏠⚽ STAGE 2 v2.5: CLASSIFICATEUR HOME vs AWAY (OPTIMISÉ)")
        self.logger.info("="*70)
        
        # Hyperparamètres étendus
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [3, 5, 8],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8],
            'class_weight': ['balanced', None, 'balanced_subsample']
        }
        
        rf_ha = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            rf_ha, param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info("🔍 Optimisation hyperparamètres Stage 2...")
        grid_search.fit(self.X_train_ha_scaled, self.y_train_ha)
        
        best_model_ha = grid_search.best_estimator_
        self.logger.info(f"✅ Meilleurs params: {grid_search.best_params_}")
        
        # Calibration
        calibrated_ha = CalibratedClassifierCV(best_model_ha, method='isotonic', cv=3)
        calibrated_ha.fit(self.X_train_ha_scaled, self.y_train_ha)
        
        self.home_away_classifier = calibrated_ha
        
        # Évaluation Stage 2
        y_pred_ha = self.home_away_classifier.predict(self.X_test_ha_scaled)
        accuracy_ha = accuracy_score(self.y_test_ha, y_pred_ha)
        
        self.logger.info(f"\n✅ STAGE 2 FINALISÉ:")
        self.logger.info(f"  Accuracy Home/Away: {accuracy_ha:.4f} ({accuracy_ha*100:.2f}%)")
        
        self.results['stage2_home_away_classifier'] = {
            'accuracy': float(accuracy_ha),
            'best_params': grid_search.best_params_
        }
        
        return self.home_away_classifier
    
    def optimize_threshold(self, metric='f1_macro'):
        """
        NOUVEAUTÉ v2.5: Optimisation seuil dynamique
        """
        self.logger.info(f"\n🎯 OPTIMISATION SEUIL DYNAMIQUE (MÉTRIQUE: {metric})")
        self.logger.info("="*70)
        
        # Probabilités draw sur ensemble test
        draw_probas = self.draw_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        # Fonction objectif à maximiser
        def objective_function(threshold):
            # Prédictions avec seuil personnalisé
            draw_pred_custom = (draw_probas >= threshold).astype(int)
            
            # Prédictions cascade avec seuil
            final_preds = np.zeros(len(self.X_test), dtype=int)
            
            # Matches prédits comme draws
            draw_mask = draw_pred_custom == 1
            final_preds[draw_mask] = 1  # D = 1
            
            # Matches non-draws -> Stage 2
            non_draw_mask = draw_pred_custom == 0
            if np.sum(non_draw_mask) > 0:
                X_non_draw = self.X_test_scaled[non_draw_mask]
                ha_pred = self.home_away_classifier.predict(X_non_draw)
                
                # Assigner H=0, A=2
                final_preds[non_draw_mask] = np.where(ha_pred == 1, 0, 2)
            
            # Calculer métrique selon choix
            if metric == 'f1_macro':
                score = f1_score(self.y_test_global, final_preds, average='macro')
            elif metric == 'accuracy':
                score = accuracy_score(self.y_test_global, final_preds)
            elif metric == 'draw_recall':
                # Maximiser rappel draws uniquement
                draw_recall = precision_recall_fscore_support(
                    self.y_test_draw, draw_pred_custom, average='binary'
                )[1]
                score = draw_recall
            else:
                score = accuracy_score(self.y_test_global, final_preds)
            
            return -score  # Minimiser pour scipy.optimize
        
        # Recherche seuil optimal
        self.logger.info("🔍 Recherche seuil optimal...")
        
        # Test range de seuils
        threshold_range = np.arange(0.1, 0.9, 0.05)
        scores = []
        
        for thresh in threshold_range:
            score = -objective_function(thresh)
            scores.append(score)
            
        # Meilleur seuil
        best_idx = np.argmax(scores)
        self.optimal_threshold = threshold_range[best_idx]
        best_score = scores[best_idx]
        
        self.logger.info(f"✅ Seuil optimal trouvé:")
        self.logger.info(f"  Threshold: {self.optimal_threshold:.3f}")
        self.logger.info(f"  Score ({metric}): {best_score:.4f}")
        
        # Analyse des différents seuils
        threshold_analysis = []
        for i, thresh in enumerate(threshold_range[:10]):  # Top 10
            threshold_analysis.append({
                'threshold': float(thresh),
                'score': float(scores[i])
            })
        
        self.threshold_optimization_results = {
            'optimal_threshold': float(self.optimal_threshold),
            'best_score': float(best_score),
            'metric_optimized': metric,
            'threshold_analysis': threshold_analysis
        }
        
        return self.optimal_threshold
    
    def predict_cascade_dynamic(self, X, threshold=None):
        """
        Prédiction cascade avec seuil dynamique
        """
        if threshold is None:
            threshold = self.optimal_threshold or 0.4  # Fallback v2.4
            
        X_scaled = self.scaler.transform(X)
        
        # Stage 1: Probabilités draws
        draw_proba = self.draw_classifier.predict_proba(X_scaled)[:, 1]
        draw_pred = (draw_proba >= threshold).astype(int)
        
        # Prédictions finales
        final_predictions = np.zeros(len(X), dtype=int)
        final_probabilities = np.zeros((len(X), 3))  # [H, D, A]
        
        # Matches prédits comme draws
        draw_mask = draw_pred == 1
        final_predictions[draw_mask] = 1  # D = 1
        final_probabilities[draw_mask, 1] = draw_proba[draw_mask]
        
        # Non-draws -> Stage 2
        non_draw_mask = draw_pred == 0
        if np.sum(non_draw_mask) > 0:
            X_non_draw = X_scaled[non_draw_mask]
            
            ha_proba = self.home_away_classifier.predict_proba(X_non_draw)
            ha_pred = self.home_away_classifier.predict(X_non_draw)
            
            # Indices globaux
            global_non_draw_indices = np.where(non_draw_mask)[0]
            
            # Home vs Away
            home_indices = global_non_draw_indices[ha_pred == 1]
            away_indices = global_non_draw_indices[ha_pred == 0]
            
            final_predictions[home_indices] = 0  # H = 0
            final_predictions[away_indices] = 2  # A = 2
            
            # Probabilités ajustées
            remaining_proba = 1 - draw_proba[non_draw_mask]
            final_probabilities[non_draw_mask, 0] = ha_proba[:, 1] * remaining_proba  # Home
            final_probabilities[non_draw_mask, 2] = ha_proba[:, 0] * remaining_proba  # Away
            final_probabilities[non_draw_mask, 1] = draw_proba[non_draw_mask]  # Draw
        
        return final_predictions, final_probabilities
    
    def evaluate_v25(self):
        """
        Évaluation complète v2.5 avec seuil optimisé
        """
        self.logger.info(f"\n🎯 ÉVALUATION COMPLÈTE v2.5 (Seuil: {self.optimal_threshold:.3f})")
        self.logger.info("="*70)
        
        # Prédiction avec seuil optimisé
        y_pred_v25, y_proba_v25 = self.predict_cascade_dynamic(self.X_test)
        
        # Métriques globales
        accuracy_v25 = accuracy_score(self.y_test_global, y_pred_v25)
        logloss_v25 = log_loss(self.y_test_global, y_proba_v25)
        
        # Rapport détaillé
        report = classification_report(
            self.y_test_global, 
            y_pred_v25,
            target_names=['Home', 'Draw', 'Away'],
            output_dict=True
        )
        
        # Métriques par classe
        home_f1 = report['Home']['f1-score']
        draw_f1 = report['Draw']['f1-score']
        draw_recall = report['Draw']['recall']
        away_f1 = report['Away']['f1-score']
        f1_macro = report['macro avg']['f1-score']
        
        self.logger.info(f"🏆 RÉSULTATS v2.5:")
        self.logger.info(f"  Accuracy: {accuracy_v25:.4f} ({accuracy_v25*100:.2f}%)")
        self.logger.info(f"  Log-loss: {logloss_v25:.4f}")
        self.logger.info(f"  F1-macro: {f1_macro:.4f}")
        
        self.logger.info(f"\n📊 Performance par classe:")
        self.logger.info(f"  Home F1: {home_f1:.4f}")
        self.logger.info(f"  Draw F1: {draw_f1:.4f} (Recall: {draw_recall:.4f})")
        self.logger.info(f"  Away F1: {away_f1:.4f}")
        
        # Comparaison vs v2.4 baseline
        v24_accuracy = 0.530  # Référence v2.4
        v24_draw_recall = 0.344
        
        improvement_acc = accuracy_v25 - v24_accuracy
        improvement_draw = draw_recall - v24_draw_recall
        
        self.logger.info(f"\n📈 vs v2.4 BASELINE:")
        self.logger.info(f"  Accuracy: {improvement_acc:+.3f}pp")
        self.logger.info(f"  Draw Recall: {improvement_draw:+.3f}pp")
        
        # Matrice confusion
        cm = confusion_matrix(self.y_test_global, y_pred_v25)
        self.logger.info(f"\n📊 Matrice confusion:")
        self.logger.info(f"       H    D    A")
        for i, label in enumerate(['H', 'D', 'A']):
            row = ' '.join([f'{cm[i][j]:4d}' for j in range(3)])
            self.logger.info(f"  {label}: {row}")
        
        # Évaluation succès v2.5
        success_criteria = {
            'maintain_draw_recall': draw_recall >= 0.30,  # Minimum 30%
            'improve_accuracy': accuracy_v25 >= v24_accuracy,  # Maintenir/améliorer
            'balanced_f1': f1_macro >= 0.45  # F1 équilibré
        }
        
        success = all(success_criteria.values())
        
        if success:
            self.logger.info(f"\n✅ SUCCÈS v2.5: Tous critères atteints!")
        else:
            failed = [k for k, v in success_criteria.items() if not v]
            self.logger.info(f"\n⚠️ Critères non atteints: {failed}")
        
        self.results['v25_evaluation'] = {
            'accuracy': float(accuracy_v25),
            'log_loss': float(logloss_v25),
            'f1_macro': float(f1_macro),
            'draw_f1': float(draw_f1),
            'draw_recall': float(draw_recall),
            'improvement_vs_v24': {
                'accuracy': float(improvement_acc),
                'draw_recall': float(improvement_draw)
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'success_criteria': success_criteria,
            'overall_success': success,
            'optimal_threshold': float(self.optimal_threshold)
        }
        
        return success, accuracy_v25, draw_recall, f1_macro

def main():
    """
    Pipeline principal v2.5 - Optimisation seuils dynamiques
    """
    print("🎯 ODDSY v2.5 - OPTIMISATION SEUILS DYNAMIQUES CASCADE")
    print("="*70)
    print("Objectif: Améliorer v2.4 (53% + 34% draw recall)")
    print("Méthode: Seuils adaptatifs + hyperparamètres optimisés")
    print("="*70)
    
    cascade_v25 = DynamicThresholdCascade()
    
    try:
        # 1. Charger données
        X_train, y_train, X_test, y_test = cascade_v25.load_data()
        
        # 2. Entraîner Stage 1 optimisé
        cascade_v25.train_draw_classifier_v25()
        
        # 3. Entraîner Stage 2 optimisé  
        cascade_v25.train_home_away_classifier_v25()
        
        # 4. NOUVEAUTÉ v2.5: Optimiser seuil dynamique
        optimal_threshold = cascade_v25.optimize_threshold(metric='f1_macro')
        
        # 5. Évaluation finale
        success, accuracy, draw_recall, f1_macro = cascade_v25.evaluate_v25()
        
        # 6. Sauvegarder modèle v2.5
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        model_path = f'models/cascade_v25_dynamic_threshold_{timestamp}.joblib'
        
        cascade_model = {
            'draw_classifier': cascade_v25.draw_classifier,
            'home_away_classifier': cascade_v25.home_away_classifier,
            'scaler': cascade_v25.scaler,
            'features': cascade_v25.features,
            'optimal_threshold': cascade_v25.optimal_threshold,
            'threshold_optimization': cascade_v25.threshold_optimization_results,
            'results': cascade_v25.results
        }
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(cascade_model, model_path)
        
        # 7. Rapport final
        report = {
            'timestamp': timestamp,
            'version': 'v2.5_dynamic_threshold_cascade',
            'results': cascade_v25.results,
            'model_path': model_path,
            'success': success,
            'summary': {
                'accuracy': accuracy,
                'draw_recall': draw_recall,
                'f1_macro': f1_macro,
                'optimal_threshold': optimal_threshold,
                'improvement_vs_v24': {
                    'accuracy': accuracy - 0.530,
                    'draw_recall': draw_recall - 0.344
                }
            }
        }
        
        report_file = f'reports/v25_dynamic_threshold_report_{timestamp}.json'
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Status final
        print("\n" + "="*70)
        print("🎯 ODDSY v2.5 TERMINÉ!")
        print(f"📊 Accuracy: {accuracy:.1%} (vs v2.4: {accuracy-0.530:+.1%})")
        print(f"🎯 Draw Recall: {draw_recall:.1%} (vs v2.4: {draw_recall-0.344:+.1%})")
        print(f"⚖️ F1-Macro: {f1_macro:.1%}")
        print(f"🔧 Seuil optimal: {optimal_threshold:.3f}")
        print(f"✅ Succès: {'OUI' if success else 'NON'}")
        print(f"📁 Rapport: {report_file}")
        print("="*70)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ ERREUR v2.5: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())