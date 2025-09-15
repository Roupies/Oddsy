#!/usr/bin/env python3
"""
Audit v2.3 avec ORDRE CORRECT des Features
==========================================

BUG CRITIQUE IDENTIFIÉ: L'ordre des features était incorrect dans tous nos
scripts de rolling validation, causant des performances désastreuses.

CORRECTION: Utiliser l'ordre EXACT attendu par le modèle v2.3 original.

OBJECTIF: Valider la vraie performance du v2.3 original en rolling.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V23CorrectOrderAuditor:
    """
    Auditeur avec ordre correct des features
    """
    
    def __init__(self):
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        self.model_path = "models/v23_retrained_2025_09_11_154613.joblib"
        
        # ORDRE CORRECT des features (inspecté depuis le modèle v2.3)
        self.v23_features_CORRECT_ORDER = [
            'form_diff_normalized',      # 1
            'elo_diff_normalized',       # 2  
            'h2h_score',                 # 3
            'matchday_normalized',       # 4
            'shots_diff_normalized',     # 5
            'corners_diff_normalized',   # 6
            'market_entropy_norm',       # 7
            'home_xg_eff_10',           # 8
            'away_goals_sum_5',         # 9
            'away_xg_eff_10'            # 10
        ]
        
        # Données
        self.dataset = None
        self.ground_truth = None
        self.model = None
        
    def load_all_data(self):
        """Charger toutes les données avec vérification d'ordre"""
        
        print("🔧 AUDIT v2.3 AVEC ORDRE CORRECT DES FEATURES")
        print("="*60)
        
        # Dataset
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        print(f"✅ Dataset: {len(self.dataset)} matches")
        
        # Ground truth
        self.ground_truth = pd.read_csv(self.ground_truth_path)
        self.ground_truth['Date'] = pd.to_datetime(self.ground_truth['Date'], dayfirst=True)
        print(f"✅ Ground truth: {len(self.ground_truth)} matches")
        
        # Modèle v2.3 original
        self.model = joblib.load(self.model_path)
        print(f"✅ Modèle v2.3 chargé: {type(self.model)}")
        
        # VERIFICATION CRITIQUE: ordre des features
        if hasattr(self.model, 'calibrated_classifiers_'):
            base_estimator = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, 'feature_names_in_'):
                model_features = base_estimator.feature_names_in_
                
                print(f"\n🔍 VÉRIFICATION ORDRE DES FEATURES:")
                print(f"   Modèle attend: {list(model_features)}")
                print(f"   On utilise:    {self.v23_features_CORRECT_ORDER}")
                
                # Vérifier que c'est identique
                if list(model_features) == self.v23_features_CORRECT_ORDER:
                    print(f"   ✅ ORDRE CORRECT!")
                else:
                    print(f"   ❌ ORDRE INCORRECT - Différences détectées")
                    for i, (model_f, our_f) in enumerate(zip(model_features, self.v23_features_CORRECT_ORDER)):
                        if model_f != our_f:
                            print(f"      Position {i+1}: Modèle='{model_f}' vs Notre='{our_f}'")
                    return False
        
        # Vérifier que toutes les features sont disponibles
        available_features = set(self.dataset.columns)
        missing_features = [f for f in self.v23_features_CORRECT_ORDER if f not in available_features]
        
        if missing_features:
            print(f"❌ Features manquantes: {missing_features}")
            return False
            
        print(f"✅ Toutes les features disponibles dans l'ordre correct")
        
        return True
        
    def prepare_features_correctly(self, data_subset):
        """Préparer les features dans l'ordre EXACT attendu par le modèle"""
        
        # Extraire features dans l'ordre correct
        X = data_subset[self.v23_features_CORRECT_ORDER].copy()
        
        # Gérer les valeurs manquantes
        X = X.fillna(X.median())
        
        return X
        
    def rolling_validation_correct_order(self):
        """Rolling validation avec ordre correct des features"""
        
        print(f"\n🔄 ROLLING VALIDATION - ORDRE FEATURES CORRECT")
        print("-" * 55)
        
        # Données EPL 2025-26 
        epl_2025_data = self.dataset[
            self.dataset['Season'] == '2025-2026'
        ].copy()
        
        print(f"📅 Données EPL 2025-26: {len(epl_2025_data)} matches disponibles")
        
        # Limiter aux matches de ground truth
        test_matches = epl_2025_data.head(len(self.ground_truth))
        print(f"🎯 Matches à tester: {len(test_matches)}")
        
        # Préparer features avec ORDRE CORRECT
        X_test = self.prepare_features_correctly(test_matches)
        
        print(f"🔧 Features préparées: {X_test.shape}")
        print(f"   Ordre utilisé: {list(X_test.columns)}")
        
        # Prédictions avec modèle v2.3 original
        print(f"🚀 Prédictions en cours...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Ground truth
        y_true = self.ground_truth['FTR'].head(len(test_matches))
        
        # Métriques
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n🎯 RÉSULTATS avec ORDRE CORRECT:")
        print(f"   Accuracy: {accuracy:.1%}")
        
        # Distribution des prédictions
        pred_dist = pd.Series(y_pred).value_counts()
        true_dist = y_true.value_counts()
        
        print(f"\n📊 Distributions:")
        print(f"   Prédictions: H={pred_dist.get('H',0)}, D={pred_dist.get('D',0)}, A={pred_dist.get('A',0)}")
        print(f"   Réalité:     H={true_dist.get('H',0)}, D={true_dist.get('D',0)}, A={true_dist.get('A',0)}")
        
        # Rapport détaillé
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, digits=3))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred, labels=['H', 'D', 'A'])
        print(f"\n📊 Matrice de Confusion:")
        print("     H   D   A")
        for i, true_label in enumerate(['H', 'D', 'A']):
            print(f"{true_label}: {cm[i][0]:3d} {cm[i][1]:3d} {cm[i][2]:3d}")
            
        # Analyse des probabilités
        print(f"\n🎲 Analyse des Probabilités:")
        avg_proba = y_proba.mean(axis=0)
        print(f"   Probabilités moyennes: P(H)={avg_proba[0]:.3f}, P(D)={avg_proba[1]:.3f}, P(A)={avg_proba[2]:.3f}")
        
        # Check si le modèle fait des prédictions équilibrées
        unique_predictions = len(pred_dist)
        print(f"   Classes prédites: {unique_predictions}/3")
        
        if unique_predictions == 1:
            print(f"   ⚠️  Modèle prédit une seule classe: {pred_dist.index[0]}")
        elif unique_predictions == 2:
            print(f"   ⚠️  Modèle prédit seulement 2 classes: {list(pred_dist.index)}")
        else:
            print(f"   ✅ Modèle prédit les 3 classes H/D/A")
            
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_true,
            'pred_distribution': pred_dist,
            'true_distribution': true_dist,
            'classes_predicted': unique_predictions
        }
        
    def comparison_with_previous_audits(self, current_results):
        """Comparer avec les audits précédents (ordre incorrect)"""
        
        print(f"\n📊 COMPARAISON AVEC AUDITS PRÉCÉDENTS")
        print("-" * 50)
        
        # Résultats précédents (ordre incorrect)
        previous_results = {
            'v23_original_incorrect_order': 0.483,  # 48.3%
            'v23_balanced_incorrect_order': 0.467,  # 46.7%  
            'v23_optimized_incorrect_order': 0.367  # 36.7%
        }
        
        current_accuracy = current_results['accuracy']
        
        print(f"🔍 Accuracy Comparison:")
        print(f"   v2.3 Original (ORDRE CORRECT):    {current_accuracy:.1%}")
        print(f"   v2.3 Original (ordre incorrect):  {previous_results['v23_original_incorrect_order']:.1%}")
        
        improvement = current_accuracy - previous_results['v23_original_incorrect_order']
        print(f"   Amélioration due à l'ordre:       {improvement:+.1%}")
        
        if improvement > 0.05:  # > 5pp d'amélioration
            print(f"   🚀 AMÉLIORATION MAJEURE grâce à la correction de l'ordre!")
        elif improvement > 0.02:  # > 2pp
            print(f"   ✅ Amélioration significative")
        elif improvement > 0:
            print(f"   🟡 Légère amélioration")
        else:
            print(f"   🔴 Pas d'amélioration - autres problèmes possibles")
            
        # Objectif de performance
        target_accuracy = 0.54  # 54%
        if current_accuracy >= target_accuracy:
            print(f"\n🎯 OBJECTIF ATTEINT! {current_accuracy:.1%} ≥ {target_accuracy:.1%}")
        else:
            gap = target_accuracy - current_accuracy
            print(f"\n🔄 Écart à l'objectif: -{gap:.1%} pour atteindre {target_accuracy:.1%}")
            
        return {
            'current_accuracy': current_accuracy,
            'previous_best': previous_results['v23_original_incorrect_order'],
            'improvement': improvement,
            'target_achieved': current_accuracy >= target_accuracy
        }
        
    def save_corrected_audit_results(self, rolling_results, comparison):
        """Sauvegarder les résultats de l'audit corrigé"""
        
        print(f"\n💾 SAUVEGARDE AUDIT CORRIGÉ")
        print("-" * 35)
        
        # Répertoire
        results_dir = Path("results/audit_corrected")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prédictions détaillées
        pred_file = results_dir / f"v23_correct_order_predictions_{timestamp}.csv"
        
        pred_df = pd.DataFrame({
            'True_Result': rolling_results['true_labels'],
            'Predicted_Result': rolling_results['predictions'],
            'Prob_H': rolling_results['probabilities'][:, 0],
            'Prob_D': rolling_results['probabilities'][:, 1],
            'Prob_A': rolling_results['probabilities'][:, 2],
            'Correct': rolling_results['true_labels'].values == rolling_results['predictions']
        })
        
        pred_df.to_csv(pred_file, index=False)
        print(f"✅ Prédictions: {pred_file}")
        
        # Résumé des résultats
        summary_file = results_dir / f"audit_summary_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'correction_applied': 'Fixed feature order to match v2.3 model expectations',
            'feature_order_used': self.v23_features_CORRECT_ORDER,
            'rolling_accuracy': float(rolling_results['accuracy']),
            'classes_predicted': int(rolling_results['classes_predicted']),
            'comparison_with_incorrect_order': {
                'improvement': float(comparison['improvement']),
                'target_achieved': bool(comparison['target_achieved'])
            },
            'pred_distribution': {k: int(v) for k, v in rolling_results['pred_distribution'].items()},
            'true_distribution': {k: int(v) for k, v in rolling_results['true_distribution'].items()}
        }
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"✅ Résumé: {summary_file}")
        
        return pred_file, summary_file
        
    def run_corrected_audit(self):
        """Exécuter l'audit complet avec ordre correct"""
        
        print("🔧 AUDIT v2.3 - CORRECTION ORDRE DES FEATURES")
        print("="*60)
        
        # Charger données
        if not self.load_all_data():
            print("❌ Échec du chargement des données")
            return None
            
        # Rolling validation avec ordre correct
        rolling_results = self.rolling_validation_correct_order()
        
        # Comparaison avec audits précédents
        comparison = self.comparison_with_previous_audits(rolling_results)
        
        # Sauvegarder
        pred_file, summary_file = self.save_corrected_audit_results(rolling_results, comparison)
        
        # Résumé final
        print(f"\n🏆 AUDIT CORRIGÉ TERMINÉ!")
        print(f"📊 Accuracy avec ordre correct: {rolling_results['accuracy']:.1%}")
        print(f"📈 Amélioration vs ordre incorrect: {comparison['improvement']:+.1%}")
        
        if comparison['target_achieved']:
            print(f"🎯 OBJECTIF 54% ATTEINT!")
        else:
            print(f"🔄 Objectif 54% pas encore atteint")
            
        return {
            'rolling_results': rolling_results,
            'comparison': comparison,
            'files': {'predictions': pred_file, 'summary': summary_file}
        }

def main():
    """Fonction principale"""
    
    auditor = V23CorrectOrderAuditor()
    results = auditor.run_corrected_audit()
    
    return auditor, results

if __name__ == "__main__":
    auditor, results = main()