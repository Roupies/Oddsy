#!/usr/bin/env python3
"""
Audit v2.3 FINAL FIX - Correction du Mapping des Labels
=======================================================

PROBLÈME RÉSOLU: Le modèle v2.3 prédit des labels numériques [0,1,2] 
mais la ground truth utilise des labels string ['H','D','A'].

MAPPING CORRECT:
- 0 → 'H' (Home)
- 1 → 'D' (Draw)  
- 2 → 'A' (Away)

OBJECTIF: Obtenir la vraie performance du v2.3 original en rolling.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V23FinalAuditor:
    """
    Auditeur final avec mapping correct des labels
    """
    
    def __init__(self):
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        self.model_path = "models/v23_retrained_2025_09_11_154613.joblib"
        
        # Features dans l'ordre correct
        self.v23_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
            'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
            'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # MAPPING CORRECT: numeric → string
        self.label_mapping = {
            0: 'H',  # Home
            1: 'D',  # Draw
            2: 'A'   # Away
        }
        
        # Données
        self.dataset = None
        self.ground_truth = None
        self.model = None
        
    def load_data(self):
        """Charger les données"""
        
        print("🔧 AUDIT v2.3 FINAL - CORRECTION MAPPING LABELS")
        print("="*60)
        
        # Dataset
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        print(f"✅ Dataset: {len(self.dataset)} matches")
        
        # Ground truth
        self.ground_truth = pd.read_csv(self.ground_truth_path)
        self.ground_truth['Date'] = pd.to_datetime(self.ground_truth['Date'], dayfirst=True)
        print(f"✅ Ground truth: {len(self.ground_truth)} matches")
        
        # Modèle v2.3
        self.model = joblib.load(self.model_path)
        print(f"✅ Modèle v2.3: {type(self.model)}")
        
        return True
        
    def final_rolling_validation(self):
        """Rolling validation finale avec mapping correct"""
        
        print(f"\n🎯 ROLLING VALIDATION FINALE")
        print("-" * 40)
        
        # Données EPL 2025-26
        epl_2025_data = self.dataset[
            self.dataset['Season'] == '2025-2026'
        ].copy()
        
        print(f"📅 Données EPL 2025-26: {len(epl_2025_data)} matches")
        
        # Préparer features
        test_matches = epl_2025_data.head(len(self.ground_truth))
        X_test = test_matches[self.v23_features].fillna(0)
        
        print(f"🔧 Features préparées: {X_test.shape}")
        
        # PRÉDICTIONS AVEC MODÈLE v2.3
        y_pred_numeric = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # CONVERSION: numeric → string avec mapping correct
        y_pred_string = [self.label_mapping[pred] for pred in y_pred_numeric]
        
        print(f"🔄 Conversion des prédictions:")
        print(f"   Numeric: {list(y_pred_numeric)}")
        print(f"   String:  {y_pred_string}")
        
        # Ground truth
        y_true = self.ground_truth['FTR'].head(len(test_matches))
        
        print(f"📊 Comparaison:")
        print(f"   Prédictions: {len(y_pred_string)} matches")
        print(f"   Ground truth: {len(y_true)} matches")
        
        # MÉTRIQUES FINALES
        accuracy = accuracy_score(y_true, y_pred_string)
        
        print(f"\n🏆 RÉSULTATS FINAUX v2.3:")
        print(f"   Accuracy: {accuracy:.1%}")
        
        # Distribution
        pred_dist = pd.Series(y_pred_string).value_counts()
        true_dist = y_true.value_counts()
        
        print(f"\n📊 Distributions:")
        print(f"   Prédictions: H={pred_dist.get('H',0)}, D={pred_dist.get('D',0)}, A={pred_dist.get('A',0)}")
        print(f"   Réalité:     H={true_dist.get('H',0)}, D={true_dist.get('D',0)}, A={true_dist.get('A',0)}")
        
        # Rapport détaillé
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred_string, digits=3))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred_string, labels=['H', 'D', 'A'])
        print(f"\n📊 Matrice de Confusion:")
        print("     H   D   A")
        for i, true_label in enumerate(['H', 'D', 'A']):
            print(f"{true_label}: {cm[i][0]:3d} {cm[i][1]:3d} {cm[i][2]:3d}")
            
        # Analyse par match
        print(f"\n🔍 Analyse Match par Match:")
        for i in range(min(10, len(y_true))):  # Afficher 10 premiers
            true_val = y_true.iloc[i]
            pred_val = y_pred_string[i]
            correct = "✅" if true_val == pred_val else "❌"
            proba = y_proba[i]
            print(f"   Match {i+1}: Prédit={pred_val} | Réel={true_val} | P(H/D/A)=({proba[0]:.2f}/{proba[1]:.2f}/{proba[2]:.2f}) {correct}")
            
        # Probabilités moyennes par classe
        print(f"\n🎲 Probabilités Moyennes:")
        avg_proba = y_proba.mean(axis=0)
        print(f"   P(H): {avg_proba[0]:.3f}")
        print(f"   P(D): {avg_proba[1]:.3f}")
        print(f"   P(A): {avg_proba[2]:.3f}")
        
        return {
            'accuracy': accuracy,
            'predictions_numeric': y_pred_numeric,
            'predictions_string': y_pred_string,
            'true_labels': y_true,
            'probabilities': y_proba,
            'pred_distribution': pred_dist,
            'true_distribution': true_dist,
            'confusion_matrix': cm
        }
        
    def compare_with_targets(self, results):
        """Comparer avec les objectifs"""
        
        print(f"\n🎯 COMPARAISON AVEC OBJECTIFS")
        print("-" * 40)
        
        accuracy = results['accuracy']
        
        # Objectifs
        targets = {
            'baseline_naive': 0.333,      # 33.3% (random)
            'baseline_majority': 0.436,   # 43.6% (always home)
            'target_good': 0.50,          # 50% (good model)
            'target_excellent': 0.55      # 55% (excellent)
        }
        
        print(f"📊 Performance vs Objectifs:")
        for target_name, target_value in targets.items():
            diff = accuracy - target_value
            status = "✅" if diff >= 0 else "❌"
            print(f"   {target_name}: {target_value:.1%} | Écart: {diff:+.1%} {status}")
            
        # Verdict final
        if accuracy >= targets['target_excellent']:
            verdict = "🏆 EXCELLENT - Performance de pointe!"
        elif accuracy >= targets['target_good']:
            verdict = "✅ BON - Objectif principal atteint"
        elif accuracy >= targets['baseline_majority']:
            verdict = "🟡 ACCEPTABLE - Bat le baseline majority"
        elif accuracy >= targets['baseline_naive']:
            verdict = "🟠 MINIMAL - Bat le baseline random"
        else:
            verdict = "🔴 ÉCHEC - Sous-performance"
            
        print(f"\n{verdict}")
        print(f"🎯 Accuracy finale: {accuracy:.1%}")
        
        return {
            'accuracy': accuracy,
            'targets': targets,
            'verdict': verdict
        }
        
    def save_final_results(self, rolling_results, comparison):
        """Sauvegarder les résultats finaux"""
        
        print(f"\n💾 SAUVEGARDE RÉSULTATS FINAUX")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/audit_final")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prédictions détaillées
        pred_file = results_dir / f"v23_final_audit_{timestamp}.csv"
        
        pred_df = pd.DataFrame({
            'Match_ID': range(1, len(rolling_results['true_labels']) + 1),
            'True_Result': rolling_results['true_labels'],
            'Predicted_Result': rolling_results['predictions_string'],
            'Predicted_Numeric': rolling_results['predictions_numeric'],
            'Prob_H': rolling_results['probabilities'][:, 0],
            'Prob_D': rolling_results['probabilities'][:, 1],
            'Prob_A': rolling_results['probabilities'][:, 2],
            'Correct': rolling_results['true_labels'].values == rolling_results['predictions_string']
        })
        
        pred_df.to_csv(pred_file, index=False)
        print(f"✅ Prédictions détaillées: {pred_file}")
        
        # Rapport final
        report_file = results_dir / f"v23_final_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'model_type': 'v2.3 Original with correct label mapping',
            'fix_applied': 'Corrected numeric to string label mapping (0→H, 1→D, 2→A)',
            'final_accuracy': float(rolling_results['accuracy']),
            'verdict': comparison['verdict'],
            'feature_order_used': self.v23_features,
            'label_mapping_used': self.label_mapping,
            'performance_vs_targets': {k: float(v) for k, v in comparison['targets'].items()},
            'detailed_metrics': {
                'pred_distribution': {k: int(v) for k, v in rolling_results['pred_distribution'].items()},
                'true_distribution': {k: int(v) for k, v in rolling_results['true_distribution'].items()},
                'avg_probabilities': {
                    'H': float(rolling_results['probabilities'][:, 0].mean()),
                    'D': float(rolling_results['probabilities'][:, 1].mean()),
                    'A': float(rolling_results['probabilities'][:, 2].mean())
                }
            }
        }
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"✅ Rapport final: {report_file}")
        
        return pred_file, report_file
        
    def run_final_audit(self):
        """Audit final complet"""
        
        print("🏁 AUDIT FINAL v2.3 - RÉSOLUTION COMPLÈTE")
        print("="*60)
        
        # Charger données
        self.load_data()
        
        # Rolling validation finale
        rolling_results = self.final_rolling_validation()
        
        # Comparaison avec objectifs
        comparison = self.compare_with_targets(rolling_results)
        
        # Sauvegarder
        pred_file, report_file = self.save_final_results(rolling_results, comparison)
        
        # CONCLUSION FINALE
        print(f"\n🏆 AUDIT FINAL TERMINÉ!")
        print(f"📊 Performance v2.3 validée: {rolling_results['accuracy']:.1%}")
        print(f"🎯 {comparison['verdict']}")
        print(f"📄 Rapport complet: {report_file}")
        
        return {
            'final_accuracy': rolling_results['accuracy'],
            'verdict': comparison['verdict'],
            'files': {'predictions': pred_file, 'report': report_file},
            'rolling_results': rolling_results
        }

def main():
    """Fonction principale"""
    
    auditor = V23FinalAuditor()
    results = auditor.run_final_audit()
    
    return auditor, results

if __name__ == "__main__":
    auditor, results = main()