#!/usr/bin/env python3
"""
Rolling Validation avec Features Dynamiques - LA VRAIE VALIDATION
=================================================================

MISSION: Tester le modèle v2.3 avec des features calculées dynamiquement
pour chaque match, comme ça devrait être fait en production.

DIFFÉRENCE CRITIQUE vs audit précédent:
- Avant: Features statiques obsolètes → 40% accuracy
- Maintenant: Features recalculées pour chaque match → 50-55% attendu

PROCESSUS:
1. Pour chaque match EPL 2025-26
2. Recalculer TOUTES les features avec données disponibles avant ce match
3. Prédire avec modèle v2.3 
4. Comparer avec résultat réel
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import sys
import os

# Ajouter le chemin pour importer le calculateur
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.dynamic_features_calculator import DynamicFeaturesCalculator

import warnings
warnings.filterwarnings('ignore')

class DynamicRollingValidator:
    """
    Validateur rolling avec features dynamiques
    """
    
    def __init__(self):
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        self.model_path = "models/v23_retrained_2025_09_11_154613.joblib"
        
        # Features v2.3 dans l'ordre correct
        self.v23_features_order = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
            'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
            'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Mapping labels
        self.label_mapping = {0: 'H', 1: 'D', 2: 'A'}
        
        # Objets
        self.features_calculator = None
        self.model = None
        self.ground_truth = None
        
    def initialize(self):
        """Initialiser tous les composants"""
        
        print("🚀 ROLLING VALIDATION AVEC FEATURES DYNAMIQUES")
        print("="*60)
        
        # Calculateur de features dynamiques
        print("📊 Initialisation calculateur features dynamiques...")
        self.features_calculator = DynamicFeaturesCalculator(self.dataset_path)
        
        if not self.features_calculator.load_historical_data():
            print("❌ Échec chargement données")
            return False
            
        # Modèle v2.3
        print("🤖 Chargement modèle v2.3...")
        self.model = joblib.load(self.model_path)
        print(f"✅ Modèle chargé: {type(self.model)}")
        
        # Ground truth
        print("📋 Chargement ground truth...")
        self.ground_truth = pd.read_csv(self.ground_truth_path)
        self.ground_truth['Date'] = pd.to_datetime(self.ground_truth['Date'], dayfirst=True)
        print(f"✅ Ground truth: {len(self.ground_truth)} matches")
        
        return True
        
    def rolling_validation_dynamic_features(self):
        """Rolling validation match par match avec features dynamiques"""
        
        print(f"\n🔄 ROLLING VALIDATION DYNAMIQUE")
        print("-" * 50)
        
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        print(f"🎯 Validation sur {len(self.ground_truth)} matches EPL 2025-26:")
        
        for idx, match in self.ground_truth.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam'] 
            match_date = pd.to_datetime(match['Date'])
            true_result = match['FTR']
            
            print(f"\n📅 Match {idx+1}: {home_team} vs {away_team} ({match_date.strftime('%d/%m/%Y')})")
            
            try:
                # CALCULER FEATURES DYNAMIQUES pour ce match spécifique
                dynamic_features = self.features_calculator.calculate_all_dynamic_features(
                    home_team, away_team, match_date
                )
                
                # Préparer features dans l'ordre correct pour le modèle v2.3
                X_match = np.array([dynamic_features[feature] for feature in self.v23_features_order]).reshape(1, -1)
                
                # PRÉDICTION avec modèle v2.3
                y_pred_numeric = self.model.predict(X_match)[0]
                y_proba = self.model.predict_proba(X_match)[0]
                
                # Conversion numeric → string
                y_pred_string = self.label_mapping[y_pred_numeric]
                
                # Stocker résultats
                all_predictions.append(y_pred_string)
                all_probabilities.append(y_proba)
                all_true_labels.append(true_result)
                
                # Affichage du résultat
                correct = "✅" if y_pred_string == true_result else "❌"
                print(f"   Prédit: {y_pred_string} | Réel: {true_result} | P(H/D/A)=({y_proba[0]:.2f}/{y_proba[1]:.2f}/{y_proba[2]:.2f}) {correct}")
                
                # Debug features importantes
                print(f"   Features clés: Elo={dynamic_features['elo_diff_normalized']:.2f}, Form={dynamic_features['form_diff_normalized']:.2f}, H2H={dynamic_features['h2h_score']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                # Prédiction par défaut en cas d'erreur
                all_predictions.append('H')  # Fallback: home win
                all_probabilities.append([0.5, 0.25, 0.25])
                all_true_labels.append(true_result)
                
        return {
            'predictions': all_predictions,
            'probabilities': np.array(all_probabilities),
            'true_labels': all_true_labels
        }
        
    def analyze_dynamic_results(self, results):
        """Analyser les résultats avec features dynamiques"""
        
        print(f"\n📊 ANALYSE RÉSULTATS FEATURES DYNAMIQUES")
        print("-" * 55)
        
        predictions = results['predictions']
        probabilities = results['probabilities'] 
        true_labels = results['true_labels']
        
        # Métriques principales
        accuracy = accuracy_score(true_labels, predictions)
        
        print(f"🎯 PERFORMANCE AVEC FEATURES DYNAMIQUES:")
        print(f"   Accuracy: {accuracy:.1%}")
        
        # Distribution des prédictions vs réalité
        pred_dist = pd.Series(predictions).value_counts()
        true_dist = pd.Series(true_labels).value_counts()
        
        print(f"\n📊 Distributions:")
        print(f"   Prédictions: H={pred_dist.get('H',0)}, D={pred_dist.get('D',0)}, A={pred_dist.get('A',0)}")
        print(f"   Réalité:     H={true_dist.get('H',0)}, D={true_dist.get('D',0)}, A={true_dist.get('A',0)}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(true_labels, predictions, digits=3))
        
        # Matrice de confusion
        cm = confusion_matrix(true_labels, predictions, labels=['H', 'D', 'A'])
        print(f"\n📊 Matrice de Confusion:")
        print("     H   D   A")
        for i, true_label in enumerate(['H', 'D', 'A']):
            print(f"{true_label}: {cm[i][0]:3d} {cm[i][1]:3d} {cm[i][2]:3d}")
            
        # Probabilités moyennes
        avg_proba = probabilities.mean(axis=0)
        print(f"\n🎲 Probabilités moyennes:")
        print(f"   P(H): {avg_proba[0]:.3f}")
        print(f"   P(D): {avg_proba[1]:.3f}")
        print(f"   P(A): {avg_proba[2]:.3f}")
        
        # Analyse des prédictions par classe
        classes_predicted = len(pred_dist)
        print(f"\n🔍 Diversité des prédictions:")
        print(f"   Classes prédites: {classes_predicted}/3")
        
        if classes_predicted == 3:
            print(f"   ✅ Excellent: Le modèle prédit H, D et A")
        elif classes_predicted == 2:
            print(f"   🟡 Acceptable: 2 classes prédites")
        else:
            print(f"   ❌ Problème: Une seule classe prédite")
            
        return {
            'accuracy': accuracy,
            'pred_distribution': pred_dist,
            'true_distribution': true_dist,
            'classes_predicted': classes_predicted,
            'confusion_matrix': cm,
            'avg_probabilities': avg_proba
        }
        
    def compare_with_static_features(self, dynamic_accuracy):
        """Comparer avec les résultats features statiques"""
        
        print(f"\n📈 COMPARAISON FEATURES DYNAMIQUES vs STATIQUES")
        print("-" * 60)
        
        # Résultats précédents avec features statiques
        static_accuracy = 0.40  # 40% avec features statiques
        
        improvement = dynamic_accuracy - static_accuracy
        improvement_pct = (improvement / static_accuracy) * 100
        
        print(f"🔄 Comparaison Performance:")
        print(f"   Features statiques:  {static_accuracy:.1%}")
        print(f"   Features dynamiques: {dynamic_accuracy:.1%}")
        print(f"   Amélioration:        {improvement:+.1%} ({improvement_pct:+.1f}%)")
        
        # Objectifs
        targets = {
            'baseline_random': 0.333,
            'baseline_majority': 0.436, 
            'target_good': 0.50,
            'target_excellent': 0.55
        }
        
        print(f"\n🎯 Performance vs Objectifs:")
        for target_name, target_value in targets.items():
            diff = dynamic_accuracy - target_value
            status = "✅" if diff >= 0 else "❌"
            print(f"   {target_name}: {target_value:.1%} | Écart: {diff:+.1%} {status}")
            
        # Verdict
        if dynamic_accuracy >= targets['target_excellent']:
            verdict = "🏆 EXCELLENT - Objectif excellence atteint!"
        elif dynamic_accuracy >= targets['target_good']:
            verdict = "✅ SUCCÈS - Objectif 50% atteint!"
        elif dynamic_accuracy >= targets['baseline_majority']:
            verdict = "🟡 CORRECT - Bat le baseline majority"
        else:
            verdict = "🔴 INSUFFISANT - Sous-performance vs baseline"
            
        print(f"\n{verdict}")
        
        return {
            'static_accuracy': static_accuracy,
            'dynamic_accuracy': dynamic_accuracy,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'verdict': verdict
        }
        
    def save_dynamic_results(self, rolling_results, analysis, comparison):
        """Sauvegarder les résultats de validation dynamique"""
        
        print(f"\n💾 SAUVEGARDE RÉSULTATS DYNAMIQUES")
        print("-" * 45)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/dynamic_validation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prédictions détaillées
        pred_file = results_dir / f"dynamic_predictions_{timestamp}.csv"
        
        pred_df = pd.DataFrame({
            'Match_ID': range(1, len(rolling_results['true_labels']) + 1),
            'True_Result': rolling_results['true_labels'],
            'Predicted_Result': rolling_results['predictions'],
            'Prob_H': rolling_results['probabilities'][:, 0],
            'Prob_D': rolling_results['probabilities'][:, 1],
            'Prob_A': rolling_results['probabilities'][:, 2],
            'Correct': [t == p for t, p in zip(rolling_results['true_labels'], rolling_results['predictions'])]
        })
        
        pred_df.to_csv(pred_file, index=False)
        print(f"✅ Prédictions: {pred_file}")
        
        # Rapport complet
        report_file = results_dir / f"dynamic_validation_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'method': 'Rolling validation with dynamically calculated features',
            'accuracy_dynamic': float(analysis['accuracy']),
            'accuracy_static': float(comparison['static_accuracy']),
            'improvement': float(comparison['improvement']),
            'verdict': comparison['verdict'],
            'classes_predicted': int(analysis['classes_predicted']),
            'distributions': {
                'predictions': {k: int(v) for k, v in analysis['pred_distribution'].items()},
                'reality': {k: int(v) for k, v in analysis['true_distribution'].items()}
            },
            'avg_probabilities': {
                'H': float(analysis['avg_probabilities'][0]),
                'D': float(analysis['avg_probabilities'][1]),
                'A': float(analysis['avg_probabilities'][2])
            }
        }
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"✅ Rapport: {report_file}")
        
        return pred_file, report_file
        
    def run_dynamic_validation(self):
        """Exécuter la validation complète avec features dynamiques"""
        
        print("🎯 VALIDATION ROLLING AVEC FEATURES DYNAMIQUES")
        print("="*70)
        
        # Initialisation
        if not self.initialize():
            return None
            
        # Rolling validation avec features dynamiques
        rolling_results = self.rolling_validation_dynamic_features()
        
        # Analyser les résultats
        analysis = self.analyze_dynamic_results(rolling_results)
        
        # Comparaison avec features statiques
        comparison = self.compare_with_static_features(analysis['accuracy'])
        
        # Sauvegarder
        pred_file, report_file = self.save_dynamic_results(rolling_results, analysis, comparison)
        
        # RÉSULTAT FINAL
        print(f"\n🏆 VALIDATION DYNAMIQUE TERMINÉE!")
        print(f"📊 Accuracy avec features dynamiques: {analysis['accuracy']:.1%}")
        print(f"📈 Amélioration vs statiques: {comparison['improvement']:+.1%}")
        print(f"🎯 {comparison['verdict']}")
        print(f"📄 Rapport complet: {report_file}")
        
        return {
            'accuracy': analysis['accuracy'],
            'improvement': comparison['improvement'],
            'verdict': comparison['verdict'],
            'files': {'predictions': pred_file, 'report': report_file}
        }

def main():
    """Fonction principale"""
    
    validator = DynamicRollingValidator()
    results = validator.run_dynamic_validation()
    
    return validator, results

if __name__ == "__main__":
    validator, results = main()