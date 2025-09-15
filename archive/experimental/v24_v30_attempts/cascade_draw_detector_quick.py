#!/usr/bin/env python3
"""
CASCADE DRAW DETECTOR - Version rapide pour proof of concept
============================================================

Test rapide de l'approche cascade pour améliorer les prédictions de draw
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import sys
import os

# Importer le calculateur dynamique
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.dynamic_features_calculator import DynamicFeaturesCalculator

import warnings
warnings.filterwarnings('ignore')

class QuickCascadeTest:
    """Test rapide du concept cascade"""
    
    def __init__(self):
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.v23_model_path = "models/v23_retrained_2025_09_11_154613.joblib"
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        
        # Features v2.3 originales
        self.v23_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
            'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
            'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        self.features_calculator = None
        self.v23_model = None
        self.ground_truth = None
        self.label_mapping = {0: 'H', 1: 'D', 2: 'A'}
        
    def initialize(self):
        """Initialiser les composants"""
        
        print("🎯 QUICK CASCADE TEST - PROOF OF CONCEPT")
        print("=" * 50)
        
        # Calculateur features dynamiques
        print("📊 Chargement calculateur features dynamiques...")
        self.features_calculator = DynamicFeaturesCalculator(self.dataset_path)
        if not self.features_calculator.load_historical_data():
            return False
            
        # Modèle v2.3 original
        print("🤖 Chargement modèle v2.3...")
        self.v23_model = joblib.load(self.v23_model_path)
        
        # Ground truth EPL 2025-26
        print("📋 Chargement ground truth...")
        self.ground_truth = pd.read_csv(self.ground_truth_path)
        self.ground_truth['Date'] = pd.to_datetime(self.ground_truth['Date'], dayfirst=True)
        
        return True
        
    def calculate_draw_features_simple(self, home_team, away_team, match_date):
        """Calculer features simples pour détection draw"""
        
        # Features v2.3 de base
        base_features = self.features_calculator.calculate_all_dynamic_features(
            home_team, away_team, match_date
        )
        
        # Features spécialisées draw simplifiées
        draw_features = {}
        
        # 1. Teams balance (équilibre Elo)
        elo_home = self.features_calculator.calculate_dynamic_elo(home_team, match_date)
        elo_away = self.features_calculator.calculate_dynamic_elo(away_team, match_date)
        elo_diff = abs(elo_home - elo_away)
        draw_features['elo_balance'] = max(0, 1.0 - (elo_diff / 200))  # Plus proche = plus de chance draw
        
        # 2. Form convergence
        home_form = self.features_calculator.calculate_dynamic_form(home_team, match_date, 5)
        away_form = self.features_calculator.calculate_dynamic_form(away_team, match_date, 5)
        form_diff = abs(home_form - away_form)
        draw_features['form_balance'] = 1.0 - form_diff
        
        # 3. Market uncertainty (entropy élevée)
        market_entropy = base_features.get('market_entropy_norm', 0.5)
        draw_features['market_uncertainty'] = market_entropy
        
        # 4. H2H équilibré
        h2h_score = base_features.get('h2h_score', 0.5)
        h2h_balance = 1.0 - abs(h2h_score - 0.5) * 2  # Distance de 0.5 (équilibre)
        draw_features['h2h_balance'] = h2h_balance
        
        return list(draw_features.values())
        
    def test_simple_draw_threshold(self):
        """Tester une approche simple: seuil sur probabilité draw"""
        
        print(f"\n🧪 TEST SIMPLE THRESHOLD")
        print("-" * 40)
        
        all_predictions_original = []
        all_predictions_threshold = []
        all_true_labels = []
        all_probabilities = []
        
        print(f"🎯 Test sur {len(self.ground_truth)} matches EPL 2025-26:")
        
        for idx, match in self.ground_truth.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam'] 
            match_date = pd.to_datetime(match['Date'])
            true_result = match['FTR']
            
            try:
                # Features dynamiques v2.3
                dynamic_features = self.features_calculator.calculate_all_dynamic_features(
                    home_team, away_team, match_date
                )
                
                X_match = np.array([dynamic_features[feature] for feature in self.v23_features]).reshape(1, -1)
                
                # Prédiction v2.3 originale
                y_pred_numeric = self.v23_model.predict(X_match)[0]
                y_proba = self.v23_model.predict_proba(X_match)[0]
                
                # Prédiction originale
                y_pred_original = self.label_mapping[y_pred_numeric]
                
                # Nouvelle stratégie: Si probabilité draw > seuil, prédire draw
                draw_threshold = 0.28  # Seuil à optimiser
                if y_proba[1] > draw_threshold:  # P(D) > seuil
                    y_pred_threshold = 'D'
                else:
                    # Sinon utiliser prédiction originale H/A
                    y_pred_threshold = y_pred_original if y_pred_original != 'D' else ('H' if y_proba[0] > y_proba[2] else 'A')
                
                all_predictions_original.append(y_pred_original)
                all_predictions_threshold.append(y_pred_threshold)
                all_true_labels.append(true_result)
                all_probabilities.append(y_proba)
                
                if idx < 10:  # Afficher détails pour 10 premiers
                    correct_orig = "✅" if y_pred_original == true_result else "❌"
                    correct_thresh = "✅" if y_pred_threshold == true_result else "❌"
                    print(f"   Match {idx+1}: {home_team} vs {away_team}")
                    print(f"     Original: {y_pred_original} | Threshold: {y_pred_threshold} | Réel: {true_result}")
                    print(f"     P(H/D/A): ({y_proba[0]:.2f}/{y_proba[1]:.2f}/{y_proba[2]:.2f})")
                    print(f"     Résultats: Orig {correct_orig} | Thresh {correct_thresh}")
                    
            except Exception as e:
                print(f"   ❌ Erreur match {idx}: {e}")
                continue
                
        # Analyser résultats
        acc_original = accuracy_score(all_true_labels, all_predictions_original)
        acc_threshold = accuracy_score(all_true_labels, all_predictions_threshold)
        
        print(f"\n📊 RÉSULTATS COMPARATIFS:")
        print(f"   V2.3 Original: {acc_original:.1%}")
        print(f"   Threshold Strategy: {acc_threshold:.1%}")
        print(f"   Amélioration: {(acc_threshold - acc_original):.1%}")
        
        # Distribution prédictions
        pred_dist_orig = pd.Series(all_predictions_original).value_counts()
        pred_dist_thresh = pd.Series(all_predictions_threshold).value_counts()
        true_dist = pd.Series(all_true_labels).value_counts()
        
        print(f"\n📊 Distributions:")
        print(f"   Original:   H={pred_dist_orig.get('H',0)}, D={pred_dist_orig.get('D',0)}, A={pred_dist_orig.get('A',0)}")
        print(f"   Threshold:  H={pred_dist_thresh.get('H',0)}, D={pred_dist_thresh.get('D',0)}, A={pred_dist_thresh.get('A',0)}")
        print(f"   Réalité:    H={true_dist.get('H',0)}, D={true_dist.get('D',0)}, A={true_dist.get('A',0)}")
        
        # Matrice de confusion pour threshold
        cm_thresh = confusion_matrix(all_true_labels, all_predictions_threshold, labels=['H', 'D', 'A'])
        print(f"\n📊 Matrice Confusion (Threshold Strategy):")
        print("     H   D   A")
        for i, true_label in enumerate(['H', 'D', 'A']):
            print(f"{true_label}: {cm_thresh[i][0]:3d} {cm_thresh[i][1]:3d} {cm_thresh[i][2]:3d}")
            
        return {
            'accuracy_original': acc_original,
            'accuracy_threshold': acc_threshold,
            'improvement': acc_threshold - acc_original,
            'predictions_threshold': all_predictions_threshold,
            'true_labels': all_true_labels
        }
        
    def optimize_draw_threshold(self):
        """Optimiser le seuil pour maximiser accuracy"""
        
        print(f"\n🔧 OPTIMISATION SEUIL DRAW")
        print("-" * 35)
        
        # Tester différents seuils
        thresholds = np.arange(0.15, 0.40, 0.02)
        best_threshold = 0.25
        best_accuracy = 0.0
        
        results = []
        
        for threshold in thresholds:
            # Calculer predictions avec ce seuil
            predictions = []
            true_labels = []
            
            for idx, match in self.ground_truth.iterrows():
                try:
                    home_team = match['HomeTeam']
                    away_team = match['AwayTeam'] 
                    match_date = pd.to_datetime(match['Date'])
                    true_result = match['FTR']
                    
                    dynamic_features = self.features_calculator.calculate_all_dynamic_features(
                        home_team, away_team, match_date
                    )
                    
                    X_match = np.array([dynamic_features[feature] for feature in self.v23_features]).reshape(1, -1)
                    y_proba = self.v23_model.predict_proba(X_match)[0]
                    
                    # Stratégie threshold
                    if y_proba[1] > threshold:  # P(D) > seuil
                        y_pred = 'D'
                    else:
                        # H vs A selon probabilités
                        y_pred = 'H' if y_proba[0] > y_proba[2] else 'A'
                    
                    predictions.append(y_pred)
                    true_labels.append(true_result)
                    
                except:
                    continue
                    
            accuracy = accuracy_score(true_labels, predictions)
            draws_predicted = sum(1 for p in predictions if p == 'D')
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'draws_predicted': draws_predicted
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                
            print(f"   Seuil {threshold:.2f}: {accuracy:.1%} (Draws prédits: {draws_predicted})")
        
        print(f"\n🏆 SEUIL OPTIMAL:")
        print(f"   Threshold: {best_threshold:.2f}")
        print(f"   Accuracy: {best_accuracy:.1%}")
        
        return best_threshold, best_accuracy, results

def main():
    """Test rapide du concept cascade"""
    
    tester = QuickCascadeTest()
    
    # Initialisation
    if not tester.initialize():
        print("❌ Échec initialisation")
        return
        
    # Test threshold simple
    results = tester.test_simple_draw_threshold()
    
    # Optimisation seuil
    best_threshold, best_accuracy, optimization_results = tester.optimize_draw_threshold()
    
    print(f"\n🏆 RÉSUMÉ PROOF OF CONCEPT:")
    print("=" * 45)
    print(f"✅ V2.3 Original: {results['accuracy_original']:.1%}")
    print(f"✅ Threshold Strategy: {results['accuracy_threshold']:.1%}")
    print(f"✅ Meilleur seuil optimisé: {best_accuracy:.1%}")
    print(f"📈 Amélioration potentielle: {(best_accuracy - results['accuracy_original']):.1%}")
    
    # Évaluer si ça vaut la peine
    improvement = best_accuracy - results['accuracy_original']
    if improvement > 0.02:  # +2pp
        verdict = "🎯 PROMETTEUR - Worth implementing full cascade!"
    elif improvement > 0:
        verdict = "🟡 MARGINAL - Small improvement"
    else:
        verdict = "❌ ÉCHEC - No improvement"
        
    print(f"\n{verdict}")
    
    return tester, results, optimization_results

if __name__ == "__main__":
    tester, results, optimization = main()