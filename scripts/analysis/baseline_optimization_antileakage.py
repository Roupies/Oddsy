#!/usr/bin/env python3
"""
🔧 OPTIMISATION BASELINE ANTI-LEAKAGE
=====================================
Optimisation du baseline RandomForest pour EPL 2025-26 avec validation anti-leakage stricte.

RÈGLES ANTI-LEAKAGE:
- Split temporel strict: Train ≤ 2025-05-25, Test ≥ 2025-08-15
- Calibration uniquement sur données historiques
- Aucune donnée future utilisée pour optimisation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline_opt")

class BaselineOptimizerAntiLeakage:
    """Optimiseur baseline avec protection anti-leakage."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Split temporel strict
        self.train_cutoff = pd.to_datetime('2025-05-25')
        self.test_start = pd.to_datetime('2025-08-15')
        
    def load_data(self):
        """Chargement données avec split temporel strict."""
        logger.info("📊 CHARGEMENT DONNÉES ANTI-LEAKAGE")
        logger.info("=" * 50)
        
        self.data = pd.read_csv(self.dataset_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Target mapping
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        self.data['target'] = self.data['FullTimeResult'].map(target_mapping)
        
        # Filtrage données valides
        valid_mask = self.data['target'].notna()
        self.data = self.data[valid_mask].sort_values('Date').reset_index(drop=True)
        
        logger.info(f"   Total échantillons: {len(self.data)}")
        logger.info(f"   Train cutoff: {self.train_cutoff.strftime('%Y-%m-%d')}")
        logger.info(f"   Test start: {self.test_start.strftime('%Y-%m-%d')}")
        
        # Split temporal
        self.train_mask = self.data['Date'] <= self.train_cutoff
        self.test_mask = self.data['Date'] >= self.test_start
        
        train_count = self.train_mask.sum()
        test_count = self.test_mask.sum()
        
        logger.info(f"   Train set: {train_count} échantillons")
        logger.info(f"   Test set: {test_count} échantillons")
        
        if test_count < 30:
            logger.warning(f"⚠️ Test set petit: {test_count} échantillons")
        
        return True
    
    def optimize_draw_thresholds(self):
        """Optimisation seuils draws sur validation historique uniquement."""
        logger.info("\\n🎯 OPTIMISATION SEUILS DRAWS (ANTI-LEAKAGE)")
        logger.info("=" * 50)
        
        # Données train uniquement
        train_data = self.data[self.train_mask]
        X_train = train_data[self.features].fillna(0)
        y_train = train_data['target'].astype(int)
        
        logger.info(f"   Optimisation sur {len(X_train)} échantillons historiques")
        
        # Cross-validation temporelle sur train set uniquement
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Modèle baseline standard
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        )
        
        # Test différents seuils draw
        draw_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
        best_threshold = None
        best_accuracy = 0
        results = []
        
        for threshold in draw_thresholds:
            logger.info(f"\\n  Test seuil draw: {threshold}")
            
            fold_accuracies = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Entraînement
                calibrated_model = CalibratedClassifierCV(base_model, cv=3)
                calibrated_model.fit(X_fold_train, y_fold_train)
                
                # Prédictions avec seuil draw
                probas = calibrated_model.predict_proba(X_val)
                predictions = self._apply_draw_threshold(probas, threshold)
                
                accuracy = accuracy_score(y_val, predictions)
                fold_accuracies.append(accuracy)
                
                logger.info(f"    Fold {fold+1}: {accuracy:.3f}")
            
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            
            results.append({
                'threshold': threshold,
                'accuracy': mean_accuracy,
                'std': std_accuracy
            })
            
            logger.info(f"    Moyenne: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
            
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_threshold = threshold
        
        logger.info(f"\\n🏆 MEILLEUR SEUIL DRAW: {best_threshold}")
        logger.info(f"   Accuracy CV: {best_accuracy:.3f}")
        
        return best_threshold, results
    
    def optimize_class_weights(self):
        """Optimisation class_weight sur validation historique."""
        logger.info("\\n⚖️ OPTIMISATION CLASS_WEIGHT (ANTI-LEAKAGE)")
        logger.info("=" * 50)
        
        # Données train uniquement
        train_data = self.data[self.train_mask]
        X_train = train_data[self.features].fillna(0)
        y_train = train_data['target'].astype(int)
        
        # Grid search class_weight
        class_weight_options = [
            "balanced",
            {0: 1, 1: 2, 2: 1},    # Boost draws x2
            {0: 1, 1: 3, 2: 1},    # Boost draws x3
            {0: 1, 1: 1.5, 2: 1},  # Boost draws x1.5
            None                    # Pas de pondération
        ]
        
        best_weights = None
        best_accuracy = 0
        results = []
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        for i, weights in enumerate(class_weight_options):
            logger.info(f"\\n  Test class_weight {i+1}: {weights}")
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=3,
                class_weight=weights,
                random_state=42
            )
            
            fold_accuracies = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Entraînement avec calibration
                calibrated_model = CalibratedClassifierCV(model, cv=3)
                calibrated_model.fit(X_fold_train, y_fold_train)
                
                # Prédictions
                predictions = calibrated_model.predict(X_val)
                accuracy = accuracy_score(y_val, predictions)
                fold_accuracies.append(accuracy)
                
                logger.info(f"    Fold {fold+1}: {accuracy:.3f}")
            
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            
            results.append({
                'class_weight': weights,
                'accuracy': mean_accuracy,
                'std': std_accuracy
            })
            
            logger.info(f"    Moyenne: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
            
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_weights = weights
        
        logger.info(f"\\n🏆 MEILLEUR CLASS_WEIGHT: {best_weights}")
        logger.info(f"   Accuracy CV: {best_accuracy:.3f}")
        
        return best_weights, results
    
    def _apply_draw_threshold(self, probas, threshold):
        """Application seuil draw sur probabilités."""
        predictions = []
        
        for proba in probas:
            # Si P(Draw) > threshold, prédire Draw
            if proba[1] > threshold:
                predictions.append(1)  # Draw
            else:
                # Sinon prédire classe avec plus haute proba entre H et A
                if proba[0] > proba[2]:
                    predictions.append(0)  # Home
                else:
                    predictions.append(2)  # Away
        
        return np.array(predictions)
    
    def test_final_model(self, best_threshold, best_weights):
        """Test final anti-leakage sur EPL 2025-26."""
        logger.info("\\n🎯 TEST FINAL ANTI-LEAKAGE EPL 2025-26")
        logger.info("=" * 50)
        
        # Split strict: train historique, test EPL 2025-26
        train_data = self.data[self.train_mask]
        test_data = self.data[self.test_mask]
        
        X_train = train_data[self.features].fillna(0)
        y_train = train_data['target'].astype(int)
        X_test = test_data[self.features].fillna(0)
        y_test = test_data['target'].astype(int)
        
        logger.info(f"   Train: {len(X_train)} échantillons (≤ {self.train_cutoff.strftime('%Y-%m-%d')})")
        logger.info(f"   Test:  {len(X_test)} échantillons (≥ {self.test_start.strftime('%Y-%m-%d')})")
        
        # Modèle optimisé
        optimized_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight=best_weights,
            random_state=42
        )
        
        # Entraînement avec calibration
        calibrated_model = CalibratedClassifierCV(optimized_model, cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Prédictions
        if best_threshold:
            probas = calibrated_model.predict_proba(X_test)
            predictions = self._apply_draw_threshold(probas, best_threshold)
        else:
            predictions = calibrated_model.predict(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, predictions)
        
        # Distributions
        test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index() * 100
        pred_dist = pd.Series(predictions).value_counts(normalize=True).sort_index() * 100
        
        logger.info(f"\\n📊 RÉSULTATS FINAUX:")
        logger.info(f"   Accuracy: {accuracy:.3f}")
        logger.info(f"   Draw threshold: {best_threshold}")
        logger.info(f"   Class weight: {best_weights}")
        
        logger.info(f"\\n📈 DISTRIBUTIONS:")
        logger.info(f"   Réel:  H={test_dist.get(0, 0):4.1f}% D={test_dist.get(1, 0):4.1f}% A={test_dist.get(2, 0):4.1f}%")
        logger.info(f"   Pred:  H={pred_dist.get(0, 0):4.1f}% D={pred_dist.get(1, 0):4.1f}% A={pred_dist.get(2, 0):4.1f}%")
        
        # Matrice confusion
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"\\n📊 MATRICE CONFUSION:")
        logger.info(f"         Pred: H    D    A")
        for i, (real_label, row) in enumerate(zip(['H', 'D', 'A'], cm)):
            logger.info(f"   Real {real_label}:    {row[0]:2d}   {row[1]:2d}   {row[2]:2d}")
        
        # Comparaison baseline
        majority_acc = np.mean(y_test == 0)
        logger.info(f"\\n🏆 COMPARAISON:")
        logger.info(f"   Majority class: {majority_acc:.3f}")
        logger.info(f"   Modèle optimisé: {accuracy:.3f}")
        logger.info(f"   Amélioration: {accuracy - majority_acc:+.3f}")
        
        # Verdict
        if accuracy > 0.50:
            verdict = "✅ PRODUCTION READY"
        elif accuracy > majority_acc:
            verdict = "⚠️ MARGINAL"
        else:
            verdict = "❌ ÉCHEC"
        
        logger.info(f"\\n🎯 VERDICT: {verdict}")
        
        return {
            'accuracy': accuracy,
            'verdict': verdict,
            'best_threshold': best_threshold,
            'best_weights': best_weights,
            'test_distribution': test_dist.to_dict(),
            'pred_distribution': pred_dist.to_dict()
        }

def main():
    """Optimisation principale anti-leakage."""
    logger.info("🔧 BASELINE OPTIMIZATION ANTI-LEAKAGE")
    logger.info("=" * 60)
    
    dataset_path = "data/processed/v_auto_update_20250916_110247.csv"
    optimizer = BaselineOptimizerAntiLeakage(dataset_path)
    
    # Chargement données
    if not optimizer.load_data():
        return None
    
    # Phase 1: Optimisation seuils draws
    best_threshold, draw_results = optimizer.optimize_draw_thresholds()
    
    # Phase 2: Optimisation class_weight
    best_weights, weight_results = optimizer.optimize_class_weights()
    
    # Phase 3: Test final anti-leakage
    final_results = optimizer.test_final_model(best_threshold, best_weights)
    
    # Synthèse finale
    print(f"\\n🔧 BASELINE OPTIMIZATION - RÉSULTATS FINAUX")
    print(f"\\n📊 Paramètres optimaux:")
    print(f"   Draw threshold: {best_threshold}")
    print(f"   Class weight: {best_weights}")
    print(f"\\n🎯 Performance EPL 2025-26:")
    print(f"   Accuracy: {final_results['accuracy']:.3f}")
    print(f"   Verdict: {final_results['verdict']}")
    
    return final_results

if __name__ == "__main__":
    results = main()