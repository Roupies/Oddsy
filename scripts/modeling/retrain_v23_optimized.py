#!/usr/bin/env python3
"""
Retraining v2.3 Optimized - Version Ultra-Affinée
================================================

MISSION: Récupérer le 54% realistic en affinant les hyperparamètres 
après les leçons de la première tentative.

AJUSTEMENTS DEPUIS v2.3 BALANCED:
1. Class weights mieux équilibrés (moins agressifs)
2. RandomForest optimisé pour généralisation
3. Calibration plus fine
4. Features enrichies si nécessaire

OBJECTIF: 52-54% rolling avec detection Draw > 0%
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V23OptimizedTrainer:
    """
    Entraîneur optimisé avec hyperparameter tuning
    """
    
    def __init__(self):
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        
        # Features v2.3 
        self.v23_features = [
            'elo_diff_normalized',
            'market_entropy_norm', 
            'shots_diff_normalized',
            'corners_diff_normalized',
            'form_diff_normalized',
            'h2h_score',
            'matchday_normalized',
            'home_xg_eff_10',
            'away_xg_eff_10',
            'away_goals_sum_5'
        ]
        
        # AJUSTEMENT: Class weights plus subtils (moins agressifs)
        self.class_weights = {
            'H': 1.0,    # Home: normal
            'D': 1.3,    # Draw: modérément renforcé (vs 1.5 avant)
            'A': 1.1     # Away: légèrement renforcé (vs 1.2 avant)
        }
        
        # Données
        self.dataset = None
        self.X_train = None
        self.y_train = None
        self.model = None
        
    def load_and_prepare_data(self):
        """Charger et préparer les données"""
        
        print("🎯 RETRAINING V2.3 OPTIMIZED - HYPERPARAMETER TUNING")
        print("="*65)
        
        # Charger dataset
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        
        # Vérifier features
        available_features = set(self.dataset.columns)
        missing_features = [f for f in self.v23_features if f not in available_features]
        
        if missing_features:
            print(f"❌ Features manquantes: {missing_features}")
            return False
        
        # Données d'entraînement (sans EPL 2025-26)
        training_data = self.dataset[
            self.dataset['Season'] != '2025-2026'
        ].copy()
        
        self.X_train = training_data[self.v23_features]
        self.y_train = training_data['FullTimeResult']
        
        print(f"✅ Données préparées: {len(self.X_train)} matches d'entraînement")
        
        # Distribution des classes
        class_counts = self.y_train.value_counts()
        print(f"📊 Classes: H:{class_counts.get('H',0)} ({class_counts.get('H',0)/len(self.y_train)*100:.1f}%), ")
        print(f"           D:{class_counts.get('D',0)} ({class_counts.get('D',0)/len(self.y_train)*100:.1f}%), ")
        print(f"           A:{class_counts.get('A',0)} ({class_counts.get('A',0)/len(self.y_train)*100:.1f}%)")
        
        return True
        
    def hyperparameter_tuning(self):
        """Hyperparameter tuning ciblé pour améliorer performance"""
        
        print(f"\n🔧 HYPERPARAMETER TUNING")
        print("-" * 40)
        
        # Grille de paramètres optimisée
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [8, 10, 12],
            'min_samples_split': [8, 12],
            'min_samples_leaf': [3, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        # RandomForest de base avec class weights
        base_rf = RandomForestClassifier(
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        # GridSearch avec TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)  # Réduit pour vitesse
        
        print("🔍 Recherche des meilleurs hyperparamètres...")
        grid_search = GridSearchCV(
            base_rf, 
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Meilleurs paramètres
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"✅ Meilleurs paramètres trouvés:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"✅ Score CV optimal: {best_score:.3f}")
        
        return best_params
        
    def train_optimized_model(self, best_params):
        """Entraîner le modèle avec les paramètres optimaux"""
        
        print(f"\n🌲 ENTRAÎNEMENT MODÈLE OPTIMISÉ")
        print("-" * 45)
        
        # RandomForest optimisé
        optimized_rf = RandomForestClassifier(
            **best_params,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        # Calibration avec stratégie 'sigmoid' pour meilleure généralisation
        print("📏 Calibration 'sigmoid' pour généralisation...")
        self.model = CalibratedClassifierCV(
            optimized_rf, 
            method='sigmoid',  # Plus conservateur que isotonic
            cv=3
        )
        
        # Entraînement
        print("🚀 Entraînement modèle optimisé...")
        self.model.fit(self.X_train, self.y_train)
        
        print("✅ Modèle optimisé entraîné")
        
        return True
        
    def detailed_rolling_validation(self):
        """Validation rolling détaillée avec analyse approfondie"""
        
        print(f"\n🔄 ROLLING VALIDATION DÉTAILLÉE")
        print("-" * 45)
        
        # Charger ground truth
        try:
            ground_truth = pd.read_csv(self.ground_truth_path)
            ground_truth['Date'] = pd.to_datetime(ground_truth['Date'], dayfirst=True)
        except FileNotFoundError:
            print(f"❌ Ground truth introuvable")
            return None
            
        # Données EPL 2025-26
        epl_2025_data = self.dataset[
            self.dataset['Season'] == '2025-2026'
        ].copy()
        
        test_matches = epl_2025_data.head(len(ground_truth))
        
        # Features et prédictions
        X_test = test_matches[self.v23_features]
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Ground truth
        y_true = ground_truth['FTR'].head(len(test_matches))
        
        # Métriques détaillées
        rolling_accuracy = accuracy_score(y_true, y_pred)
        
        print(f"🎯 ROLLING ACCURACY: {rolling_accuracy:.1%}")
        
        # Analyse par classe
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=['H', 'D', 'A'], average=None
        )
        
        print(f"\n📊 MÉTRIQUES PAR CLASSE:")
        for i, cls in enumerate(['H', 'D', 'A']):
            print(f"   {cls}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")
            
        # Distribution des prédictions
        pred_dist = pd.Series(y_pred).value_counts()
        true_dist = y_true.value_counts()
        
        print(f"\n📈 DISTRIBUTIONS:")
        print(f"   Prédictions: H={pred_dist.get('H',0)}, D={pred_dist.get('D',0)}, A={pred_dist.get('A',0)}")
        print(f"   Réalité:     H={true_dist.get('H',0)}, D={true_dist.get('D',0)}, A={true_dist.get('A',0)}")
        
        # Probabilités moyennes par classe prédite
        print(f"\n🎲 PROBABILITÉS MOYENNES:")
        for cls in ['H', 'D', 'A']:
            mask = y_pred == cls
            if mask.any():
                avg_proba = y_proba[mask].mean(axis=0)
                print(f"   Quand prédit {cls}: P(H)={avg_proba[0]:.3f}, P(D)={avg_proba[1]:.3f}, P(A)={avg_proba[2]:.3f}")
                
        # Analyse des erreurs
        errors = y_true != y_pred
        error_rate = errors.mean()
        print(f"\n❌ ANALYSE DES ERREURS:")
        print(f"   Taux d'erreur: {error_rate:.1%}")
        
        if errors.any():
            error_types = []
            for i in range(len(y_true)):
                if errors.iloc[i]:
                    error_types.append(f"{y_true.iloc[i]}→{y_pred[i]}")
            
            error_counts = pd.Series(error_types).value_counts()
            print(f"   Types d'erreurs fréquentes:")
            for error_type, count in error_counts.head(3).items():
                print(f"      {error_type}: {count} fois")
        
        return {
            'rolling_accuracy': rolling_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_true,
            'error_analysis': {
                'error_rate': error_rate,
                'error_types': error_counts if 'error_counts' in locals() else {}
            }
        }
        
    def save_optimized_model(self, best_params, rolling_results):
        """Sauvegarder le modèle optimisé"""
        
        print(f"\n💾 SAUVEGARDE MODÈLE OPTIMISÉ")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Modèle
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_file = models_dir / f"v23_optimized_calibrated_{timestamp}.joblib"
        
        model_info = {
            'model': self.model,
            'features': self.v23_features,
            'class_weights': self.class_weights,
            'best_hyperparams': best_params,
            'rolling_results': {k: v for k, v in rolling_results.items() 
                              if not isinstance(v, np.ndarray)},
            'timestamp': timestamp,
            'description': 'v2.3 Optimized with hyperparameter tuning and balanced class weights'
        }
        
        joblib.dump(model_info, model_file)
        print(f"✅ Modèle optimisé: {model_file}")
        
        # Résultats détaillés
        results_dir = Path("results/v23_optimized")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"optimization_results_{timestamp}.json"
        
        results_summary = {
            'timestamp': timestamp,
            'rolling_accuracy': float(rolling_results['rolling_accuracy']),
            'best_hyperparams': best_params,
            'class_weights': self.class_weights,
            'precision_by_class': {
                'H': float(rolling_results['precision'][0]),
                'D': float(rolling_results['precision'][1]), 
                'A': float(rolling_results['precision'][2])
            },
            'recall_by_class': {
                'H': float(rolling_results['recall'][0]),
                'D': float(rolling_results['recall'][1]),
                'A': float(rolling_results['recall'][2])
            },
            'f1_by_class': {
                'H': float(rolling_results['f1_scores'][0]),
                'D': float(rolling_results['f1_scores'][1]),
                'A': float(rolling_results['f1_scores'][2])
            }
        }
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        print(f"✅ Résultats détaillés: {results_file}")
        
        return model_file, results_file
        
    def run_complete_optimization(self):
        """Pipeline complet d'optimisation"""
        
        print("🎯 OPTIMISATION COMPLÈTE V2.3 - RÉCUPÉRATION 54%")
        print("="*60)
        
        # Charger données
        if not self.load_and_prepare_data():
            return None
            
        # Hyperparameter tuning
        best_params = self.hyperparameter_tuning()
        
        # Entraînement optimisé
        if not self.train_optimized_model(best_params):
            return None
            
        # Validation rolling détaillée
        rolling_results = self.detailed_rolling_validation()
        
        if rolling_results is None:
            return None
            
        # Sauvegarde
        model_file, results_file = self.save_optimized_model(best_params, rolling_results)
        
        # Résumé final
        print(f"\n🏆 OPTIMISATION TERMINÉE!")
        print(f"📊 Rolling Accuracy: {rolling_results['rolling_accuracy']:.1%}")
        
        # Vérifier si objectif atteint
        target_accuracy = 0.52
        if rolling_results['rolling_accuracy'] >= target_accuracy:
            print(f"✅ OBJECTIF ATTEINT! ({rolling_results['rolling_accuracy']:.1%} ≥ {target_accuracy:.1%})")
        else:
            improvement_needed = target_accuracy - rolling_results['rolling_accuracy']
            print(f"🔄 Amélioration nécessaire: +{improvement_needed:.1%} pour atteindre {target_accuracy:.1%}")
            
        # Check détection des draws
        draw_recall = rolling_results['recall'][1]  # Indice 1 = Draw
        if draw_recall > 0:
            print(f"✅ DRAWS DÉTECTÉS! Recall Draw: {draw_recall:.1%}")
        else:
            print(f"⚠️  Draws toujours non détectés (Recall = 0%)")
            
        return {
            'model_file': model_file,
            'results_file': results_file,
            'rolling_accuracy': rolling_results['rolling_accuracy'],
            'best_params': best_params,
            'draw_recall': draw_recall
        }

def main():
    """Fonction principale"""
    
    trainer = V23OptimizedTrainer()
    results = trainer.run_complete_optimization()
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()