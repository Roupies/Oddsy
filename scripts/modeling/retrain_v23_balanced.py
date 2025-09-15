#!/usr/bin/env python3
"""
Retraining v2.3 Balanced - Récupération du 54% Réaliste
=======================================================

OBJECTIF: Corriger le home bias systématique du v2.3 original (48.3% accuracy)
et récupérer une performance équilibrée de ~54% en rolling validation.

STRATÉGIE:
1. Class weight équilibré pour forcer détection Draw/Away
2. Calibration multi-classe pour éviter biais probabiliste
3. Validation rolling stricte sur EPL 2025-26 GW1-4
4. Comparaison avec v2.3 original

RÉSULTAT ATTENDU: 52-54% accuracy avec recall équilibré H/D/A
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V23BalancedTrainer:
    """
    Entraîneur pour modèle v2.3 équilibré et calibré
    """
    
    def __init__(self):
        # Chemins de données
        self.dataset_path = "data/processed/v15_final_enhanced.csv"
        self.ground_truth_path = "data/validation/ground_truth_gw1_4.csv"
        
        # Features v2.3 exactes (validées dans le dataset v15)
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
        
        # CORRECTION DU HOME BIAS: Class weights équilibrés
        self.class_weights = {
            'H': 1.0,    # Home: poids normal
            'D': 1.5,    # Draw: surpondéré pour forcer détection
            'A': 1.2     # Away: modérément surpondéré
        }
        
        # Données
        self.dataset = None
        self.X_train = None
        self.y_train = None
        self.model = None
        
    def load_and_prepare_data(self):
        """Charger et préparer les données d'entraînement"""
        
        print("📊 RETRAINING V2.3 BALANCED - CORRECTION HOME BIAS")
        print("="*60)
        
        # Charger dataset
        print(f"📂 Chargement: {self.dataset_path}")
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        
        print(f"✅ Dataset chargé: {len(self.dataset)} matches")
        
        # Vérifier features disponibles
        available_features = set(self.dataset.columns)
        missing_features = [f for f in self.v23_features if f not in available_features]
        
        if missing_features:
            print(f"❌ Features manquantes: {missing_features}")
            return False
        
        print(f"✅ Toutes les features v2.3 disponibles ({len(self.v23_features)})")
        
        # Filtrer données d'entraînement (exclure EPL 2025-26 pour éviter data leakage)
        training_data = self.dataset[
            self.dataset['Season'] != '2025-2026'
        ].copy()
        
        print(f"📈 Données d'entraînement: {len(training_data)} matches (EPL 2025-26 exclue)")
        
        # Préparer X et y
        self.X_train = training_data[self.v23_features]
        self.y_train = training_data['FullTimeResult']
        
        # Statistiques des classes
        class_counts = self.y_train.value_counts()
        print(f"📊 Distribution des classes:")
        for cls, count in class_counts.items():
            print(f"   {cls}: {count} ({count/len(self.y_train)*100:.1f}%)")
            
        return True
        
    def train_balanced_model(self):
        """Entraîner le modèle v2.3 équilibré avec calibration"""
        
        print(f"\n🌲 ENTRAÎNEMENT MODÈLE ÉQUILIBRÉ")
        print("-" * 50)
        
        # Modèle RandomForest optimisé avec class weights
        print("🔧 Configuration RandomForest équilibré:")
        print(f"   Class weights: {self.class_weights}")
        
        base_rf = RandomForestClassifier(
            n_estimators=300,      # Optimisé pour performance
            max_depth=12,          # Éviter overfitting
            max_features='sqrt',   # Réduction dimensionnalité
            min_samples_split=10,  # Robustesse
            min_samples_leaf=5,    # Généralisation
            class_weight=self.class_weights,  # CORRECTION HOME BIAS
            random_state=42,
            n_jobs=-1
        )
        
        # CALIBRATION MULTI-CLASSE pour probabilités équilibrées
        print("📏 Ajout calibration multi-classe (isotonic)...")
        self.model = CalibratedClassifierCV(
            base_rf, 
            method='isotonic',  # Plus robuste que 'sigmoid'
            cv=3               # Cross-validation pour calibration
        )
        
        # Entraînement
        print("🚀 Entraînement en cours...")
        self.model.fit(self.X_train, self.y_train)
        
        print("✅ Modèle équilibré entraîné avec succès")
        
        return True
        
    def validate_with_cross_validation(self):
        """Validation croisée temporelle pour évaluer le modèle"""
        
        print(f"\n📊 VALIDATION CROISÉE TEMPORELLE")
        print("-" * 40)
        
        # TimeSeriesSplit pour respect temporel
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Métriques de validation
        cv_results = cross_validate(
            self.model, 
            self.X_train, 
            self.y_train,
            cv=tscv,
            scoring=['accuracy', 'f1_macro'],
            return_train_score=False
        )
        
        accuracy_scores = cv_results['test_accuracy']
        f1_macro_scores = cv_results['test_f1_macro']
        
        print(f"🎯 Résultats Cross-Validation:")
        print(f"   Accuracy: {accuracy_scores.mean():.3f} ± {accuracy_scores.std():.3f}")
        print(f"   F1-Macro: {f1_macro_scores.mean():.3f} ± {f1_macro_scores.std():.3f}")
        
        # Prédictions sur l'ensemble d'entraînement pour analyse
        y_pred_train = self.model.predict(self.X_train)
        
        print(f"\n📈 Performance sur données d'entraînement:")
        train_acc = accuracy_score(self.y_train, y_pred_train)
        print(f"   Accuracy: {train_acc:.3f}")
        
        # Distribution des prédictions
        pred_counts = pd.Series(y_pred_train).value_counts()
        print(f"   Distribution prédictions:")
        for cls, count in pred_counts.items():
            print(f"      {cls}: {count} ({count/len(y_pred_train)*100:.1f}%)")
            
        return {
            'cv_accuracy_mean': accuracy_scores.mean(),
            'cv_accuracy_std': accuracy_scores.std(),
            'cv_f1_macro_mean': f1_macro_scores.mean(),
            'train_accuracy': train_acc
        }
        
    def rolling_validation_gw1_4(self):
        """Validation rolling sur EPL 2025-26 GW1-4"""
        
        print(f"\n🔄 ROLLING VALIDATION EPL 2025-26 GW1-4")
        print("-" * 50)
        
        # Charger ground truth
        try:
            ground_truth = pd.read_csv(self.ground_truth_path)
            ground_truth['Date'] = pd.to_datetime(ground_truth['Date'], dayfirst=True)
            print(f"✅ Ground truth: {len(ground_truth)} matches")
        except FileNotFoundError:
            print(f"❌ Ground truth introuvable: {self.ground_truth_path}")
            return None
            
        # Données EPL 2025-26 pour features
        epl_2025_data = self.dataset[
            self.dataset['Season'] == '2025-2026'
        ].copy()
        
        if len(epl_2025_data) == 0:
            print("❌ Aucune donnée EPL 2025-26 dans le dataset")
            return None
            
        print(f"📅 Données EPL 2025-26: {len(epl_2025_data)} matches")
        
        # Limiter aux matches avec ground truth
        test_matches = epl_2025_data.head(len(ground_truth))
        
        if len(test_matches) < len(ground_truth):
            print(f"⚠️  Seulement {len(test_matches)} matches EPL vs {len(ground_truth)} ground truth")
            
        # Features pour prédiction
        X_test = test_matches[self.v23_features]
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Ground truth (ajusté à la taille disponible)
        y_true = ground_truth['FTR'].head(len(test_matches))
        
        # Métriques
        rolling_accuracy = accuracy_score(y_true, y_pred)
        
        print(f"🎯 RÉSULTATS ROLLING VALIDATION:")
        print(f"   Accuracy: {rolling_accuracy:.1%}")
        
        # Distribution des prédictions vs réalité
        pred_dist = pd.Series(y_pred).value_counts()
        true_dist = y_true.value_counts()
        
        print(f"\n📊 Distribution H/D/A:")
        print(f"   Prédictions → H: {pred_dist.get('H', 0)}, D: {pred_dist.get('D', 0)}, A: {pred_dist.get('A', 0)}")
        print(f"   Réalité     → H: {true_dist.get('H', 0)}, D: {true_dist.get('D', 0)}, A: {true_dist.get('A', 0)}")
        
        # Rapport détaillé
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, digits=3))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred, labels=['H', 'D', 'A'])
        print(f"\n📊 Matrice de Confusion:")
        print("     H   D   A")
        for i, true_label in enumerate(['H', 'D', 'A']):
            print(f"{true_label}: {cm[i][0]:3d} {cm[i][1]:3d} {cm[i][2]:3d}")
            
        return {
            'rolling_accuracy': rolling_accuracy,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_true,
            'pred_distribution': pred_dist,
            'true_distribution': true_dist
        }
        
    def save_balanced_model(self, cv_results, rolling_results):
        """Sauvegarder le modèle équilibré avec métadonnées"""
        
        print(f"\n💾 SAUVEGARDE MODÈLE ÉQUILIBRÉ")
        print("-" * 40)
        
        # Répertoires
        models_dir = Path("models")
        results_dir = Path("results/v23_balanced")
        models_dir.mkdir(exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder modèle
        model_file = models_dir / f"v23_balanced_calibrated_{timestamp}.joblib"
        
        model_info = {
            'model': self.model,
            'features': self.v23_features,
            'class_weights': self.class_weights,
            'training_samples': len(self.X_train),
            'cv_results': cv_results,
            'rolling_results': {k: v for k, v in rolling_results.items() 
                              if k not in ['predictions', 'probabilities', 'true_labels']},
            'timestamp': timestamp
        }
        
        joblib.dump(model_info, model_file)
        print(f"✅ Modèle sauvegardé: {model_file}")
        
        # Sauvegarder prédictions rolling pour analyse
        if rolling_results:
            pred_file = results_dir / f"rolling_predictions_gw1_4_{timestamp}.csv"
            
            pred_df = pd.DataFrame({
                'True_Result': rolling_results['true_labels'],
                'Predicted_Result': rolling_results['predictions'],
                'Prob_H': rolling_results['probabilities'][:, 0] if len(rolling_results['probabilities'][0]) > 0 else 0,
                'Prob_D': rolling_results['probabilities'][:, 1] if len(rolling_results['probabilities'][0]) > 1 else 0,
                'Prob_A': rolling_results['probabilities'][:, 2] if len(rolling_results['probabilities'][0]) > 2 else 0,
                'Correct': rolling_results['true_labels'].values == rolling_results['predictions']
            })
            
            pred_df.to_csv(pred_file, index=False)
            print(f"✅ Prédictions rolling: {pred_file}")
            
        return model_file
        
    def run_complete_retraining(self):
        """Exécuter le retraining complet"""
        
        print("🚀 RETRAINING V2.3 BALANCED - CORRECTION HOME BIAS")
        print("="*70)
        
        # Charger données
        if not self.load_and_prepare_data():
            return None
            
        # Entraîner modèle équilibré
        if not self.train_balanced_model():
            return None
            
        # Validation croisée
        cv_results = self.validate_with_cross_validation()
        
        # Rolling validation
        rolling_results = self.rolling_validation_gw1_4()
        
        if rolling_results is None:
            print("⚠️  Rolling validation échouée")
            rolling_results = {}
            
        # Sauvegarder
        model_file = self.save_balanced_model(cv_results, rolling_results)
        
        # Résumé final
        print(f"\n🏆 RETRAINING TERMINÉ!")
        if rolling_results:
            print(f"📊 Accuracy Rolling GW1-4: {rolling_results['rolling_accuracy']:.1%}")
            print(f"📈 Amélioration vs Home-Bias: {rolling_results['rolling_accuracy'] - 0.483:+.1%}")
        
        print(f"💾 Modèle équilibré: {model_file}")
        
        return {
            'model_file': model_file,
            'cv_results': cv_results,
            'rolling_results': rolling_results
        }

def main():
    """Fonction principale"""
    
    trainer = V23BalancedTrainer()
    results = trainer.run_complete_retraining()
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()