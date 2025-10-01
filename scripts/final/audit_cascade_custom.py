#!/usr/bin/env python3
"""
🔍 AUDIT CASCADE CUSTOMISÉ
=========================
Audit complet du modèle cascade sans dépendance externe.
Equivalent à audit_pipeline.py mais adapté pour le modèle cascade.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, classification_report
from sklearn.calibration import calibration_curve
import joblib
import logging

# Import du modèle cascade
sys.path.append('scripts/final')
from cascade_model_production import CascadeModelProduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit_cascade")

class CascadeAuditor:
    """Auditeur spécialisé pour le modèle cascade."""
    
    def __init__(self, dataset_path, model_path=None):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
        self.data = None
        self.X = None
        self.y = None
        
    def load_data_and_model(self):
        """Chargement du dataset et du modèle."""
        try:
            logger.info("📊 CHARGEMENT DONNÉES ET MODÈLE")
            
            # Dataset
            self.data = pd.read_csv(self.dataset_path)
            logger.info(f"   Dataset: {len(self.data)} échantillons")
            
            # Features
            features = [
                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
            ]
            
            self.X = self.data[features].fillna(0)
            
            # Target
            target_mapping = {'H': 0, 'D': 1, 'A': 2}
            self.y = self.data['FullTimeResult'].map(target_mapping)
            
            # Filtrage valides
            valid_mask = self.y.notna()
            self.X = self.X[valid_mask]
            self.y = self.y[valid_mask].astype(int)
            self.data = self.data[valid_mask].reset_index(drop=True)
            
            logger.info(f"   Features: {len(features)}")
            logger.info(f"   Échantillons valides: {len(self.X)}")
            logger.info(f"   Distribution: {pd.Series(self.y).value_counts().sort_index().to_dict()}")
            
            # Modèle (créer nouveau avec bons paramètres au lieu de charger)
            self.model = CascadeModelProduction(
                draw_weight=3.0,
                draw_threshold=0.35,
                calibration_factor=0.85,
                random_state=42
            )
            # Entraînement sur données complètes
            self.model.fit(self.X, self.y)
            logger.info(f"   Modèle créé et entraîné: {type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return False
    
    def test_basic_functionality(self):
        """Test de fonctionnement de base."""
        logger.info("\n🧪 TEST FONCTIONNEMENT DE BASE")
        
        try:
            # Test sur 10 échantillons
            X_test = self.X[:10]
            
            # Prédictions
            preds = self.model.predict(X_test)
            probas = self.model.predict_proba(X_test)
            
            logger.info(f"   Prédictions: {preds}")
            logger.info(f"   Probabilités shape: {probas.shape}")
            logger.info(f"   Distribution test: {pd.Series(preds).value_counts().to_dict()}")
            
            # Vérification cohérence
            assert len(preds) == len(X_test), "Longueur prédictions incohérente"
            assert probas.shape == (len(X_test), 3), "Shape probabilités incorrecte"
            assert np.allclose(probas.sum(axis=1), 1.0, rtol=1e-3), "Probabilités non normalisées"
            
            logger.info("   ✅ Tests de base réussis")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Échec tests de base: {e}")
            return False
    
    def cross_validation_temporal(self, n_splits=5):
        """Cross-validation temporelle."""
        logger.info(f"\n📈 CROSS-VALIDATION TEMPORELLE ({n_splits} splits)")
        
        try:
            # Tri par date
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            sorted_indices = self.data['Date'].argsort()
            
            X_sorted = self.X.iloc[sorted_indices]
            y_sorted = self.y.iloc[sorted_indices]
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_accuracies = []
            cv_balanced_accuracies = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
                X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
                y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
                
                # Entraînement sur ce fold
                fold_model = CascadeModelProduction(
                    draw_weight=3.0,
                    draw_threshold=0.35,
                    calibration_factor=0.85,
                    random_state=42
                )
                fold_model.fit(X_train, y_train)
                
                # Prédictions
                fold_preds = fold_model.predict(X_test)
                
                # Conversion pour métriques
                y_test_str = pd.Series(y_test).map({0: 'H', 1: 'D', 2: 'A'})
                
                # Métriques
                acc = accuracy_score(y_test_str, fold_preds)
                bal_acc = balanced_accuracy_score(y_test_str, fold_preds)
                
                cv_accuracies.append(acc)
                cv_balanced_accuracies.append(bal_acc)
                
                logger.info(f"   Fold {fold+1}: {acc:.3f} accuracy, {bal_acc:.3f} balanced")
            
            # Statistiques CV
            mean_acc = np.mean(cv_accuracies)
            std_acc = np.std(cv_accuracies)
            mean_bal_acc = np.mean(cv_balanced_accuracies)
            
            logger.info(f"\n   📊 RÉSULTATS CV:")
            logger.info(f"   Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            logger.info(f"   Balanced Accuracy: {mean_bal_acc:.3f}")
            
            return {
                'cv_accuracy_mean': mean_acc,
                'cv_accuracy_std': std_acc,
                'cv_balanced_accuracy': mean_bal_acc,
                'cv_scores': cv_accuracies
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur CV: {e}")
            return None
    
    def performance_analysis(self):
        """Analyse de performance complète."""
        logger.info(f"\n📊 ANALYSE DE PERFORMANCE")
        
        try:
            # Prédictions complètes
            all_preds = self.model.predict(self.X)
            all_probas = self.model.predict_proba(self.X)
            
            # Conversion targets
            y_str = pd.Series(self.y).map({0: 'H', 1: 'D', 2: 'A'})
            
            # Métriques globales
            accuracy = accuracy_score(y_str, all_preds)
            balanced_acc = balanced_accuracy_score(y_str, all_preds)
            
            logger.info(f"   Accuracy globale: {accuracy:.3f}")
            logger.info(f"   Balanced Accuracy: {balanced_acc:.3f}")
            
            # Distribution prédictions
            pred_dist = pd.Series(all_preds).value_counts(normalize=True).sort_index() * 100
            true_dist = y_str.value_counts(normalize=True).sort_index() * 100
            
            logger.info(f"\n   📊 DISTRIBUTIONS:")
            logger.info(f"   Prédites: H={pred_dist.get('H', 0):.1f}%, D={pred_dist.get('D', 0):.1f}%, A={pred_dist.get('A', 0):.1f}%")
            logger.info(f"   Réelles:  H={true_dist.get('H', 0):.1f}%, D={true_dist.get('D', 0):.1f}%, A={true_dist.get('A', 0):.1f}%")
            
            # Performance par classe
            logger.info(f"\n   📈 PERFORMANCE PAR CLASSE:")
            for outcome in ['H', 'D', 'A']:
                mask = y_str == outcome
                if mask.sum() > 0:
                    correct = (all_preds[mask] == outcome).sum()
                    total = mask.sum()
                    recall = correct / total
                    logger.info(f"   {outcome}: {correct}/{total} = {recall:.3f} recall")
            
            # Détection des draws
            draws_predicted = (all_preds == 'D').sum()
            draws_actual = (y_str == 'D').sum()
            draws_correct = ((all_preds == 'D') & (y_str == 'D')).sum()
            
            logger.info(f"\n   🎯 SPÉCIALISATION DRAWS:")
            logger.info(f"   Draws prédits: {draws_predicted}/{len(y_str)} ({draws_predicted/len(y_str)*100:.1f}%)")
            logger.info(f"   Draws réels: {draws_actual}/{len(y_str)} ({draws_actual/len(y_str)*100:.1f}%)")
            logger.info(f"   Draws corrects: {draws_correct}/{draws_actual} = {draws_correct/draws_actual:.3f} recall")
            
            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'draws_predicted': draws_predicted,
                'draws_actual': draws_actual,
                'draws_recall': draws_correct/draws_actual if draws_actual > 0 else 0,
                'pred_distribution': pred_dist.to_dict(),
                'true_distribution': true_dist.to_dict()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse: {e}")
            return None
    
    def robustness_test(self, n_seeds=3):
        """Test de robustesse avec différents seeds."""
        logger.info(f"\n🔄 TEST ROBUSTESSE ({n_seeds} seeds)")
        
        try:
            seeds = [42, 123, 456][:n_seeds]
            seed_accuracies = []
            
            for seed in seeds:
                # Modèle avec ce seed
                test_model = CascadeModelProduction(
                    draw_weight=3.0,
                    draw_threshold=0.35,
                    calibration_factor=0.85,
                    random_state=seed
                )
                test_model.fit(self.X, self.y)
                
                # Prédictions
                test_preds = test_model.predict(self.X)
                y_str = pd.Series(self.y).map({0: 'H', 1: 'D', 2: 'A'})
                
                # Accuracy
                acc = accuracy_score(y_str, test_preds)
                seed_accuracies.append(acc)
                
                logger.info(f"   Seed {seed}: {acc:.3f} accuracy")
            
            # Variance
            variance = np.var(seed_accuracies)
            mean_robust = np.mean(seed_accuracies)
            
            logger.info(f"\n   📊 ROBUSTESSE:")
            logger.info(f"   Accuracy moyenne: {mean_robust:.3f}")
            logger.info(f"   Variance: {variance:.6f}")
            
            # Critère de robustesse
            robust = variance < 0.001  # Seuil strict
            logger.info(f"   Robuste: {'✅' if robust else '❌'}")
            
            return {
                'robust_accuracy_mean': mean_robust,
                'robust_variance': variance,
                'is_robust': robust,
                'seed_accuracies': seed_accuracies
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur robustesse: {e}")
            return None
    
    def baseline_comparison(self):
        """Comparaison avec baselines."""
        logger.info(f"\n🏆 COMPARAISON BASELINES")
        
        try:
            y_str = pd.Series(self.y).map({0: 'H', 1: 'D', 2: 'A'})
            model_preds = self.model.predict(self.X)
            model_acc = accuracy_score(y_str, model_preds)
            
            # Baseline majoritaire
            majority_class = y_str.mode()[0]
            majority_preds = [majority_class] * len(y_str)
            majority_acc = accuracy_score(y_str, majority_preds)
            
            # Baseline aléatoire
            np.random.seed(42)
            random_preds = np.random.choice(['H', 'D', 'A'], size=len(y_str))
            random_acc = accuracy_score(y_str, random_preds)
            
            # Baseline pondérée
            class_weights = y_str.value_counts(normalize=True)
            weighted_preds = np.random.choice(
                y_str.unique(), 
                size=len(y_str), 
                p=[class_weights[c] for c in y_str.unique()]
            )
            weighted_acc = accuracy_score(y_str, weighted_preds)
            
            logger.info(f"   Modèle cascade: {model_acc:.3f}")
            logger.info(f"   Majoritaire ({majority_class}): {majority_acc:.3f}")
            logger.info(f"   Aléatoire: {random_acc:.3f}")
            logger.info(f"   Pondérée: {weighted_acc:.3f}")
            
            # Améliorations
            vs_majority = model_acc - majority_acc
            vs_random = model_acc - random_acc
            
            logger.info(f"\n   📈 AMÉLIORATIONS:")
            logger.info(f"   vs Majoritaire: {vs_majority:+.3f}")
            logger.info(f"   vs Aléatoire: {vs_random:+.3f}")
            
            return {
                'model_accuracy': model_acc,
                'majority_accuracy': majority_acc,
                'random_accuracy': random_acc,
                'weighted_accuracy': weighted_acc,
                'improvement_vs_majority': vs_majority,
                'improvement_vs_random': vs_random
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur baselines: {e}")
            return None
    
    def full_audit(self):
        """Audit complet du modèle cascade."""
        logger.info("🔍 AUDIT COMPLET MODÈLE CASCADE")
        logger.info("=" * 50)
        
        audit_results = {}
        
        # 1. Chargement
        if not self.load_data_and_model():
            return None
        
        # 2. Tests de base
        if not self.test_basic_functionality():
            return None
        
        # 3. Cross-validation
        cv_results = self.cross_validation_temporal()
        if cv_results:
            audit_results.update(cv_results)
        
        # 4. Performance
        perf_results = self.performance_analysis()
        if perf_results:
            audit_results.update(perf_results)
        
        # 5. Robustesse
        robust_results = self.robustness_test()
        if robust_results:
            audit_results.update(robust_results)
        
        # 6. Baselines
        baseline_results = self.baseline_comparison()
        if baseline_results:
            audit_results.update(baseline_results)
        
        # 7. Résumé final
        self.audit_summary(audit_results)
        
        return audit_results
    
    def audit_summary(self, results):
        """Résumé de l'audit."""
        logger.info(f"\n🏆 RÉSUMÉ AUDIT CASCADE")
        logger.info("=" * 30)
        
        # Critères de validation
        cv_acc = results.get('cv_accuracy_mean', 0)
        cv_std = results.get('cv_accuracy_std', 999)
        accuracy = results.get('accuracy', 0)
        robust = results.get('is_robust', False)
        vs_majority = results.get('improvement_vs_majority', 0)
        draws_recall = results.get('draws_recall', 0)
        
        logger.info(f"📊 MÉTRIQUES CLÉS:")
        logger.info(f"   CV Accuracy: {cv_acc:.3f} ± {cv_std:.3f}")
        logger.info(f"   Accuracy globale: {accuracy:.3f}")
        logger.info(f"   Amélioration vs majoritaire: {vs_majority:+.3f}")
        logger.info(f"   Recall draws: {draws_recall:.3f}")
        logger.info(f"   Robuste: {'✅' if robust else '❌'}")
        
        # Critères de production
        production_ready = (
            cv_acc > 0.50 and  # > 50% CV accuracy
            cv_std < 0.05 and  # Variance < 5%
            robust and         # Robuste aux seeds
            vs_majority > 0.05 # Amélioration significative
        )
        
        logger.info(f"\n🎯 VALIDATION PRODUCTION:")
        logger.info(f"   CV > 50%: {'✅' if cv_acc > 0.50 else '❌'}")
        logger.info(f"   Variance < 5%: {'✅' if cv_std < 0.05 else '❌'}")
        logger.info(f"   Robuste: {'✅' if robust else '❌'}")
        logger.info(f"   Amélioration > 5%: {'✅' if vs_majority > 0.05 else '❌'}")
        
        if production_ready:
            logger.info(f"\n✅ MODÈLE CASCADE VALIDÉ POUR PRODUCTION")
            print(f"\n🎯 VALIDATION RÉUSSIE!")
            print(f"   CV Accuracy: {cv_acc:.1%} ± {cv_std:.1%}")
            print(f"   Draws recall: {draws_recall:.1%}")
            print(f"   Status: ✅ PRODUCTION READY")
        else:
            logger.info(f"\n❌ MODÈLE CASCADE NON VALIDÉ")
            print(f"\n⚠️  VALIDATION ÉCHOUÉE")
            print(f"   Voir les métriques ci-dessus pour diagnostics")
        
        return production_ready

def main():
    """Audit principal."""
    dataset_path = "data/processed/v15_final_enhanced.csv"
    
    auditor = CascadeAuditor(dataset_path)
    results = auditor.full_audit()
    
    return results is not None

if __name__ == "__main__":
    success = main()