#!/usr/bin/env python3
"""
🔍 AUDIT HYBRIDE COMPLET
=======================
Audit spécialisé pour le modèle hybride avec analyse par phase de saison.
Cross-validation temporelle et comparaison vs baseline v2.3.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import logging

# Import du modèle hybride
sys.path.append('scripts/final')
from hybrid_model_clean import HybridModelClean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit_hybrid")

class HybridAuditor:
    """Auditeur spécialisé pour le modèle hybride."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None
        self.features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
    def load_data(self):
        """Chargement et préparation des données."""
        try:
            logger.info("📊 CHARGEMENT DONNÉES POUR AUDIT HYBRIDE")
            
            self.data = pd.read_csv(self.dataset_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Création target
            target_mapping = {'H': 0, 'D': 1, 'A': 2}
            self.data['target'] = self.data['FullTimeResult'].map(target_mapping)
            
            # Filtrage données valides
            valid_mask = self.data['target'].notna()
            self.data = self.data[valid_mask].reset_index(drop=True)
            
            # Préparation features
            self.X = self.data[self.features].fillna(0)
            self.y = self.data['target'].astype(int)
            
            # Tri par date
            sort_indices = self.data['Date'].argsort()
            self.data = self.data.iloc[sort_indices].reset_index(drop=True)
            self.X = self.X.iloc[sort_indices].reset_index(drop=True)
            self.y = self.y.iloc[sort_indices].reset_index(drop=True)
            
            logger.info(f"   Dataset: {len(self.data)} échantillons")
            logger.info(f"   Période: {self.data['Date'].min()} → {self.data['Date'].max()}")
            logger.info(f"   Features: {len(self.features)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return False
    
    def test_basic_functionality(self):
        """Test de fonctionnement de base."""
        logger.info("\n🧪 TEST FONCTIONNEMENT DE BASE")
        
        try:
            # Test sur échantillon
            X_test = self.X[:50]
            y_test = self.y[:50]
            
            # Création et entraînement modèle hybride
            hybrid_model = HybridModelClean(
                early_season_threshold=0.15,
                cascade_draw_weight=3.0,
                cascade_draw_threshold=0.35,
                cascade_calibration_factor=0.85,
                random_state=42
            )
            
            hybrid_model.fit(X_test, y_test)
            
            # Prédictions
            preds = hybrid_model.predict(X_test)
            probas = hybrid_model.predict_proba(X_test)
            info = hybrid_model.get_model_info(X_test)
            
            logger.info(f"   Prédictions: {preds[:5]}")
            logger.info(f"   Probabilités shape: {probas.shape}")
            logger.info(f"   Info modèles: {info}")
            
            # Vérification switch automatique
            early_samples = info['cascade_samples']
            late_samples = info['baseline_samples']
            
            logger.info(f"   Switch automatique: {early_samples} early, {late_samples} late")
            
            # Validations
            assert len(preds) == len(X_test), "Longueur prédictions incorrecte"
            assert probas.shape == (len(X_test), 3), "Shape probabilités incorrecte"
            assert np.allclose(probas.sum(axis=1), 1.0, rtol=1e-3), "Probabilités non normalisées"
            assert early_samples + late_samples == len(X_test), "Switch incomplet"
            
            logger.info("   ✅ Tests de base réussis")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Échec tests de base: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cross_validation_temporal(self, n_splits=5):
        """Cross-validation temporelle avec analyse par phase."""
        logger.info(f"\n📈 CROSS-VALIDATION TEMPORELLE ({n_splits} splits)")
        
        try:
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Résultats par fold
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
                logger.info(f"\n   📊 FOLD {fold+1}/{n_splits}")
                
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
                
                # 1. Modèle hybride
                hybrid_model = HybridModelClean(
                    early_season_threshold=0.15,
                    cascade_draw_weight=3.0,
                    cascade_draw_threshold=0.35,
                    cascade_calibration_factor=0.85,
                    random_state=42
                )
                hybrid_model.fit(X_train, y_train)
                
                # 2. Modèle baseline pour comparaison
                baseline_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42
                )
                baseline_model.fit(X_train, y_train)
                
                # Prédictions
                hybrid_preds = hybrid_model.predict(X_test)
                baseline_preds = baseline_model.predict(X_test)
                
                # Conversion pour métriques
                y_test_str = pd.Series(y_test).map({0: 'H', 1: 'D', 2: 'A'})
                
                # Métriques globales
                hybrid_acc = accuracy_score(y_test_str, hybrid_preds)
                baseline_acc = accuracy_score(y_test_str, baseline_preds)
                
                # Analyse par phase de saison
                test_matchdays = X_test['matchday_normalized']
                early_mask = test_matchdays <= 0.15
                
                early_results = None
                late_results = None
                
                if early_mask.sum() > 0:
                    # Performance early season
                    early_hybrid_acc = accuracy_score(y_test_str[early_mask], hybrid_preds[early_mask])
                    early_baseline_acc = accuracy_score(y_test_str[early_mask], baseline_preds[early_mask])
                    early_results = {
                        'samples': early_mask.sum(),
                        'hybrid_acc': early_hybrid_acc,
                        'baseline_acc': early_baseline_acc,
                        'boost': early_hybrid_acc - early_baseline_acc
                    }
                
                late_mask = ~early_mask
                if late_mask.sum() > 0:
                    # Performance late season
                    late_hybrid_acc = accuracy_score(y_test_str[late_mask], hybrid_preds[late_mask])
                    late_baseline_acc = accuracy_score(y_test_str[late_mask], baseline_preds[late_mask])
                    late_results = {
                        'samples': late_mask.sum(),
                        'hybrid_acc': late_hybrid_acc,
                        'baseline_acc': late_baseline_acc,
                        'boost': late_hybrid_acc - late_baseline_acc
                    }
                
                # Stockage résultats fold
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'hybrid_accuracy': hybrid_acc,
                    'baseline_accuracy': baseline_acc,
                    'accuracy_boost': hybrid_acc - baseline_acc,
                    'early_season': early_results,
                    'late_season': late_results
                }
                fold_results.append(fold_result)
                
                logger.info(f"     Global: Hybrid {hybrid_acc:.3f}, Baseline {baseline_acc:.3f} (Δ{hybrid_acc-baseline_acc:+.3f})")
                if early_results:
                    logger.info(f"     Early:  Hybrid {early_results['hybrid_acc']:.3f}, Baseline {early_results['baseline_acc']:.3f} (Δ{early_results['boost']:+.3f}) - {early_results['samples']} échantillons")
                if late_results:
                    logger.info(f"     Late:   Hybrid {late_results['hybrid_acc']:.3f}, Baseline {late_results['baseline_acc']:.3f} (Δ{late_results['boost']:+.3f}) - {late_results['samples']} échantillons")
            
            # Synthèse CV
            global_accuracies = [r['hybrid_accuracy'] for r in fold_results]
            global_boosts = [r['accuracy_boost'] for r in fold_results]
            
            early_boosts = [r['early_season']['boost'] for r in fold_results if r['early_season']]
            late_boosts = [r['late_season']['boost'] for r in fold_results if r['late_season']]
            
            cv_mean = np.mean(global_accuracies)
            cv_std = np.std(global_accuracies)
            boost_mean = np.mean(global_boosts)
            
            logger.info(f"\n   📊 RÉSULTATS CV GLOBAUX:")
            logger.info(f"   Hybrid CV: {cv_mean:.3f} ± {cv_std:.3f}")
            logger.info(f"   Boost moyen: {boost_mean:+.3f}")
            
            if early_boosts:
                early_boost_mean = np.mean(early_boosts)
                logger.info(f"   Boost early season: {early_boost_mean:+.3f} (sur {len(early_boosts)} folds)")
            
            if late_boosts:
                late_boost_mean = np.mean(late_boosts)
                logger.info(f"   Boost late season: {late_boost_mean:+.3f} (sur {len(late_boosts)} folds)")
            
            return {
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'boost_mean': boost_mean,
                'early_boost_mean': np.mean(early_boosts) if early_boosts else None,
                'late_boost_mean': np.mean(late_boosts) if late_boosts else None,
                'fold_results': fold_results
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur CV: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def performance_analysis_complete(self):
        """Analyse de performance complète avec détail par phase."""
        logger.info(f"\n📊 ANALYSE PERFORMANCE COMPLÈTE")
        
        try:
            # Split train/test pour analyse détaillée
            split_point = int(len(self.X) * 0.8)
            X_train = self.X[:split_point]
            X_test = self.X[split_point:]
            y_train = self.y[:split_point]
            y_test = self.y[split_point:]
            
            # Entraînement modèles
            hybrid_model = HybridModelClean(
                early_season_threshold=0.15,
                cascade_draw_weight=3.0,
                cascade_draw_threshold=0.35,
                cascade_calibration_factor=0.85,
                random_state=42
            )
            hybrid_model.fit(X_train, y_train)
            
            baseline_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42
            )
            baseline_model.fit(X_train, y_train)
            
            # Prédictions
            hybrid_preds = hybrid_model.predict(X_test)
            baseline_preds = baseline_model.predict(X_test)
            
            # Conversion targets
            y_test_str = pd.Series(y_test).map({0: 'H', 1: 'D', 2: 'A'})
            
            # Métriques globales
            hybrid_accuracy = accuracy_score(y_test_str, hybrid_preds)
            baseline_accuracy = accuracy_score(y_test_str, baseline_preds)
            
            logger.info(f"   Hybrid accuracy: {hybrid_accuracy:.3f}")
            logger.info(f"   Baseline accuracy: {baseline_accuracy:.3f}")
            logger.info(f"   Boost global: {hybrid_accuracy - baseline_accuracy:+.3f}")
            
            # Analyse par phase de saison
            test_matchdays = X_test['matchday_normalized']
            early_mask = test_matchdays <= 0.15
            late_mask = ~early_mask
            
            logger.info(f"\n   📈 ANALYSE PAR PHASE:")
            logger.info(f"   Early season: {early_mask.sum()} échantillons")
            logger.info(f"   Late season: {late_mask.sum()} échantillons")
            
            phase_results = {}
            
            if early_mask.sum() > 0:
                early_hybrid_acc = accuracy_score(y_test_str[early_mask], hybrid_preds[early_mask])
                early_baseline_acc = accuracy_score(y_test_str[early_mask], baseline_preds[early_mask])
                early_boost = early_hybrid_acc - early_baseline_acc
                
                logger.info(f"   Early: Hybrid {early_hybrid_acc:.3f}, Baseline {early_baseline_acc:.3f} (Δ{early_boost:+.3f})")
                
                phase_results['early'] = {
                    'hybrid_acc': early_hybrid_acc,
                    'baseline_acc': early_baseline_acc,
                    'boost': early_boost,
                    'samples': early_mask.sum()
                }
            
            if late_mask.sum() > 0:
                late_hybrid_acc = accuracy_score(y_test_str[late_mask], hybrid_preds[late_mask])
                late_baseline_acc = accuracy_score(y_test_str[late_mask], baseline_preds[late_mask])
                late_boost = late_hybrid_acc - late_baseline_acc
                
                logger.info(f"   Late:  Hybrid {late_hybrid_acc:.3f}, Baseline {late_baseline_acc:.3f} (Δ{late_boost:+.3f})")
                
                phase_results['late'] = {
                    'hybrid_acc': late_hybrid_acc,
                    'baseline_acc': late_baseline_acc,
                    'boost': late_boost,
                    'samples': late_mask.sum()
                }
            
            # Info utilisation modèles
            model_info = hybrid_model.get_model_info(X_test)
            logger.info(f"\n   🔧 UTILISATION MODÈLES:")
            logger.info(f"   Cascade: {model_info['cascade_samples']} échantillons ({model_info['cascade_ratio']:.1%})")
            logger.info(f"   Baseline: {model_info['baseline_samples']} échantillons ({1-model_info['cascade_ratio']:.1%})")
            
            return {
                'hybrid_accuracy': hybrid_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'global_boost': hybrid_accuracy - baseline_accuracy,
                'phase_results': phase_results,
                'model_usage': model_info
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def robustness_test(self, n_seeds=3):
        """Test de robustesse avec différents seeds."""
        logger.info(f"\n🔄 TEST ROBUSTESSE ({n_seeds} seeds)")
        
        try:
            seeds = [42, 123, 456][:n_seeds]
            seed_results = []
            
            # Split pour robustesse
            split_point = int(len(self.X) * 0.8)
            X_train = self.X[:split_point]
            X_test = self.X[split_point:]
            y_train = self.y[:split_point]
            y_test = self.y[split_point:]
            
            for seed in seeds:
                # Modèle hybride avec ce seed
                hybrid_model = HybridModelClean(
                    early_season_threshold=0.15,
                    cascade_draw_weight=3.0,
                    cascade_draw_threshold=0.35,
                    cascade_calibration_factor=0.85,
                    random_state=seed
                )
                hybrid_model.fit(X_train, y_train)
                
                # Prédictions
                test_preds = hybrid_model.predict(X_test)
                y_test_str = pd.Series(y_test).map({0: 'H', 1: 'D', 2: 'A'})
                
                # Accuracy
                acc = accuracy_score(y_test_str, test_preds)
                seed_results.append(acc)
                
                logger.info(f"   Seed {seed}: {acc:.3f} accuracy")
            
            # Analyse robustesse
            mean_robust = np.mean(seed_results)
            variance = np.var(seed_results)
            std_robust = np.std(seed_results)
            
            logger.info(f"\n   📊 ROBUSTESSE:")
            logger.info(f"   Accuracy moyenne: {mean_robust:.3f}")
            logger.info(f"   Écart-type: {std_robust:.3f}")
            logger.info(f"   Variance: {variance:.6f}")
            
            # Critère robustesse
            robust = variance < 0.001
            logger.info(f"   Robuste: {'✅' if robust else '❌'}")
            
            return {
                'robust_accuracy_mean': mean_robust,
                'robust_std': std_robust,
                'robust_variance': variance,
                'is_robust': robust,
                'seed_accuracies': seed_results
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur robustesse: {e}")
            return None
    
    def full_audit(self):
        """Audit complet du modèle hybride."""
        logger.info("🔍 AUDIT COMPLET MODÈLE HYBRIDE")
        logger.info("=" * 50)
        
        audit_results = {}
        
        # 1. Chargement
        if not self.load_data():
            return None
        
        # 2. Tests de base
        if not self.test_basic_functionality():
            return None
        
        # 3. Cross-validation
        cv_results = self.cross_validation_temporal()
        if cv_results:
            audit_results.update(cv_results)
        
        # 4. Performance détaillée
        perf_results = self.performance_analysis_complete()
        if perf_results:
            audit_results.update(perf_results)
        
        # 5. Robustesse
        robust_results = self.robustness_test()
        if robust_results:
            audit_results.update(robust_results)
        
        # 6. Verdict final
        success = self.audit_summary(audit_results)
        
        return audit_results if success else None
    
    def audit_summary(self, results):
        """Résumé et verdict de l'audit."""
        logger.info(f"\n🏆 RÉSUMÉ AUDIT HYBRIDE")
        logger.info("=" * 30)
        
        # Extraction métriques
        cv_acc = results.get('cv_accuracy_mean', 0)
        cv_std = results.get('cv_accuracy_std', 999)
        global_boost = results.get('global_boost', 0)
        early_boost = results.get('early_boost_mean', 0)
        robust = results.get('is_robust', False)
        robust_var = results.get('robust_variance', 999)
        
        logger.info(f"📊 MÉTRIQUES CLÉS:")
        logger.info(f"   CV Accuracy: {cv_acc:.3f} ± {cv_std:.3f}")
        logger.info(f"   Boost global: {global_boost:+.3f}")
        logger.info(f"   Boost early season: {early_boost:+.3f}" if early_boost else "   Boost early season: N/A")
        logger.info(f"   Robuste: {'✅' if robust else '❌'} (variance: {robust_var:.6f})")
        
        # Critères de validation
        cv_criteria = cv_acc > 0.50  # > 50% CV accuracy
        variance_criteria = cv_std < 0.05  # Variance < 5%
        robustness_criteria = robust  # Variance seeds < 0.1%
        boost_criteria = global_boost > 0.01  # Amélioration > 1%
        
        logger.info(f"\n🎯 CRITÈRES VALIDATION:")
        logger.info(f"   CV > 50%: {'✅' if cv_criteria else '❌'} ({cv_acc:.1%})")
        logger.info(f"   Variance < 5%: {'✅' if variance_criteria else '❌'} ({cv_std:.1%})")
        logger.info(f"   Robuste: {'✅' if robustness_criteria else '❌'}")
        logger.info(f"   Boost > 1%: {'✅' if boost_criteria else '❌'} ({global_boost:+.1%})")
        
        # Verdict final
        validation_score = sum([cv_criteria, variance_criteria, robustness_criteria, boost_criteria])
        production_ready = validation_score >= 3  # Au moins 3/4 critères
        
        if production_ready:
            logger.info(f"\n✅ MODÈLE HYBRIDE VALIDÉ POUR PRODUCTION")
            logger.info(f"   Score validation: {validation_score}/4")
            print(f"\n🚀 VALIDATION HYBRIDE RÉUSSIE!")
            print(f"   CV Accuracy: {cv_acc:.1%} ± {cv_std:.1%}")
            print(f"   Boost vs baseline: {global_boost:+.1%}")
            print(f"   Status: ✅ PRODUCTION READY")
        else:
            logger.info(f"\n❌ MODÈLE HYBRIDE NON VALIDÉ")
            logger.info(f"   Score validation: {validation_score}/4")
            print(f"\n⚠️  VALIDATION HYBRIDE ÉCHOUÉE")
            print(f"   Score: {validation_score}/4 critères")
            print(f"   Recommandation: Rollback baseline v2.3")
        
        return production_ready

def main():
    """Audit principal."""
    dataset_path = "data/processed/v15_final_enhanced.csv"
    
    auditor = HybridAuditor(dataset_path)
    results = auditor.full_audit()
    
    return results is not None

if __name__ == "__main__":
    success = main()