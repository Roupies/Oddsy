#!/usr/bin/env python3
"""
⚡ CARTOGRAPHIE TEMPORELLE CASCADE
================================
Audit rapide pour identifier la zone d'efficacité optimale du cascade.
Test fenêtres progressives: J1-J4, J5-J8, J9-J12, J13+ vs baseline.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import logging

# Import modèles
sys.path.append('scripts/final')
from cascade_model_production import CascadeModelProduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("temporal_mapping")

class CascadeTemporalMapper:
    """Cartographe la performance cascade par fenêtre temporelle."""
    
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
        
        # Définition des fenêtres temporelles
        self.temporal_windows = {
            'J1-J4': (0.0, 0.15),    # Sweet spot confirmé
            'J5-J8': (0.15, 0.30),   # Extension possible ?
            'J9-J12': (0.30, 0.45),  # Milieu saison
            'J13-J20': (0.45, 0.65), # Fin première partie
            'J21+': (0.65, 1.0)      # Fin de saison
        }
        
    def load_data(self):
        """Chargement et préparation des données."""
        try:
            logger.info("📊 CHARGEMENT DONNÉES CARTOGRAPHIE")
            
            self.data = pd.read_csv(self.dataset_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Création target
            target_mapping = {'H': 0, 'D': 1, 'A': 2}
            self.data['target'] = self.data['FullTimeResult'].map(target_mapping)
            
            # Filtrage et tri
            valid_mask = self.data['target'].notna()
            self.data = self.data[valid_mask].sort_values('Date').reset_index(drop=True)
            
            # Features
            self.X = self.data[self.features].fillna(0)
            self.y = self.data['target'].astype(int)
            
            logger.info(f"   Dataset: {len(self.data)} échantillons")
            logger.info(f"   Fenêtres à tester: {list(self.temporal_windows.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return False
    
    def test_window_performance(self, window_name, matchday_range, cv_splits=3):
        """
        Test performance cascade vs baseline sur une fenêtre temporelle.
        
        Args:
            window_name: Nom de la fenêtre (ex: 'J1-J4')
            matchday_range: Tuple (min_matchday, max_matchday)
            cv_splits: Nombre de splits pour CV
            
        Returns:
            dict: Résultats détaillés de la fenêtre
        """
        min_matchday, max_matchday = matchday_range
        
        logger.info(f"\n🎯 TEST FENÊTRE {window_name}")
        logger.info(f"   Matchday range: {min_matchday:.2f} - {max_matchday:.2f}")
        
        try:
            # Filtrage données dans la fenêtre temporelle
            matchday_col = self.X['matchday_normalized']
            window_mask = (matchday_col >= min_matchday) & (matchday_col < max_matchday)
            
            if window_mask.sum() < 50:  # Pas assez d'échantillons
                logger.warning(f"   ⚠️  Trop peu d'échantillons: {window_mask.sum()}")
                return None
            
            logger.info(f"   Échantillons dans fenêtre: {window_mask.sum()}")
            
            # Cross-validation sur échantillons de la fenêtre
            X_window = self.X[window_mask]
            y_window = self.y[window_mask]
            
            # Indices temporels pour respecter l'ordre
            window_indices = np.where(window_mask)[0]
            
            # Splits temporels
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            
            cascade_scores = []
            baseline_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_window)):
                # Indices absolus pour entraînement global
                global_train_mask = np.zeros(len(self.X), dtype=bool)
                
                # Prendre historique complet avant la fenêtre de test
                test_window_start = window_indices[test_idx[0]]
                global_train_mask[:test_window_start] = True
                
                # Données d'entraînement et test
                X_train_global = self.X[global_train_mask]
                y_train_global = self.y[global_train_mask]
                X_test_window = X_window.iloc[test_idx]
                y_test_window = y_window.iloc[test_idx]
                
                if len(X_train_global) < 100:  # Pas assez d'historique
                    continue
                
                # 1. Modèle cascade
                cascade_model = CascadeModelProduction(
                    draw_weight=3.0,
                    draw_threshold=0.35,
                    calibration_factor=0.85,
                    random_state=42
                )
                cascade_model.fit(X_train_global, y_train_global)
                cascade_preds = cascade_model.predict(X_test_window)
                
                # 2. Modèle baseline
                baseline_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42
                )
                baseline_model.fit(X_train_global, y_train_global)
                baseline_preds = baseline_model.predict(X_test_window)
                
                # Conversion pour métriques
                y_test_str = pd.Series(y_test_window).map({0: 'H', 1: 'D', 2: 'A'})
                
                # Conversion prédictions baseline si nécessaire
                if hasattr(baseline_preds[0], 'dtype') and np.issubdtype(baseline_preds.dtype, np.integer):
                    baseline_preds_str = pd.Series(baseline_preds).map({0: 'H', 1: 'D', 2: 'A'}).values
                else:
                    baseline_preds_str = baseline_preds
                
                # Accuracy
                cascade_acc = accuracy_score(y_test_str, cascade_preds)
                baseline_acc = accuracy_score(y_test_str, baseline_preds_str)
                
                cascade_scores.append(cascade_acc)
                baseline_scores.append(baseline_acc)
                
                logger.info(f"     Fold {fold+1}: Cascade {cascade_acc:.3f}, Baseline {baseline_acc:.3f} (Δ{cascade_acc-baseline_acc:+.3f})")
            
            if not cascade_scores:  # Aucun fold valide
                logger.warning(f"   ⚠️  Aucun fold valide pour {window_name}")
                return None
            
            # Statistiques de la fenêtre
            cascade_mean = np.mean(cascade_scores)
            baseline_mean = np.mean(baseline_scores)
            boost_mean = cascade_mean - baseline_mean
            boost_std = np.std([c - b for c, b in zip(cascade_scores, baseline_scores)])
            
            # Significativité (test t simple)
            import scipy.stats as stats
            if len(cascade_scores) > 1:
                t_stat, p_value = stats.ttest_rel(cascade_scores, baseline_scores)
                significant = p_value < 0.05
            else:
                significant = abs(boost_mean) > 0.02  # Seuil arbitraire
                p_value = None
            
            # Distribution prédictions
            # Analyse rapide sur dernier fold
            if len(cascade_preds) > 0:
                cascade_dist = pd.Series(cascade_preds).value_counts(normalize=True).sort_index() * 100
                baseline_dist = pd.Series(baseline_preds_str).value_counts(normalize=True).sort_index() * 100
            else:
                cascade_dist = baseline_dist = pd.Series()
            
            result = {
                'window': window_name,
                'matchday_range': matchday_range,
                'samples': window_mask.sum(),
                'valid_folds': len(cascade_scores),
                'cascade_accuracy': cascade_mean,
                'baseline_accuracy': baseline_mean,
                'boost_mean': boost_mean,
                'boost_std': boost_std,
                'significant': significant,
                'p_value': p_value,
                'cascade_distribution': cascade_dist.to_dict() if not cascade_dist.empty else {},
                'baseline_distribution': baseline_dist.to_dict() if not baseline_dist.empty else {}
            }
            
            # Verdict fenêtre
            if boost_mean > 0.02 and significant:
                verdict = "✅ AVANTAGEUX"
            elif boost_mean > 0.01:
                verdict = "⚠️  MARGINAL"
            else:
                verdict = "❌ DÉSAVANTAGEUX"
            
            logger.info(f"   📊 RÉSULTAT {window_name}:")
            logger.info(f"     Cascade: {cascade_mean:.3f}, Baseline: {baseline_mean:.3f}")
            logger.info(f"     Boost: {boost_mean:+.3f} ± {boost_std:.3f}")
            logger.info(f"     Significatif: {'Oui' if significant else 'Non'}")
            logger.info(f"     🎯 VERDICT: {verdict}")
            
            result['verdict'] = verdict
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur test {window_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def map_temporal_zones(self):
        """Cartographie complète des zones temporelles."""
        logger.info("⚡ CARTOGRAPHIE ZONES TEMPORELLES CASCADE")
        logger.info("=" * 50)
        
        if not self.load_data():
            return None
        
        # Test de toutes les fenêtres
        results = {}
        
        for window_name, matchday_range in self.temporal_windows.items():
            result = self.test_window_performance(window_name, matchday_range)
            if result:
                results[window_name] = result
        
        # Synthèse globale
        self.analyze_temporal_strategy(results)
        
        return results
    
    def analyze_temporal_strategy(self, results):
        """Analyse et recommandation de stratégie temporelle."""
        logger.info(f"\n🗺️  SYNTHÈSE CARTOGRAPHIE TEMPORELLE")
        logger.info("=" * 40)
        
        # Classification des fenêtres
        advantageous_windows = []
        marginal_windows = []
        disadvantageous_windows = []
        
        for window, result in results.items():
            if result['verdict'] == "✅ AVANTAGEUX":
                advantageous_windows.append((window, result['boost_mean']))
            elif result['verdict'] == "⚠️  MARGINAL":
                marginal_windows.append((window, result['boost_mean']))
            else:
                disadvantageous_windows.append((window, result['boost_mean']))
        
        # Affichage par catégorie
        logger.info(f"📈 FENÊTRES AVANTAGEUSES:")
        if advantageous_windows:
            for window, boost in sorted(advantageous_windows, key=lambda x: x[1], reverse=True):
                logger.info(f"   {window}: {boost:+.3f} boost")
        else:
            logger.info(f"   Aucune")
        
        logger.info(f"\n⚠️  FENÊTRES MARGINALES:")
        if marginal_windows:
            for window, boost in sorted(marginal_windows, key=lambda x: x[1], reverse=True):
                logger.info(f"   {window}: {boost:+.3f} boost")
        else:
            logger.info(f"   Aucune")
        
        logger.info(f"\n❌ FENÊTRES DÉSAVANTAGEUSES:")
        if disadvantageous_windows:
            for window, boost in sorted(disadvantageous_windows, key=lambda x: x[1], reverse=True):
                logger.info(f"   {window}: {boost:+.3f} boost")
        else:
            logger.info(f"   Aucune")
        
        # Recommandation stratégique
        logger.info(f"\n🎯 RECOMMANDATION STRATÉGIQUE:")
        
        if len(advantageous_windows) == 1 and advantageous_windows[0][0] == 'J1-J4':
            strategy = "HYBRIDE J1-J4 SEULEMENT"
            explanation = "Cascade efficace uniquement en début de saison. Garder switch à J4."
        elif len(advantageous_windows) >= 2:
            max_window = max([w for w, _ in advantageous_windows])
            strategy = f"HYBRIDE ÉTENDU JUSQU'À {max_window}"
            explanation = f"Cascade avantageux sur {len(advantageous_windows)} fenêtres. Étendre le switch."
        elif marginal_windows:
            strategy = "HYBRIDE PONDÉRÉ"
            explanation = "Effets marginaux détectés. Considérer pondération hybride (50/50) sur fenêtres marginales."
        else:
            strategy = "BASELINE COMPLET"
            explanation = "Aucun avantage cascade détecté. Revenir au modèle baseline v2.3."
        
        logger.info(f"   📋 STRATÉGIE: {strategy}")
        logger.info(f"   📝 EXPLICATION: {explanation}")
        
        # Tableau résumé pour l'utilisateur
        print(f"\n⚡ CARTOGRAPHIE TEMPORELLE CASCADE TERMINÉE")
        print(f"\n{'Fenêtre':<10} {'Boost':<8} {'Verdict':<15} {'Échantillons':<12}")
        print(f"{'='*10} {'='*8} {'='*15} {'='*12}")
        
        for window, result in results.items():
            boost_str = f"{result['boost_mean']:+.3f}"
            verdict_short = result['verdict'].split()[1] if len(result['verdict'].split()) > 1 else result['verdict']
            samples_str = str(result['samples'])
            print(f"{window:<10} {boost_str:<8} {verdict_short:<15} {samples_str:<12}")
        
        print(f"\n🎯 RECOMMANDATION: {strategy}")
        
        return {
            'strategy': strategy,
            'explanation': explanation,
            'advantageous_windows': advantageous_windows,
            'marginal_windows': marginal_windows,
            'window_results': results
        }

def main():
    """Cartographie principale."""
    dataset_path = "data/processed/v_auto_update_20250916_110247.csv"
    
    mapper = CascadeTemporalMapper(dataset_path)
    results = mapper.map_temporal_zones()
    
    return results

if __name__ == "__main__":
    results = main()