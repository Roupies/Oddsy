#!/usr/bin/env python3
"""
🔍 ANALYSE SIGNAL CASCADE
========================
Protocole d'audit ciblé pour déterminer si le signal 52.5% cascade 
sur EPL 2025-26 est un pattern exploitable ou un artefact.

3 TESTS DÉCISIONNELS:
1. Backtest J1-J4 saisons précédentes  
2. Analyse distribution temporelle draws/H/A
3. Feature importance cascade vs baseline
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import logging

# Import modèles
sys.path.append('scripts/final')
from cascade_model_production import CascadeModelProduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal_analysis")

class CascadeSignalAnalyzer:
    """Analyseur pour déterminer si le signal cascade est exploitable."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
    def load_data(self):
        """Chargement et préparation des données."""
        try:
            logger.info("📊 CHARGEMENT DONNÉES POUR ANALYSE SIGNAL")
            
            self.data = pd.read_csv(self.dataset_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Création target
            target_mapping = {'H': 0, 'D': 1, 'A': 2}
            self.data['target'] = self.data['FullTimeResult'].map(target_mapping)
            
            # Filtrage données valides
            valid_mask = self.data['target'].notna()
            self.data = self.data[valid_mask].reset_index(drop=True)
            
            # Tri par date
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"   Dataset: {len(self.data)} matchs")
            logger.info(f"   Période: {self.data['Date'].min()} → {self.data['Date'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return False
    
    def test_1_backtest_early_season(self):
        """
        TEST 1: Backtest J1-J4 des saisons précédentes
        
        Objective: Vérifier si le boost cascade sur J1-J4 EPL 2025-26 
        se reproduit sur les débuts de saisons précédentes.
        """
        logger.info("\n🧪 TEST 1: BACKTEST J1-J4 SAISONS PRÉCÉDENTES")
        logger.info("=" * 50)
        
        results = {}
        
        try:
            # Identification des saisons
            self.data['Season_Year'] = self.data['Season'].str[:4].astype(int)
            seasons = sorted(self.data['Season_Year'].unique())
            
            logger.info(f"   Saisons disponibles: {seasons}")
            
            # Pour chaque saison sauf la dernière (test)
            for season in seasons[:-1]:  # Exclure 2025 (notre test)
                season_data = self.data[self.data['Season_Year'] == season].copy()
                
                if len(season_data) < 50:  # Pas assez de données
                    continue
                
                # Simulation J1-J4: premiers 40 matchs de la saison
                early_season = season_data.head(40)
                historical_train = self.data[
                    (self.data['Season_Year'] < season) & 
                    (self.data['Season_Year'] >= season - 3)  # 3 ans d'historique
                ].copy()
                
                if len(historical_train) < 500:  # Pas assez d'historique
                    continue
                
                # Préparation données
                X_train = historical_train[self.features].fillna(0)
                y_train = historical_train['target']
                X_test = early_season[self.features].fillna(0)
                y_test = early_season['target']
                
                # Modèle baseline
                baseline_model = RandomForestClassifier(n_estimators=150, random_state=42)
                baseline_model.fit(X_train, y_train)
                baseline_preds = baseline_model.predict(X_test)
                
                # Modèle cascade
                cascade_model = CascadeModelProduction(
                    draw_weight=3.0,
                    draw_threshold=0.35,
                    calibration_factor=0.85,
                    random_state=42
                )
                cascade_model.fit(X_train, y_train)
                cascade_preds = cascade_model.predict(X_test)
                
                # Conversion pour métriques
                y_test_str = y_test.map({0: 'H', 1: 'D', 2: 'A'})
                
                # Accuracy
                baseline_acc = accuracy_score(y_test_str, baseline_preds)
                cascade_acc = accuracy_score(y_test_str, cascade_preds)
                
                # Détection draws
                baseline_draws = (baseline_preds == 'D').sum()
                cascade_draws = (cascade_preds == 'D').sum()
                real_draws = (y_test_str == 'D').sum()
                
                results[season] = {
                    'baseline_accuracy': baseline_acc,
                    'cascade_accuracy': cascade_acc,
                    'accuracy_boost': cascade_acc - baseline_acc,
                    'baseline_draws': baseline_draws,
                    'cascade_draws': cascade_draws,
                    'real_draws': real_draws,
                    'total_matches': len(y_test)
                }
                
                logger.info(f"   {season}: Baseline {baseline_acc:.3f}, Cascade {cascade_acc:.3f} (Δ{cascade_acc-baseline_acc:+.3f})")
            
            # Analyse des résultats
            if results:
                boosts = [r['accuracy_boost'] for r in results.values()]
                avg_boost = np.mean(boosts)
                std_boost = np.std(boosts)
                positive_boosts = len([b for b in boosts if b > 0])
                
                logger.info(f"\n   📊 SYNTHÈSE BACKTEST:")
                logger.info(f"   Boost moyen: {avg_boost:+.3f} ± {std_boost:.3f}")
                logger.info(f"   Boosts positifs: {positive_boosts}/{len(boosts)} saisons")
                logger.info(f"   Range: {min(boosts):+.3f} → {max(boosts):+.3f}")
                
                # Verdict
                consistent_boost = avg_boost > 0.02 and positive_boosts >= len(boosts) * 0.7
                verdict = "✅ PATTERN RÉCURRENT" if consistent_boost else "❌ EFFET SPÉCIFIQUE 2025-26"
                
                logger.info(f"   🎯 VERDICT: {verdict}")
                
                return {
                    'pattern_detected': consistent_boost,
                    'avg_boost': avg_boost,
                    'consistency': positive_boosts / len(boosts),
                    'details': results
                }
            else:
                logger.warning("   ⚠️  Pas assez de données pour backtest")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur test 1: {e}")
            return None
    
    def test_2_temporal_distribution_analysis(self):
        """
        TEST 2: Analyse distribution temporelle H/D/A
        
        Objective: Vérifier si EPL 2025-26 a une distribution atypique
        qui expliquerait le boost cascade.
        """
        logger.info("\n🧪 TEST 2: ANALYSE DISTRIBUTION TEMPORELLE")
        logger.info("=" * 45)
        
        try:
            # Distribution par saison
            season_stats = []
            
            for season in self.data['Season'].unique():
                season_data = self.data[self.data['Season'] == season]
                
                if len(season_data) < 10:
                    continue
                
                dist = season_data['FullTimeResult'].value_counts(normalize=True).sort_index()
                
                # Début de saison (premiers 40 matchs)
                early_season = season_data.head(40)
                early_dist = early_season['FullTimeResult'].value_counts(normalize=True).sort_index() if len(early_season) >= 20 else None
                
                stats = {
                    'season': season,
                    'total_matches': len(season_data),
                    'H_pct': dist.get('H', 0) * 100,
                    'D_pct': dist.get('D', 0) * 100,
                    'A_pct': dist.get('A', 0) * 100
                }
                
                if early_dist is not None:
                    stats.update({
                        'early_H_pct': early_dist.get('H', 0) * 100,
                        'early_D_pct': early_dist.get('D', 0) * 100,
                        'early_A_pct': early_dist.get('A', 0) * 100,
                        'early_matches': len(early_season)
                    })
                
                season_stats.append(stats)
            
            df_stats = pd.DataFrame(season_stats)
            
            # Statistiques globales
            logger.info(f"   📊 DISTRIBUTION HISTORIQUE (moyenne):")
            logger.info(f"   H: {df_stats['H_pct'].mean():.1f}% ± {df_stats['H_pct'].std():.1f}%")
            logger.info(f"   D: {df_stats['D_pct'].mean():.1f}% ± {df_stats['D_pct'].std():.1f}%")
            logger.info(f"   A: {df_stats['A_pct'].mean():.1f}% ± {df_stats['A_pct'].std():.1f}%")
            
            # Focus sur 2025-26
            epl_2025_26 = df_stats[df_stats['season'].str.contains('2025', na=False)]
            
            if not epl_2025_26.empty:
                current = epl_2025_26.iloc[0]
                logger.info(f"\n   📊 EPL 2025-26 vs MOYENNE:")
                
                h_diff = current['H_pct'] - df_stats['H_pct'].mean()
                d_diff = current['D_pct'] - df_stats['D_pct'].mean()
                a_diff = current['A_pct'] - df_stats['A_pct'].mean()
                
                logger.info(f"   H: {current['H_pct']:.1f}% (Δ{h_diff:+.1f}%)")
                logger.info(f"   D: {current['D_pct']:.1f}% (Δ{d_diff:+.1f}%)")
                logger.info(f"   A: {current['A_pct']:.1f}% (Δ{a_diff:+.1f}%)")
                
                # Early season si disponible
                if 'early_D_pct' in current and pd.notna(current['early_D_pct']):
                    early_d_avg = df_stats['early_D_pct'].mean()
                    early_d_diff = current['early_D_pct'] - early_d_avg
                    
                    logger.info(f"\n   📊 DÉBUT SAISON 2025-26:")
                    logger.info(f"   Draws early: {current['early_D_pct']:.1f}% vs {early_d_avg:.1f}% moyen (Δ{early_d_diff:+.1f}%)")
                
                # Détection d'anomalie
                d_zscore = abs(d_diff) / df_stats['D_pct'].std() if df_stats['D_pct'].std() > 0 else 0
                anomaly_detected = d_zscore > 2.0  # Plus de 2 écarts-types
                
                verdict = "⚠️ DISTRIBUTION ATYPIQUE" if anomaly_detected else "✅ DISTRIBUTION NORMALE"
                logger.info(f"   🎯 VERDICT: {verdict} (Z-score draws: {d_zscore:.1f})")
                
                return {
                    'anomaly_detected': anomaly_detected,
                    'draw_zscore': d_zscore,
                    'current_stats': current.to_dict(),
                    'historical_avg': {
                        'H_pct': df_stats['H_pct'].mean(),
                        'D_pct': df_stats['D_pct'].mean(),
                        'A_pct': df_stats['A_pct'].mean()
                    }
                }
            else:
                logger.warning("   ⚠️  Données 2025-26 non trouvées")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur test 2: {e}")
            return None
    
    def test_3_feature_importance_comparison(self):
        """
        TEST 3: Comparaison importance features cascade vs baseline
        
        Objective: Identifier quelles features donnent l'avantage cascade
        sur EPL 2025-26.
        """
        logger.info("\n🧪 TEST 3: COMPARAISON FEATURE IMPORTANCE")
        logger.info("=" * 45)
        
        try:
            # Données EPL 2025-26
            epl_2025_data = self.data[self.data['Date'] >= '2025-08-01'].head(40)
            train_data = self.data[self.data['Date'] < '2025-08-01']
            
            if len(train_data) < 1000 or len(epl_2025_data) < 20:
                logger.error("   ❌ Pas assez de données pour analyse")
                return None
            
            X_train = train_data[self.features].fillna(0)
            y_train = train_data['target']
            X_test = epl_2025_data[self.features].fillna(0)
            y_test = epl_2025_data['target']
            
            # 1. Modèle baseline
            baseline_model = RandomForestClassifier(n_estimators=150, random_state=42)
            baseline_model.fit(X_train, y_train)
            baseline_importance = baseline_model.feature_importances_
            
            # 2. Modèle cascade - analyse des sous-modèles
            cascade_model = CascadeModelProduction(
                draw_weight=3.0,
                draw_threshold=0.35,
                calibration_factor=0.85,
                random_state=42
            )
            cascade_model.fit(X_train, y_train)
            
            # Feature importance du draw classifier
            draw_importance = cascade_model.clf_draw.feature_importances_
            homeaway_importance = cascade_model.clf_homeaway.feature_importances_
            
            # 3. Analyse différentielle
            logger.info(f"   📊 TOP 5 FEATURES PAR MODÈLE:")
            
            # Baseline
            baseline_ranking = sorted(zip(self.features, baseline_importance), key=lambda x: x[1], reverse=True)
            logger.info(f"\n   🔥 BASELINE:")
            for i, (feature, importance) in enumerate(baseline_ranking[:5]):
                logger.info(f"   {i+1}. {feature}: {importance:.3f}")
            
            # Draw model
            draw_ranking = sorted(zip(self.features, draw_importance), key=lambda x: x[1], reverse=True)
            logger.info(f"\n   🎯 CASCADE DRAW:")
            for i, (feature, importance) in enumerate(draw_ranking[:5]):
                logger.info(f"   {i+1}. {feature}: {importance:.3f}")
            
            # Home/Away model
            homeaway_ranking = sorted(zip(self.features, homeaway_importance), key=lambda x: x[1], reverse=True)
            logger.info(f"\n   ⚖️  CASCADE HOME/AWAY:")
            for i, (feature, importance) in enumerate(homeaway_ranking[:5]):
                logger.info(f"   {i+1}. {feature}: {importance:.3f}")
            
            # 4. Différences significatives
            logger.info(f"\n   📈 DIFFÉRENCES SIGNIFICATIVES:")
            
            importance_diff = []
            for i, feature in enumerate(self.features):
                baseline_imp = baseline_importance[i]
                draw_imp = draw_importance[i]
                diff = abs(draw_imp - baseline_imp)
                
                importance_diff.append({
                    'feature': feature,
                    'baseline': baseline_imp,
                    'cascade_draw': draw_imp,
                    'diff': diff
                })
            
            # Tri par différence
            importance_diff.sort(key=lambda x: x['diff'], reverse=True)
            
            for item in importance_diff[:3]:  # Top 3 différences
                direction = "↗️" if item['cascade_draw'] > item['baseline'] else "↘️"
                logger.info(f"   {direction} {item['feature']}: {item['diff']:.3f} écart")
            
            # 5. Features clés cascade
            key_cascade_features = [item['feature'] for item in draw_ranking[:3]]
            logger.info(f"\n   🔑 FEATURES CLÉS CASCADE: {key_cascade_features}")
            
            # Verdict
            significant_differences = len([x for x in importance_diff if x['diff'] > 0.05])
            has_distinct_pattern = significant_differences >= 2
            
            verdict = "✅ PATTERN FEATURE DISTINCT" if has_distinct_pattern else "❌ FEATURES SIMILAIRES"
            logger.info(f"   🎯 VERDICT: {verdict}")
            
            return {
                'distinct_pattern': has_distinct_pattern,
                'key_features': key_cascade_features,
                'significant_differences': significant_differences,
                'importance_analysis': importance_diff
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur test 3: {e}")
            return None
    
    def run_full_analysis(self):
        """Exécution complète des 3 tests et verdict final."""
        logger.info("🔍 ANALYSE COMPLÈTE SIGNAL CASCADE")
        logger.info("=" * 60)
        
        if not self.load_data():
            return None
        
        # Exécution des 3 tests
        test1_result = self.test_1_backtest_early_season()
        test2_result = self.test_2_temporal_distribution_analysis()
        test3_result = self.test_3_feature_importance_comparison()
        
        # Verdict final
        logger.info(f"\n🏆 VERDICT FINAL")
        logger.info("=" * 20)
        
        scores = []
        
        # Test 1: Pattern récurrent
        if test1_result and test1_result['pattern_detected']:
            scores.append(1)
            logger.info("✅ Test 1: Pattern détecté sur historique")
        else:
            scores.append(0)
            logger.info("❌ Test 1: Pas de pattern historique")
        
        # Test 2: Distribution normale
        if test2_result and not test2_result['anomaly_detected']:
            scores.append(1)
            logger.info("✅ Test 2: Distribution EPL 2025-26 normale")
        else:
            scores.append(0)
            logger.info("❌ Test 2: Distribution EPL 2025-26 atypique")
        
        # Test 3: Pattern feature distinct
        if test3_result and test3_result['distinct_pattern']:
            scores.append(1)
            logger.info("✅ Test 3: Pattern feature cascade distinct")
        else:
            scores.append(0)
            logger.info("❌ Test 3: Features cascade similaires baseline")
        
        # Score final
        final_score = sum(scores)
        total_tests = len(scores)
        
        logger.info(f"\n📊 SCORE: {final_score}/{total_tests}")
        
        if final_score >= 2:
            recommendation = "🚀 CONTINUER CASCADE v2"
            action = "Le signal cascade montre un pattern exploitable. Recommandation: itérer vers cascade v2 robuste."
        elif final_score == 1:
            recommendation = "⚠️ ANALYSE APPROFONDIE"
            action = "Signal mitigé. Analyser plus en détail avant de décider."
        else:
            recommendation = "🔄 ROLLBACK BASELINE"
            action = "Signal cascade = artefact. Recommandation: revenir au modèle v2.3 baseline."
        
        logger.info(f"🎯 RECOMMANDATION: {recommendation}")
        logger.info(f"📋 ACTION: {action}")
        
        print(f"\n🔍 ANALYSE SIGNAL CASCADE TERMINÉE")
        print(f"Score: {final_score}/{total_tests}")
        print(f"Recommandation: {recommendation}")
        
        return {
            'score': final_score,
            'total_tests': total_tests,
            'recommendation': recommendation,
            'action': action,
            'test_results': {
                'backtest': test1_result,
                'distribution': test2_result,
                'features': test3_result
            }
        }

def main():
    """Analyse principale."""
    dataset_path = "data/processed/v15_final_enhanced.csv"
    
    analyzer = CascadeSignalAnalyzer(dataset_path)
    results = analyzer.run_full_analysis()
    
    return results

if __name__ == "__main__":
    results = main()