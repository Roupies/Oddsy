#!/usr/bin/env python3
"""
🎯 TEST 40 MATCHS COMPLETS J1-J4 EPL 2025-26
==========================================

Test sur les 40 premiers matchs candidats EPL 2025-26 du dataset v16,
indépendamment des vrais résultats disponibles.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_40_complets")

class HybridCascade40:
    """Cascade hybride pour test 40 matchs complets"""
    
    def __init__(self, draw_weight=2.5, draw_threshold=0.40, max_draw_ratio=0.20):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight={0: 1, 1: draw_weight}, random_state=42
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )
        self.draw_threshold = draw_threshold
        self.max_draw_ratio = max_draw_ratio
    
    def fit(self, X, y):
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'}) 
        else:
            y_str = y.copy()
        
        # Draw Forest
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X, y_draw)
        
        # Home/Away Forest
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            valid_homeaway = y_homeaway.notna()
            
            if valid_homeaway.sum() > 5:
                self.clf_homeaway.fit(X_notdraw[valid_homeaway], y_homeaway[valid_homeaway])
        
        return self
    
    def predict(self, X):
        proba_draw = self.clf_draw.predict_proba(X)[:, 1]
        pred_draw = (proba_draw > self.draw_threshold).astype(int)
        
        # Limitation draws
        n_draws_target = int(len(X) * self.max_draw_ratio)
        if pred_draw.sum() > n_draws_target:
            top_indices = np.argsort(proba_draw)[-n_draws_target:]
            pred_draw_filtered = np.zeros_like(pred_draw)
            pred_draw_filtered[top_indices] = 1
            pred_draw = pred_draw_filtered
        
        pred_homeaway = self.clf_homeaway.predict(X)
        
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            y_pred.append('D' if is_draw == 1 else ('H' if home_away == 1 else 'A'))
        
        return np.array(y_pred)

def test_40_matchs_complets():
    """Test complet sur 40 matchs J1-J4"""
    logger.info("🎯 TEST 40 MATCHS COMPLETS J1-J4 EPL 2025-26")
    logger.info("=" * 50)
    
    try:
        # Dataset v16
        df = pd.read_csv('data/processed/v16_specialized_features_enhanced.csv', parse_dates=['Date'])
        
        # Target encoding
        if 'target' not in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Split temporel
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        
        # 40 premiers matchs 2025-26 (candidats complets)
        df_season_2025 = df[df['Date'] >= '2025-08-01'].copy().sort_values('Date')
        df_test_40 = df_season_2025.head(40).copy()
        
        logger.info(f"📊 Train: {len(df_train)}, Test: {len(df_test_40)} matchs")
        logger.info(f"📅 Test période: {df_test_40['Date'].min()} à {df_test_40['Date'].max()}")
        
        # Features hybrides (10 + 2)
        base_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        selected_specialized = ['elo_variance_recent', 'market_odds_spread']
        hybrid_features = base_features + selected_specialized
        
        logger.info(f"🎯 Features: {len(hybrid_features)} (10 base + 2 spécialisées)")
        
        # Données train
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        X_train = df_train_clean[hybrid_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        # Données test (sans restriction vrais résultats)
        X_test = df_test_40[hybrid_features].fillna(0.5)
        y_test_target = df_test_40['target']  # Target du dataset (peut être prédictif)
        
        logger.info(f"📊 Features test shape: {X_test.shape}")
        
        # Test 3 approches sur 40 matchs
        results = {}
        
        # 1. Baseline Simple
        logger.info("\n⚙️  Test Baseline Simple (10 features)...")
        baseline = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        baseline.fit(X_train[base_features], y_train)
        pred_baseline = baseline.predict(X_test[base_features])
        pred_baseline_str = pd.Series(pred_baseline).map({0: 'H', 1: 'D', 2: 'A'}).values
        
        # 2. Cascade Hybride
        logger.info("\n⚙️  Test Cascade Hybride (12 features)...")
        cascade = HybridCascade40(draw_weight=2.5, draw_threshold=0.40, max_draw_ratio=0.20)
        cascade.fit(X_train, y_train)
        pred_cascade = cascade.predict(X_test)
        
        # Résultats
        logger.info(f"\n🏆 RÉSULTATS SUR 40 MATCHS COMPLETS")
        logger.info(f"=" * 45)
        
        # Distributions prédites
        dist_baseline = pd.Series(pred_baseline_str).value_counts(normalize=True)
        dist_cascade = pd.Series(pred_cascade).value_counts(normalize=True)
        
        logger.info(f"\n📊 DISTRIBUTIONS PRÉDITES:")
        logger.info(f"   Baseline:  H={dist_baseline.get('H', 0):.1%}, D={dist_baseline.get('D', 0):.1%}, A={dist_baseline.get('A', 0):.1%}")
        logger.info(f"   Cascade:   H={dist_cascade.get('H', 0):.1%}, D={dist_cascade.get('D', 0):.1%}, A={dist_cascade.get('A', 0):.1%}")
        
        # Draws prédits
        draws_baseline = (pred_baseline_str == 'D').sum()
        draws_cascade = (pred_cascade == 'D').sum()
        
        logger.info(f"\n🎯 DÉTECTION DRAWS:")
        logger.info(f"   Baseline:  {draws_baseline}/40 draws prédits ({draws_baseline/40*100:.1f}%)")
        logger.info(f"   Cascade:   {draws_cascade}/40 draws prédits ({draws_cascade/40*100:.1f}%)")
        
        # Features spécialisées stats
        logger.info(f"\n📊 FEATURES SPÉCIALISÉES (sur 40 matchs):")
        for feat in selected_specialized:
            mean_val = X_test[feat].mean()
            std_val = X_test[feat].std()
            logger.info(f"   {feat}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Quelques exemples prédictions
        logger.info(f"\n🔍 EXEMPLES PRÉDICTIONS (5 premiers matchs):")
        for i in range(min(5, len(df_test_40))):
            home = df_test_40.iloc[i]['HomeTeam']
            away = df_test_40.iloc[i]['AwayTeam']
            date = df_test_40.iloc[i]['Date'].strftime('%Y-%m-%d')
            pred_b = pred_baseline_str[i]
            pred_c = pred_cascade[i]
            
            logger.info(f"   {date}: {home} vs {away} | Baseline: {pred_b}, Cascade: {pred_c}")
        
        results = {
            'n_test': 40,
            'baseline_draws': draws_baseline,
            'cascade_draws': draws_cascade,
            'baseline_dist': dist_baseline.to_dict(),
            'cascade_dist': dist_cascade.to_dict()
        }
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_40_matchs_complets()
    
    if result:
        print(f"\n🎯 TEST 40 MATCHS COMPLETS TERMINÉ")
        print(f"   Draws Baseline: {result['baseline_draws']}/40 ({result['baseline_draws']/40*100:.1f}%)")
        print(f"   Draws Cascade: {result['cascade_draws']}/40 ({result['cascade_draws']/40*100:.1f}%)")
    else:
        print("❌ Échec test 40 matchs")