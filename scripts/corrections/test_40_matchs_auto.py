#!/usr/bin/env python3
"""
🎯 TEST 40 MATCHS J1-J4 AVEC AUTO DATASET
======================================

Test sur 40 matchs EPL 2025-26 avec dataset auto-update.
Comparaison baseline vs cascade sur échantillon complet.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_40_auto")

class CascadeModel40:
    """Cascade optimisé pour 40 matchs"""
    
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
        # Conversion vers classes string
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'}) 
        else:
            y_str = y.copy()
        
        # 1. Draw Forest
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X, y_draw)
        
        # 2. Home/Away Forest
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

def test_40_matchs_auto():
    """Test sur 40 matchs avec dataset auto"""
    logger.info("🎯 TEST 40 MATCHS J1-J4 AVEC AUTO DATASET")
    logger.info("=" * 48)
    
    try:
        # Charger dataset v15 pour train
        df_train_full = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df_train_full[df_train_full['Date'] < train_cutoff].copy()
        
        # Charger dataset auto pour test 40 matchs
        df_auto = pd.read_csv('data/processed/v_auto_update_20250916_105039.csv', parse_dates=['Date'])
        df_season_2025_auto = df_auto[df_auto['Date'] >= '2025-08-01'].copy().sort_values('Date')
        df_test_40 = df_season_2025_auto.head(40).copy()
        
        logger.info(f"📊 Train: {len(df_train)}, Test: {len(df_test_40)} matchs")
        logger.info(f"📅 Test période: {df_test_40['Date'].min()} à {df_test_40['Date'].max()}")
        
        # Target encoding
        if 'target' not in df_train.columns:
            df_train['target'] = df_train['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Features (10 de base - compatibles entre datasets)
        base_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        logger.info(f"🎯 Features: {len(base_features)} (base compatibles)")
        
        # Préparation données train
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        X_train = df_train_clean[base_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        # Données test
        X_test = df_test_40[base_features].fillna(0.5)
        
        # Test 2 approches
        logger.info(f"\n⚙️  Tests sur {len(df_test_40)} matchs...")
        
        # 1. Baseline Simple
        baseline = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        baseline.fit(X_train, y_train)
        pred_baseline_encoded = baseline.predict(X_test)
        pred_baseline = pd.Series(pred_baseline_encoded).map({0: 'H', 1: 'D', 2: 'A'}).values
        
        # 2. Cascade
        cascade = CascadeModel40(draw_weight=2.5, draw_threshold=0.40, max_draw_ratio=0.20)
        cascade.fit(X_train, y_train)
        pred_cascade = cascade.predict(X_test)
        
        # Résultats
        logger.info(f"\n🏆 RÉSULTATS SUR 40 MATCHS COMPLETS J1-J4")
        logger.info(f"=" * 50)
        
        # Distributions
        dist_baseline = pd.Series(pred_baseline).value_counts(normalize=True)
        dist_cascade = pd.Series(pred_cascade).value_counts(normalize=True)
        
        logger.info(f"\n📊 DISTRIBUTIONS PRÉDITES:")
        logger.info(f"   Baseline:  H={dist_baseline.get('H', 0):.1%}, D={dist_baseline.get('D', 0):.1%}, A={dist_baseline.get('A', 0):.1%}")
        logger.info(f"   Cascade:   H={dist_cascade.get('H', 0):.1%}, D={dist_cascade.get('D', 0):.1%}, A={dist_cascade.get('A', 0):.1%}")
        
        # Détection draws
        draws_baseline = (pred_baseline == 'D').sum()
        draws_cascade = (pred_cascade == 'D').sum()
        
        logger.info(f"\n🎯 DÉTECTION DRAWS SUR 40 MATCHS:")
        logger.info(f"   Baseline:  {draws_baseline}/40 draws prédits ({draws_baseline/40*100:.1f}%)")
        logger.info(f"   Cascade:   {draws_cascade}/40 draws prédits ({draws_cascade/40*100:.1f}%)")
        
        # Distribution réelle théorique EPL
        logger.info(f"\n📈 RÉFÉRENCE EPL HISTORIQUE:")
        logger.info(f"   Théorique: H=43.6%, D=23.0%, A=33.4%")
        
        # Échantillon de prédictions
        logger.info(f"\n🔍 ÉCHANTILLON PRÉDICTIONS (8 premiers matchs):")
        for i in range(min(8, len(df_test_40))):
            home = df_test_40.iloc[i]['HomeTeam']
            away = df_test_40.iloc[i]['AwayTeam']
            date = df_test_40.iloc[i]['Date'].strftime('%Y-%m-%d')
            pred_b = pred_baseline[i]
            pred_c = pred_cascade[i]
            
            # Ajouter différence si différente
            diff_marker = " ✓" if pred_b == pred_c else f" ≠ ({pred_b}→{pred_c})"
            logger.info(f"   {date}: {home} vs {away} | Cascade: {pred_c}{diff_marker}")
        
        # Matchs où cascade prédit draw mais pas baseline
        cascade_draws = df_test_40[pred_cascade == 'D']
        if len(cascade_draws) > 0:
            logger.info(f"\n🎯 MATCHS PRÉDITS COMME DRAWS PAR CASCADE:")
            for i, (_, match) in enumerate(cascade_draws.iterrows()):
                home = match['HomeTeam']
                away = match['AwayTeam'] 
                date = match['Date'].strftime('%Y-%m-%d')
                logger.info(f"   {i+1}. {date}: {home} vs {away}")
        
        results = {
            'n_test': 40,
            'baseline_draws': draws_baseline,
            'cascade_draws': draws_cascade,
            'baseline_dist': dist_baseline.to_dict(),
            'cascade_dist': dist_cascade.to_dict(),
            'draw_improvement': draws_cascade - draws_baseline
        }
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_40_matchs_auto()
    
    if result:
        print(f"\n🎯 RÉSUMÉ 40 MATCHS J1-J4:")
        print(f"   Baseline draws: {result['baseline_draws']}/40 ({result['baseline_draws']/40*100:.1f}%)")
        print(f"   Cascade draws: {result['cascade_draws']}/40 ({result['cascade_draws']/40*100:.1f}%)")
        print(f"   Amélioration: +{result['draw_improvement']} draws détectés")
    else:
        print("❌ Échec test 40 matchs auto")