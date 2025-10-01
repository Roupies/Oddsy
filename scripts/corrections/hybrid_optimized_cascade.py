#!/usr/bin/env python3
"""
🎯 CASCADE HYBRIDE OPTIMISÉ - ÉQUILIBRE ACCURACY/DRAWS
=====================================================

Cascade optimisé qui vise l'équilibre entre accuracy globale et détection draws.
- Features sélectionnées intelligemment
- Seuils adaptatifs conservateurs  
- Limitation stricte des draws prédits
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_cascade")

class HybridOptimizedCascade:
    """Cascade optimisé pour équilibre accuracy/draws"""
    
    def __init__(self, draw_weight=2.5, draw_threshold=0.40, max_draw_ratio=0.20):
        # Draw Forest (plus conservateur)
        self.clf_draw = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: draw_weight},
            random_state=42
        )
        
        # Home/Away Forest
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        )
        
        self.draw_threshold = draw_threshold
        self.max_draw_ratio = max_draw_ratio
        self.is_fitted = False
    
    def fit(self, X, y):
        """Entraîner cascade avec features optimisées"""
        
        # Conversion target
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
                X_notdraw_clean = X_notdraw[valid_homeaway]
                y_homeaway_clean = y_homeaway[valid_homeaway]
                self.clf_homeaway.fit(X_notdraw_clean, y_homeaway_clean)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Prédiction équilibrée"""
        
        # 1. Probabilité Draw
        proba_draw = self.clf_draw.predict_proba(X)[:, 1]
        
        # 2. Seuil adaptatif très conservateur
        adaptive_threshold = self.draw_threshold
        
        # 3. Prédiction Draw initiale
        pred_draw = (proba_draw > adaptive_threshold).astype(int)
        
        # 4. LIMITATION STRICTE des draws (max 20% du dataset)
        n_draws_target = int(len(X) * self.max_draw_ratio)
        
        if pred_draw.sum() > n_draws_target:
            # Garder seulement les N plus probables
            top_draw_indices = np.argsort(proba_draw)[-n_draws_target:]
            pred_draw_filtered = np.zeros_like(pred_draw)
            pred_draw_filtered[top_draw_indices] = 1
            pred_draw = pred_draw_filtered
        
        # 5. Home/Away pour non-draws
        pred_homeaway = self.clf_homeaway.predict(X)
        
        # 6. Assemblage final
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_away == 1 else 'A')
        
        return np.array(y_pred)

def test_hybrid_cascade():
    """Test cascade hybride optimisé"""
    logger.info("🎯 TEST CASCADE HYBRIDE OPTIMISÉ")
    logger.info("=" * 40)
    
    try:
        # Charger dataset v16
        df = pd.read_csv('data/processed/v16_specialized_features_enhanced.csv', parse_dates=['Date'])
        
        # Charger vrais résultats
        df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
        team_mapping = {
            'Spurs': 'Tottenham',
            "Nott'm Forest": "Nott'm Forest"
        }
        df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
        df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
        real_matches = df_real[['HomeTeam', 'AwayTeam', 'FTR']]
        
        # Target encoding
        if 'target' not in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Split
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
        df_test_candidates = df_season_2025.head(40).copy()
        df_test = pd.merge(df_test_candidates, real_matches, on=['HomeTeam', 'AwayTeam'], how='inner')
        
        # Features HYBRIDES : 10 base + 2 sélectionnées
        base_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Sélection: 2 features spécialisées les plus prometteuses
        selected_specialized = [
            'elo_variance_recent',  # Variance ELO utile
            'market_odds_spread'    # Spread bookmakers
        ]
        
        hybrid_features = base_features + selected_specialized
        logger.info(f"🎯 Features hybrides: {len(hybrid_features)} (10 + 2 sélectionnées)")
        
        # Données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[hybrid_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[hybrid_features].fillna(0.5)
        y_real = df_test['FTR']
        
        logger.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Test cascade hybride
        logger.info("⚙️  Test cascade hybride...")
        
        model = HybridOptimizedCascade(draw_weight=2.5, draw_threshold=0.40, max_draw_ratio=0.20)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        # Résultats
        logger.info(f"\n🏆 RÉSULTATS CASCADE HYBRIDE")
        logger.info(f"   Accuracy: {accuracy:.1%} ({int(accuracy * len(y_real))}/{len(y_real)})")
        
        # Distribution
        real_dist = y_real.value_counts(normalize=True)
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        
        logger.info(f"   Réel: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
        logger.info(f"   Prédit: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
        
        # Draws
        draws_predicted = (y_pred == 'D').sum()
        draws_correct = ((y_pred == 'D') & (y_real == 'D')).sum()
        draws_real = (y_real == 'D').sum()
        
        logger.info(f"   Draws: {draws_correct}/{draws_real} capturés ({draws_correct/draws_real*100 if draws_real > 0 else 0:.1f}%)")
        logger.info(f"   Draws prédits: {draws_predicted} (precision: {draws_correct/draws_predicted*100 if draws_predicted > 0 else 0:.1f}%)")
        
        # Comparaisons
        logger.info(f"\n📈 COMPARAISONS:")
        logger.info(f"   Cascade Hybride: {accuracy:.1%}")
        logger.info(f"   Baseline simple: 46.7%")
        logger.info(f"   Cascade spécialisé: 36.7%")
        logger.info(f"   Amélioration vs baseline: {accuracy - 0.467:+.1%}")
        
        return {
            'accuracy_hybrid': accuracy,
            'improvement_vs_baseline': accuracy - 0.467,
            'draws_captured': draws_correct,
            'draws_total': draws_real,
            'draws_precision': draws_correct/draws_predicted if draws_predicted > 0 else 0,
            'n_test': len(y_real)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_hybrid_cascade()
    
    if result:
        print(f"\n🎯 CASCADE HYBRIDE: {result['accuracy_hybrid']:.1%}")
        print(f"Amélioration: {result['improvement_vs_baseline']:+.1%}")
        print(f"Draws: {result['draws_captured']}/{result['draws_total']} ({result['draws_precision']:.1%} précision)")
    else:
        print("❌ Échec test hybride")