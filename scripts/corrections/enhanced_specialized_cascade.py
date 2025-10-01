#!/usr/bin/env python3
"""
🎯 CASCADE SPÉCIALISÉ AVEC FEATURES RÉELLES CRAFTÉES
=================================================

Test du modèle cascade optimisé avec les 15 features spécialisées 
du dataset v16 sur les 40 matchs J1-J4 EPL 2025-26.

Features utilisées:
- 10 features originales du modèle v2.3
- 5 nouvelles features craftées: elo_variance_recent, team_parity_score, 
  market_odds_spread, low_scoring_potential, is_promoted
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_cascade")

class EnhancedSpecializedCascade:
    """Cascade avec features spécialisées réelles"""
    
    def __init__(self, draw_weight=4, draw_threshold=0.30, parity_boost=1.5):
        # Draw Specialist Forest
        self.clf_draw = RandomForestClassifier(
            n_estimators=300,
            max_depth=15, 
            min_samples_leaf=3,
            class_weight={0: 1, 1: draw_weight},
            random_state=42
        )
        
        # Home/Away Forest
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=200,
            max_depth=12, 
            random_state=42,
            class_weight="balanced"
        )
        
        self.draw_threshold = draw_threshold
        self.parity_boost = parity_boost
        self.is_fitted = False
    
    def _get_enhanced_features(self, X):
        """Calculer features combinées pour draws"""
        X_enhanced = X.copy()
        
        # Boost des features spécialisées pour draws
        if 'team_parity_score' in X.columns:
            X_enhanced['parity_boosted'] = X['team_parity_score'] * self.parity_boost
            
        if 'market_odds_spread' in X.columns and 'elo_variance_recent' in X.columns:
            # Feature combinée: uncertainty signal
            X_enhanced['uncertainty_signal'] = (X['market_odds_spread'] + X['elo_variance_recent']) / 2
            
        return X_enhanced
    
    def fit(self, X, y):
        """Entraîner les deux forêts spécialisées"""
        
        # Conversion target
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'}) 
        else:
            y_str = y.copy()
        
        # Features améliorées
        X_enhanced = self._get_enhanced_features(X)
        
        # 1. Entraîner Draw Forest
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X_enhanced, y_draw)
        
        # 2. Entraîner Home/Away Forest (uniquement sur non-draws)
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X_enhanced[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            valid_homeaway = y_homeaway.notna()
            
            if valid_homeaway.sum() > 5:
                X_notdraw_clean = X_notdraw[valid_homeaway] 
                y_homeaway_clean = y_homeaway[valid_homeaway]
                self.clf_homeaway.fit(X_notdraw_clean, y_homeaway_clean)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Prédiction cascade avec features spécialisées"""
        
        X_enhanced = self._get_enhanced_features(X)
        
        # 1. Probabilité Draw
        proba_draw = self.clf_draw.predict_proba(X_enhanced)[:, 1]
        
        # 2. Seuil adaptatif basé sur features spécialisées
        adaptive_threshold = np.full(len(X), self.draw_threshold)
        
        if 'team_parity_score' in X.columns:
            # Réduire seuil pour matchs très équilibrés
            high_parity_mask = X['team_parity_score'] > 0.9
            adaptive_threshold[high_parity_mask] *= 0.8
            
        if 'is_promoted' in X.columns:
            # Augmenter seuil pour matchs avec équipe promue (plus imprévisible)
            promoted_mask = X['is_promoted'] == 1
            adaptive_threshold[promoted_mask] *= 1.2
        
        # 3. Prédiction Draw
        pred_draw = (proba_draw > adaptive_threshold).astype(int)
        
        # 4. Limitation draws (max 30% du dataset)
        target_draw_ratio = 0.30
        n_draws_target = int(len(X) * target_draw_ratio)
        
        if pred_draw.sum() > n_draws_target:
            top_draw_indices = np.argsort(proba_draw)[-n_draws_target:]
            pred_draw_filtered = np.zeros_like(pred_draw)
            pred_draw_filtered[top_draw_indices] = 1
            pred_draw = pred_draw_filtered
        
        # 5. Home/Away pour non-draws
        pred_homeaway = self.clf_homeaway.predict(X_enhanced)
        
        # 6. Assemblage final
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_away == 1 else 'A')
        
        return np.array(y_pred)

def test_enhanced_cascade():
    """Test cascade avec features spécialisées réelles"""
    logger.info("🎯 TEST CASCADE AVEC FEATURES SPÉCIALISÉES RÉELLES")
    logger.info("=" * 60)
    
    try:
        # Charger dataset amélioré v16
        df = pd.read_csv('data/processed/v16_specialized_features_enhanced.csv', parse_dates=['Date'])
        logger.info(f"📊 Dataset v16 chargé: {len(df)} matchs, {len(df.columns)} features")
        
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
        
        # Split temporel
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        
        # Test sur 40 matchs J1-J4 depuis dataset v16 (avec features spécialisées)
        df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
        df_test_candidates = df_season_2025.head(40).copy()
        df_test = pd.merge(df_test_candidates, real_matches, on=['HomeTeam', 'AwayTeam'], how='inner')
        
        logger.info(f"📊 Données: train={len(df_train)}, test={len(df_test)} matchs")
        
        # Features modèle (15 features spécialisées)
        base_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        specialized_features = [
            'elo_variance_recent', 'team_parity_score', 'market_odds_spread',
            'low_scoring_potential', 'is_promoted'
        ]
        
        all_features = base_features + specialized_features
        logger.info(f"🎯 Features: {len(all_features)} (10 base + 5 spécialisées)")
        
        # Préparer données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[all_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[all_features].fillna(0.5)
        y_real = df_test['FTR']
        
        # Test modèle cascade amélioré
        logger.info("⚙️  Entraînement cascade avec features spécialisées...")
        
        model = EnhancedSpecializedCascade(draw_weight=4, draw_threshold=0.30, parity_boost=1.5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        # Résultats détaillés
        logger.info(f"\n🏆 RÉSULTATS CASCADE SPÉCIALISÉ")
        logger.info(f"   Accuracy: {accuracy:.1%} ({int(accuracy * len(y_real))}/{len(y_real)})") 
        
        # Distribution
        real_dist = y_real.value_counts(normalize=True)
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        
        logger.info(f"   Réel: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
        logger.info(f"   Prédit: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
        
        # Analyse des draws
        draws_predicted = (y_pred == 'D').sum()
        draws_correct = ((y_pred == 'D') & (y_real == 'D')).sum()
        draws_real = (y_real == 'D').sum()
        
        logger.info(f"   Draws: {draws_correct}/{draws_real} réels capturés ({draws_correct/draws_real*100 if draws_real > 0 else 0:.1f}%)")
        logger.info(f"   Draws prédits: {draws_predicted} (precision: {draws_correct/draws_predicted*100 if draws_predicted > 0 else 0:.1f}%)")
        
        # Importance des features spécialisées
        logger.info(f"\n🔍 IMPORTANCE FEATURES DRAW FOREST:")
        feature_names = all_features + ['parity_boosted', 'uncertainty_signal']  # Features combinées
        if hasattr(model.clf_draw, 'feature_importances_'):
            importances = model.clf_draw.feature_importances_
            for i, importance in enumerate(importances[:len(all_features)]):
                if importance > 0.05:  # Seulement features importantes
                    logger.info(f"   {all_features[i]}: {importance:.3f}")
        
        # Comparaison avec baselines
        logger.info(f"\n📈 COMPARAISONS:")
        logger.info(f"   Cascade Spécialisé v16: {accuracy:.1%}")
        logger.info(f"   Baseline v2.3 (10 features): 52.5%")
        logger.info(f"   Amélioration: {accuracy - 0.525:+.1%}")
        
        # Features spécialisées stats
        if len(specialized_features) > 0:
            logger.info(f"\n📊 FEATURES SPÉCIALISÉES UTILISÉES:")
            for feat in specialized_features:
                if feat in X_test.columns:
                    mean_val = X_test[feat].mean()
                    std_val = X_test[feat].std()
                    logger.info(f"   {feat}: {mean_val:.3f} ± {std_val:.3f}")
        
        return {
            'accuracy_enhanced': accuracy,
            'improvement_vs_baseline': accuracy - 0.525,
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
    result = test_enhanced_cascade()
    
    if result:
        print(f"\n🎯 CASCADE SPÉCIALISÉ v16: {result['accuracy_enhanced']:.1%}")
        print(f"Amélioration vs baseline: {result['improvement_vs_baseline']:+.1%}")
        print(f"Draws capturés: {result['draws_captured']}/{result['draws_total']}")
        print(f"Précision draws: {result['draws_precision']:.1%}")
    else:
        print("❌ Échec test cascade spécialisé")