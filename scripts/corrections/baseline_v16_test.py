#!/usr/bin/env python3
"""
🎯 BASELINE TEST v16 - CONFIRMER PERFORMANCE 10 FEATURES
=====================================================

Test rapide pour confirmer la performance baseline avec les 10 features
originales du modèle v2.3 sur le dataset v16.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline_v16")

class BaselineModel:
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    
    def fit(self, X, y):
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_encoded = y
        elif y.dtype == 'int64':
            y_encoded = y 
        else:
            y_encoded = y.map({'H': 0, 'D': 1, 'A': 2})
        
        self.clf.fit(X, y_encoded)
        return self
    
    def predict(self, X):
        y_pred_encoded = self.clf.predict(X)
        return pd.Series(y_pred_encoded).map({0: 'H', 1: 'D', 2: 'A'}).values

def test_baseline_v16():
    logger.info("🎯 TEST BASELINE 10 FEATURES sur v16")
    logger.info("=" * 45)
    
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
        
        # SEULEMENT 10 features de base
        base_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[base_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[base_features].fillna(0.5)
        y_real = df_test['FTR']
        
        logger.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"🎯 Features: {len(base_features)} (base seulement)")
        
        # Modèle simple
        model = BaselineModel()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        # Résultats
        logger.info(f"\n🏆 BASELINE v16 (10 features)")
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
        
        return {
            'accuracy': accuracy,
            'n_test': len(y_real),
            'draws_captured': draws_correct,
            'draws_total': draws_real
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_baseline_v16()
    
    if result:
        print(f"\n🎯 BASELINE v16: {result['accuracy']:.1%}")
        print(f"Test matchs: {result['n_test']}")
        print(f"Draws: {result['draws_captured']}/{result['draws_total']}")
    else:
        print("❌ Échec baseline test")