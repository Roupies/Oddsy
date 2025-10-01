#!/usr/bin/env python3
"""
🎯 TEST SEUIL 0.27 SIMPLE
========================
Test direct avec seuil 0.27 pour vérifier la détection de draws.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seuil027")

def test_seuil_027():
    try:
        logger.info("🎯 TEST SEUIL 0.27 DIRECT")
        logger.info("=" * 30)
        
        # Chargement données
        auto_data = pd.read_csv("data/processed/v_auto_update_20250916_110247.csv")
        auto_data['Date'] = pd.to_datetime(auto_data['Date'])
        
        # Splits
        train_data = auto_data[auto_data['Date'] < '2025-08-01'].copy()
        test_data = auto_data[auto_data['Date'] >= '2025-08-01'].head(40).copy()
        
        # Création target
        def create_target_from_result(df):
            df_copy = df.copy()
            mask_no_target = df_copy['target'].isna()
            if mask_no_target.any():
                df_copy.loc[mask_no_target, 'target'] = df_copy.loc[mask_no_target, 'FullTimeResult'].map({
                    'H': 0, 'D': 1, 'A': 2
                })
            return df_copy
        
        train_data = create_target_from_result(train_data)
        test_data = create_target_from_result(test_data)
        
        # Features
        feature_cols = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        X_train = train_data[feature_cols].fillna(0)
        X_test = test_data[feature_cols].fillna(0)
        y_train = train_data['target']
        y_test = test_data['target']
        
        # Filtrage NaN
        train_valid = y_train.notna()
        test_valid = y_test.notna()
        
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        test_data = test_data[test_valid]
        
        y_train_str = y_train.map({0: 'H', 1: 'D', 2: 'A'})
        y_test_str = y_test.map({0: 'H', 1: 'D', 2: 'A'})
        
        logger.info(f"📊 Données: Train {len(X_train)}, Test {len(X_test)}")
        
        # 1. MODÈLE BASELINE
        baseline_model = RandomForestClassifier(n_estimators=150, random_state=42)
        baseline_model.fit(X_train, y_train_str)
        baseline_preds = baseline_model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test_str, baseline_preds) * 100
        
        # 2. CASCADE MANUEL ÉTAPE PAR ÉTAPE
        logger.info(f"\n🔧 CASCADE MANUEL SEUIL 0.27")
        
        # 2a. Entraînement Draw model
        y_draw_train = (y_train_str == 'D').astype(int)
        draw_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight={0: 1, 1: 3.0}, random_state=42
        )
        draw_model.fit(X_train, y_draw_train)
        
        # 2b. Prédiction draws
        draw_proba = draw_model.predict_proba(X_test)[:, 1]
        threshold = 0.27
        is_draw = draw_proba >= threshold
        
        logger.info(f"   Proba Draw min/max: {draw_proba.min():.3f}/{draw_proba.max():.3f}")
        logger.info(f"   Seuil utilisé: {threshold}")
        logger.info(f"   Draws détectés: {is_draw.sum()}/40")
        
        # 2c. Entraînement Home/Away model
        mask_notdraw_train = y_train_str != 'D'
        X_homeaway = X_train[mask_notdraw_train]
        y_homeaway = y_train_str[mask_notdraw_train].map({'H': 1, 'A': 0})
        
        homeaway_model = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )
        homeaway_model.fit(X_homeaway, y_homeaway)
        
        # 2d. Prédictions cascade
        cascade_preds = np.full(len(X_test), 'D', dtype=object)
        
        # Pour les non-draws
        mask_notdraw = ~is_draw
        if mask_notdraw.sum() > 0:
            homeaway_pred = homeaway_model.predict(X_test[mask_notdraw])
            cascade_preds[mask_notdraw] = np.where(homeaway_pred == 1, 'H', 'A')
        
        # 3. RÉSULTATS
        cascade_accuracy = accuracy_score(y_test_str, cascade_preds) * 100
        
        logger.info(f"\n🏆 RÉSULTATS AVEC SEUIL 0.27:")
        logger.info(f"   Baseline: {baseline_accuracy:.1f}% accuracy")
        logger.info(f"   Cascade:  {cascade_accuracy:.1f}% accuracy")
        logger.info(f"   Différence: {cascade_accuracy - baseline_accuracy:+.1f}pp")
        
        # Distribution
        baseline_dist = pd.Series(baseline_preds).value_counts(normalize=True).sort_index() * 100
        cascade_dist = pd.Series(cascade_preds).value_counts(normalize=True).sort_index() * 100
        
        logger.info(f"\n📊 DISTRIBUTIONS:")
        logger.info(f"   Baseline: H={baseline_dist.get('H', 0):.1f}%, D={baseline_dist.get('D', 0):.1f}%, A={baseline_dist.get('A', 0):.1f}%")
        logger.info(f"   Cascade:  H={cascade_dist.get('H', 0):.1f}%, D={cascade_dist.get('D', 0):.1f}%, A={cascade_dist.get('A', 0):.1f}%")
        
        # Détection draws
        baseline_draws = (baseline_preds == 'D').sum()
        cascade_draws = (cascade_preds == 'D').sum()
        
        logger.info(f"\n🎯 DÉTECTION DRAWS:")
        logger.info(f"   Baseline: {baseline_draws}/40")
        logger.info(f"   Cascade:  {cascade_draws}/40")
        logger.info(f"   Vrais draws: {(y_test_str == 'D').sum()}/40")
        
        # Performance par classe
        logger.info(f"\n📈 PERFORMANCE PAR CLASSE:")
        for outcome in ['H', 'D', 'A']:
            mask = y_test_str == outcome
            if mask.sum() > 0:
                actual_count = mask.sum()
                baseline_correct = (baseline_preds[mask] == outcome).sum()
                cascade_correct = (cascade_preds[mask] == outcome).sum()
                
                logger.info(f"   {outcome} ({actual_count} vrais): Baseline {baseline_correct}/{actual_count}, Cascade {cascade_correct}/{actual_count}")
        
        # Échantillon draws détectés
        if cascade_draws > 0:
            logger.info(f"\n🔍 DRAWS DÉTECTÉS PAR CASCADE:")
            draw_indices = np.where(cascade_preds == 'D')[0]
            for i, idx in enumerate(draw_indices[:5]):  # Max 5
                row = test_data.iloc[idx]
                actual = y_test_str.iloc[idx]
                proba = draw_proba[idx]
                logger.info(f"   {i+1}. {row['Date'].strftime('%Y-%m-%d')}: {row['HomeTeam']} vs {row['AwayTeam']}")
                logger.info(f"      Réel: {actual} | Proba: {proba:.3f}")
        
        print(f"\n🎯 RÉSUMÉ SEUIL 0.27:")
        print(f"   Baseline: {baseline_accuracy:.1f}% accuracy, {baseline_draws} draws")
        print(f"   Cascade:  {cascade_accuracy:.1f}% accuracy, {cascade_draws} draws")
        print(f"   Amélioration: {cascade_accuracy - baseline_accuracy:+.1f}pp accuracy, {cascade_draws - baseline_draws:+d} draws")
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'cascade_accuracy': cascade_accuracy,
            'cascade_draws': cascade_draws,
            'accuracy_improvement': cascade_accuracy - baseline_accuracy
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_seuil_027()