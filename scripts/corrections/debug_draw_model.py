#!/usr/bin/env python3
"""
🔍 DEBUG MODÈLE DRAW
==================
Analyse en profondeur du modèle Draw pour comprendre pourquoi il ne détecte aucun nul.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_draw")

def debug_draw_model():
    try:
        logger.info("🔍 DEBUG MODÈLE DRAW")
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
        
        logger.info(f"📊 Données: Train {len(X_train)}, Test {len(X_test)}")
        
        # 1. ANALYSE DISTRIBUTION CLASSES DANS TRAIN
        y_train_str = y_train.map({0: 'H', 1: 'D', 2: 'A'})
        train_dist = y_train_str.value_counts()
        
        logger.info(f"\n📊 DISTRIBUTION TRAIN:")
        for classe, count in train_dist.items():
            pct = count / len(y_train_str) * 100
            logger.info(f"   {classe}: {count} matchs ({pct:.1f}%)")
        
        # 2. ANALYSE DISTRIBUTION TEST
        y_test_str = y_test.map({0: 'H', 1: 'D', 2: 'A'})
        test_dist = y_test_str.value_counts()
        
        logger.info(f"\n📊 DISTRIBUTION TEST (40 matchs):")
        for classe, count in test_dist.items():
            pct = count / len(y_test_str) * 100
            logger.info(f"   {classe}: {count} matchs ({pct:.1f}%)")
        
        # 3. ENTRAÎNEMENT MODÈLE DRAW SIMPLE
        logger.info(f"\n🤖 ENTRAÎNEMENT MODÈLE DRAW")
        
        y_draw_train = (y_train_str == 'D').astype(int)
        y_draw_test = (y_test_str == 'D').astype(int)
        
        logger.info(f"   Draw train: {y_draw_train.sum()}/{len(y_draw_train)} ({y_draw_train.mean()*100:.1f}%)")
        logger.info(f"   Draw test: {y_draw_test.sum()}/{len(y_draw_test)} ({y_draw_test.mean()*100:.1f}%)")
        
        # Modèle simple sans class_weight d'abord
        draw_model_simple = RandomForestClassifier(n_estimators=100, random_state=42)
        draw_model_simple.fit(X_train, y_draw_train)
        
        # Prédictions
        draw_proba_simple = draw_model_simple.predict_proba(X_test)
        draw_pred_simple = draw_model_simple.predict(X_test)
        
        logger.info(f"\n📈 MODÈLE DRAW SIMPLE (sans class_weight):")
        logger.info(f"   Classes détectées: {draw_model_simple.classes_}")
        logger.info(f"   Shape proba: {draw_proba_simple.shape}")
        
        if draw_proba_simple.shape[1] > 1:
            draw_proba_1 = draw_proba_simple[:, 1]
            logger.info(f"   Proba Draw min: {draw_proba_1.min():.3f}")
            logger.info(f"   Proba Draw max: {draw_proba_1.max():.3f}")
            logger.info(f"   Proba Draw mean: {draw_proba_1.mean():.3f}")
        else:
            logger.info(f"   UNE SEULE CLASSE DÉTECTÉE: {draw_model_simple.classes_[0]}")
            
        # Performance
        from sklearn.metrics import accuracy_score, classification_report
        accuracy_simple = accuracy_score(y_draw_test, draw_pred_simple)
        logger.info(f"   Accuracy Draw: {accuracy_simple:.3f}")
        
        # 4. MODÈLE AVEC CLASS_WEIGHT
        logger.info(f"\n📈 MODÈLE DRAW AVEC CLASS_WEIGHT:")
        
        draw_model_weighted = RandomForestClassifier(
            n_estimators=100, 
            class_weight={0: 1, 1: 3.0}, 
            random_state=42
        )
        draw_model_weighted.fit(X_train, y_draw_train)
        
        draw_proba_weighted = draw_model_weighted.predict_proba(X_test)
        draw_pred_weighted = draw_model_weighted.predict(X_test)
        
        logger.info(f"   Classes détectées: {draw_model_weighted.classes_}")
        logger.info(f"   Shape proba: {draw_proba_weighted.shape}")
        
        if draw_proba_weighted.shape[1] > 1:
            draw_proba_1_weighted = draw_proba_weighted[:, 1]
            logger.info(f"   Proba Draw min: {draw_proba_1_weighted.min():.3f}")
            logger.info(f"   Proba Draw max: {draw_proba_1_weighted.max():.3f}")
            logger.info(f"   Proba Draw mean: {draw_proba_1_weighted.mean():.3f}")
            
            # Test différents seuils
            logger.info(f"\n🎯 DÉTECTION DRAWS AVEC DIFFÉRENTS SEUILS:")
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                draws_detected = (draw_proba_1_weighted >= threshold).sum()
                logger.info(f"   Seuil {threshold}: {draws_detected}/40 draws détectés")
        else:
            logger.info(f"   UNE SEULE CLASSE DÉTECTÉE: {draw_model_weighted.classes_[0]}")
        
        accuracy_weighted = accuracy_score(y_draw_test, draw_pred_weighted)
        logger.info(f"   Accuracy Draw weighted: {accuracy_weighted:.3f}")
        
        # 5. FEATURES IMPORTANTES
        logger.info(f"\n🔍 FEATURES IMPORTANTES POUR DRAWS:")
        feature_importance = draw_model_weighted.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.3f}")
        
        # 6. ANALYSE ÉCHANTILLON PRÉDICTIONS
        logger.info(f"\n🔍 ÉCHANTILLON PRÉDICTIONS DRAW (5 premiers):")
        for i in range(min(5, len(test_data))):
            row = test_data.iloc[i]
            actual = y_test_str.iloc[i]
            if draw_proba_weighted.shape[1] > 1:
                proba_draw = draw_proba_weighted[i, 1]
                logger.info(f"   {row['Date'].strftime('%Y-%m-%d')}: {row['HomeTeam']} vs {row['AwayTeam']}")
                logger.info(f"      Réel: {actual} | Proba Draw: {proba_draw:.3f}")
        
        return {
            'train_draws': y_draw_train.sum(),
            'test_draws': y_draw_test.sum(),
            'model_classes': draw_model_weighted.classes_.tolist(),
            'max_proba': draw_proba_weighted.max() if draw_proba_weighted.shape[1] > 1 else None
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_draw_model()