#!/usr/bin/env python3
"""
🧪 TEST FEATURES V16 ANTI-LEAKAGE
=================================
Test des features contextuelles v16 pour améliorer début saison EPL 2025-26.
Split temporel strict pour éviter data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v16_test")

def test_v16_features_antileakage():
    """Test features v16 avec protection anti-leakage."""
    logger.info("🧪 TEST FEATURES V16 ANTI-LEAKAGE")
    logger.info("=" * 50)
    
    # Chargement dataset v16 (20 features contextuelles)
    v16_path = "data/processed/v16_contextual_features_20250915_171540.csv"
    logger.info(f"📊 Dataset v16: {v16_path}")
    
    data = pd.read_csv(v16_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Features v16 (10 originales + 5 contextuelles)
    base_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    contextual_features = [
        'rest_days_diff', 'promoted_team_factor', 'early_season_volatility',
        'manager_continuity', 'transfer_window_impact'
    ]
    
    all_features = base_features + contextual_features
    
    logger.info(f"   Features base: {len(base_features)}")
    logger.info(f"   Features contextuelles: {len(contextual_features)}")
    logger.info(f"   Total features: {len(all_features)}")
    
    # Target mapping
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    data['target'] = data['FullTimeResult'].map(target_mapping)
    
    # Filtrage données valides
    valid_mask = data['target'].notna()
    data = data[valid_mask].sort_values('Date').reset_index(drop=True)
    
    logger.info(f"   Échantillons valides: {len(data)}")
    
    # Split temporel strict ANTI-LEAKAGE
    train_cutoff = pd.to_datetime('2025-05-25')
    test_start = pd.to_datetime('2025-08-15')
    
    train_mask = data['Date'] <= train_cutoff
    test_mask = data['Date'] >= test_start
    
    train_count = train_mask.sum()
    test_count = test_mask.sum()
    
    logger.info(f"   Train (≤ {train_cutoff.strftime('%Y-%m-%d')}): {train_count}")
    logger.info(f"   Test (≥ {test_start.strftime('%Y-%m-%d')}): {test_count}")
    
    # Comparaison 10 features vs 20 features
    logger.info("\\n🔬 COMPARAISON FEATURES")
    logger.info("=" * 30)
    
    results = {}
    
    for feature_set_name, features in [("10_base", base_features), ("20_contextual", all_features)]:
        logger.info(f"\\n  📊 Test {feature_set_name} ({len(features)} features)")
        
        # Données train
        train_data = data[train_mask]
        X_train = train_data[features].fillna(0)
        y_train = train_data['target'].astype(int)
        
        # Cross-validation historique
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Modèle
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42
            )
            
            calibrated_model = CalibratedClassifierCV(model, cv=3)
            calibrated_model.fit(X_fold_train, y_fold_train)
            
            predictions = calibrated_model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            cv_scores.append(accuracy)
            
            logger.info(f"    Fold {fold+1}: {accuracy:.3f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        logger.info(f"    CV Moyenne: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Test final EPL 2025-26
        test_data = data[test_mask]
        X_test = test_data[features].fillna(0)
        y_test = test_data['target'].astype(int)
        
        # Entraînement final sur tout le train set
        final_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        )
        
        final_calibrated = CalibratedClassifierCV(final_model, cv=3)
        final_calibrated.fit(X_train, y_train)
        
        # Prédictions test
        test_predictions = final_calibrated.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Distributions
        test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index() * 100
        pred_dist = pd.Series(test_predictions).value_counts(normalize=True).sort_index() * 100
        
        logger.info(f"    Test EPL 2025-26: {test_accuracy:.3f}")
        logger.info(f"    Réel:  H={test_dist.get(0, 0):4.1f}% D={test_dist.get(1, 0):4.1f}% A={test_dist.get(2, 0):4.1f}%")
        logger.info(f"    Pred:  H={pred_dist.get(0, 0):4.1f}% D={pred_dist.get(1, 0):4.1f}% A={pred_dist.get(2, 0):4.1f}%")
        
        results[feature_set_name] = {
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_accuracy,
            'test_distribution': test_dist.to_dict(),
            'pred_distribution': pred_dist.to_dict()
        }
    
    # Analyse features contextuelles spécifiques
    logger.info("\\n🎯 ANALYSE FEATURES CONTEXTUELLES")
    logger.info("=" * 40)
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    for feature in contextual_features:
        train_values = train_data[feature].fillna(0)
        test_values = test_data[feature].fillna(0)
        
        train_mean = train_values.mean()
        test_mean = test_values.mean()
        
        logger.info(f"   {feature}:")
        logger.info(f"     Train moyenne: {train_mean:.3f}")
        logger.info(f"     Test moyenne:  {test_mean:.3f}")
        logger.info(f"     Différence:    {test_mean - train_mean:+.3f}")
    
    # Comparaison finale
    logger.info("\\n🏆 COMPARAISON FINALE")
    logger.info("=" * 25)
    
    base_cv = results["10_base"]["cv_accuracy"]
    context_cv = results["20_contextual"]["cv_accuracy"]
    base_test = results["10_base"]["test_accuracy"]
    context_test = results["20_contextual"]["test_accuracy"]
    
    logger.info(f"   CV Historique:")
    logger.info(f"     10 features: {base_cv:.3f}")
    logger.info(f"     20 features: {context_cv:.3f}")
    logger.info(f"     Amélioration: {context_cv - base_cv:+.3f}")
    
    logger.info(f"   Test EPL 2025-26:")
    logger.info(f"     10 features: {base_test:.3f}")
    logger.info(f"     20 features: {context_test:.3f}")
    logger.info(f"     Amélioration: {context_test - base_test:+.3f}")
    
    # Verdict
    if context_test > 0.50:
        verdict = "✅ FEATURES CONTEXTUELLES EFFICACES"
    elif context_test > base_test:
        verdict = "⚠️ AMÉLIORATION MARGINALE"
    else:
        verdict = "❌ FEATURES CONTEXTUELLES INEFFICACES"
    
    logger.info(f"\\n🎯 VERDICT: {verdict}")
    
    # Tableau synthèse
    print(f"\\n🧪 TEST FEATURES V16 - SYNTHÈSE ANTI-LEAKAGE")
    print(f"\\n{'Feature Set':<15} {'CV Hist':<10} {'Test 2025':<12} {'Amélioration':<12}")
    print(f"{'='*15} {'='*10} {'='*12} {'='*12}")
    print(f"{'10 base':<15} {base_cv:.3f}      {base_test:.3f}        baseline")
    print(f"{'20 contextual':<15} {context_cv:.3f}      {context_test:.3f}        {context_test - base_test:+.3f}")
    print(f"\\n🎯 VERDICT: {verdict}")
    
    return results

if __name__ == "__main__":
    results = test_v16_features_antileakage()