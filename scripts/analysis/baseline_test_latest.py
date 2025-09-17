#!/usr/bin/env python3
"""
🔬 TEST BASELINE SUR DATASET RÉCENT
===================================
Test du RandomForest baseline sur le dataset auto-update le plus récent (10 features).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline_test")

def test_baseline_latest():
    """Test du baseline sur dataset le plus récent."""
    logger.info("🔬 TEST BASELINE SUR DATASET RÉCENT")
    logger.info("=" * 50)
    
    # Chargement données
    dataset_path = "data/processed/v_auto_update_20250916_110247.csv"
    logger.info(f"📊 Dataset: {dataset_path}")
    
    data = pd.read_csv(dataset_path)
    logger.info(f"   Échantillons: {len(data)}")
    
    # Features (10 features production)
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    logger.info(f"   Features: {len(features)}")
    for i, feat in enumerate(features, 1):
        logger.info(f"     {i:2d}. {feat}")
    
    # Préparation données
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    data['target'] = data['FullTimeResult'].map(target_mapping)
    
    # Filtrage et tri
    valid_mask = data['target'].notna()
    data = data[valid_mask].sort_values('Date').reset_index(drop=True)
    
    X = data[features].fillna(0)
    y = data['target'].astype(int)
    
    logger.info(f"   Données valides: {len(X)}")
    logger.info(f"   Distribution: H={np.mean(y==0):.1%}, D={np.mean(y==1):.1%}, A={np.mean(y==2):.1%}")
    
    # Cross-validation temporelle
    logger.info("\n🎯 CROSS-VALIDATION TEMPORELLE")
    tscv = TimeSeriesSplit(n_splits=5)
    
    baseline_scores = []
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        logger.info(f"\n  Fold {fold+1}/5:")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        logger.info(f"    Train: {len(X_train)} échantillons")
        logger.info(f"    Test:  {len(X_test)} échantillons")
        
        # Baseline RandomForest
        baseline_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        )
        
        # Avec calibration
        calibrated_model = CalibratedClassifierCV(baseline_model, cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = calibrated_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        baseline_scores.append(accuracy)
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test)
        })
        
        logger.info(f"    Accuracy: {accuracy:.3f}")
        
        # Distribution prédictions
        pred_dist = pd.Series(y_pred).value_counts(normalize=True).sort_index() * 100
        actual_dist = pd.Series(y_test).value_counts(normalize=True).sort_index() * 100
        
        logger.info(f"    Pred:  H={pred_dist.get(0, 0):.1f}% D={pred_dist.get(1, 0):.1f}% A={pred_dist.get(2, 0):.1f}%")
        logger.info(f"    Real:  H={actual_dist.get(0, 0):.1f}% D={actual_dist.get(1, 0):.1f}% A={actual_dist.get(2, 0):.1f}%")
    
    # Statistiques globales
    logger.info("\n📊 RÉSULTATS GLOBAUX")
    logger.info("=" * 30)
    
    mean_accuracy = np.mean(baseline_scores)
    std_accuracy = np.std(baseline_scores)
    
    logger.info(f"Accuracy moyenne: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    logger.info(f"Min accuracy:     {np.min(baseline_scores):.3f}")
    logger.info(f"Max accuracy:     {np.max(baseline_scores):.3f}")
    
    # Comparaison avec baselines naïfs
    logger.info("\n🎯 COMPARAISON BASELINES NAÏFS")
    majority_class_acc = np.mean(y == 0)  # Always predict Home
    random_acc = 1/3  # Random prediction
    
    logger.info(f"Random (33.3%):     {random_acc:.3f}")
    logger.info(f"Majority class:     {majority_class_acc:.3f}")
    logger.info(f"Notre modèle:       {mean_accuracy:.3f}")
    logger.info(f"Boost vs majority:  {mean_accuracy - majority_class_acc:+.3f}")
    logger.info(f"Boost vs random:    {mean_accuracy - random_acc:+.3f}")
    
    # Verdict
    logger.info("\n🏆 VERDICT")
    if mean_accuracy > 0.52:
        verdict = "✅ EXCELLENT"
    elif mean_accuracy > 0.50:
        verdict = "🎯 BON"
    elif mean_accuracy > majority_class_acc:
        verdict = "⚠️  ACCEPTABLE"
    else:
        verdict = "❌ INSUFFISANT"
    
    logger.info(f"Performance: {verdict}")
    logger.info(f"Variance: {'✅ Stable' if std_accuracy < 0.03 else '⚠️ Variable'}")
    
    # Tableau résumé
    print(f"\n🔬 BASELINE TEST - DATASET 10 FEATURES")
    print(f"\n{'Fold':<6} {'Accuracy':<10} {'Train':<8} {'Test':<8}")
    print(f"{'='*6} {'='*10} {'='*8} {'='*8}")
    
    for result in fold_results:
        print(f"{result['fold']:<6} {result['accuracy']:.3f}      {result['train_size']:<8} {result['test_size']:<8}")
    
    print(f"\n📊 MOYENNE: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"🎯 VERDICT: {verdict}")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'fold_results': fold_results,
        'verdict': verdict
    }

if __name__ == "__main__":
    results = test_baseline_latest()