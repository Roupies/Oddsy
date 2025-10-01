#!/usr/bin/env python3
"""
🎯 TEST CASCADE OPTIMAL SUR 50 MATCHS EPL 2025-26
================================================
Test du cascade avec paramètres optimaux sur tous les matchs EPL 2025-26 disponibles.
Comparaison avec baseline pour validation finale.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import logging
import sys

# Import cascade model
sys.path.append('scripts/final')
from cascade_model_production import CascadeModelProduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cascade_test_50")

def test_cascade_optimal_50_matchs():
    """Test cascade optimal vs baseline sur 50 matchs EPL 2025-26."""
    logger.info("🎯 TEST CASCADE OPTIMAL - 50 MATCHS EPL 2025-26")
    logger.info("=" * 60)
    
    # Chargement données
    dataset_path = "data/processed/v_auto_update_20250916_110247.csv"
    logger.info(f"📊 Dataset: {dataset_path}")
    
    data = pd.read_csv(dataset_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Features production
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    # Target mapping
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    data['target'] = data['FullTimeResult'].map(target_mapping)
    
    # Filtrage données valides
    valid_mask = data['target'].notna()
    data = data[valid_mask].sort_values('Date').reset_index(drop=True)
    
    logger.info(f"   Total échantillons: {len(data)}")
    
    # Split temporel strict ANTI-LEAKAGE
    train_cutoff = pd.to_datetime('2025-05-25')
    test_start = pd.to_datetime('2025-08-15')
    
    train_mask = data['Date'] <= train_cutoff
    test_mask = data['Date'] >= test_start
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    logger.info(f"   Train: {len(train_data)} échantillons (≤ {train_cutoff.strftime('%Y-%m-%d')})")
    logger.info(f"   Test EPL 2025-26: {len(test_data)} échantillons (≥ {test_start.strftime('%Y-%m-%d')})")
    
    # Préparation données
    X_train = train_data[features].fillna(0)
    y_train = train_data['target'].astype(int)
    X_test = test_data[features].fillna(0)
    y_test = test_data['target'].astype(int)
    
    # Distribution réelle EPL 2025-26
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100
    logger.info(f"   Distribution EPL 2025-26: H={test_dist.get(0, 0):.1f}% D={test_dist.get(1, 0):.1f}% A={test_dist.get(2, 0):.1f}%")
    
    # 1. TEST CASCADE OPTIMAL
    logger.info("\\n🎯 TEST CASCADE OPTIMAL")
    logger.info("=" * 30)
    
    # Paramètres optimaux cascade (de l'analyse précédente)
    cascade_model = CascadeModelProduction(
        draw_weight=3.0,              # Optimal identifié
        draw_threshold=0.35,          # Optimal identifié  
        calibration_factor=0.85,      # Optimal identifié
        random_state=42
    )
    
    logger.info(f"   Paramètres cascade:")
    logger.info(f"     draw_weight: {cascade_model.draw_weight}")
    logger.info(f"     draw_threshold: {cascade_model.draw_threshold}")
    logger.info(f"     calibration_factor: {cascade_model.calibration_factor}")
    
    # Entraînement cascade
    cascade_model.fit(X_train, y_train)
    
    # Prédictions cascade
    cascade_preds = cascade_model.predict(X_test)
    cascade_accuracy = accuracy_score(y_test, cascade_preds)
    
    # Distribution prédictions cascade
    cascade_dist = pd.Series(cascade_preds).value_counts(normalize=True).sort_index() * 100
    
    logger.info(f"\\n   📊 RÉSULTATS CASCADE:")
    logger.info(f"     Accuracy: {cascade_accuracy:.3f}")
    logger.info(f"     Pred: H={cascade_dist.get('H', 0):.1f}% D={cascade_dist.get('D', 0):.1f}% A={cascade_dist.get('A', 0):.1f}%")
    
    # 2. TEST BASELINE OPTIMAL  
    logger.info("\\n📊 TEST BASELINE OPTIMAL")
    logger.info("=" * 30)
    
    baseline_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )
    
    # Avec calibration
    calibrated_baseline = CalibratedClassifierCV(baseline_model, cv=3)
    calibrated_baseline.fit(X_train, y_train)
    
    # Prédictions baseline
    baseline_preds = calibrated_baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    
    # Distribution prédictions baseline
    baseline_pred_dist = pd.Series(baseline_preds).value_counts(normalize=True).sort_index() * 100
    
    logger.info(f"\\n   📊 RÉSULTATS BASELINE:")
    logger.info(f"     Accuracy: {baseline_accuracy:.3f}")
    logger.info(f"     Pred: H={baseline_pred_dist.get(0, 0):.1f}% D={baseline_pred_dist.get(1, 0):.1f}% A={baseline_pred_dist.get(2, 0):.1f}%")
    
    # 3. COMPARAISON DÉTAILLÉE
    logger.info("\\n⚖️  COMPARAISON DÉTAILLÉE")
    logger.info("=" * 35)
    
    accuracy_diff = cascade_accuracy - baseline_accuracy
    logger.info(f"   Cascade:  {cascade_accuracy:.3f}")
    logger.info(f"   Baseline: {baseline_accuracy:.3f}")
    logger.info(f"   Différence: {accuracy_diff:+.3f}")
    
    # Matrice confusion cascade
    y_test_str = pd.Series(y_test).map({0: 'H', 1: 'D', 2: 'A'})
    cm_cascade = confusion_matrix(y_test_str, cascade_preds, labels=['H', 'D', 'A'])
    
    logger.info(f"\\n   📊 MATRICE CONFUSION CASCADE:")
    logger.info(f"         Pred: H    D    A")
    for i, (real_label, row) in enumerate(zip(['H', 'D', 'A'], cm_cascade)):
        logger.info(f"   Real {real_label}:    {row[0]:2d}   {row[1]:2d}   {row[2]:2d}")
    
    # Matrice confusion baseline
    baseline_preds_str = pd.Series(baseline_preds).map({0: 'H', 1: 'D', 2: 'A'})
    cm_baseline = confusion_matrix(y_test_str, baseline_preds_str, labels=['H', 'D', 'A'])
    
    logger.info(f"\\n   📊 MATRICE CONFUSION BASELINE:")
    logger.info(f"         Pred: H    D    A")
    for i, (real_label, row) in enumerate(zip(['H', 'D', 'A'], cm_baseline)):
        logger.info(f"   Real {real_label}:    {row[0]:2d}   {row[1]:2d}   {row[2]:2d}")
    
    # 4. ANALYSE PAR CLASSE
    logger.info("\\n🎯 ANALYSE PAR CLASSE")
    logger.info("=" * 25)
    
    for outcome in ['H', 'D', 'A']:
        outcome_mask = y_test_str == outcome
        if outcome_mask.sum() > 0:
            # Conversion pour éviter problème d'index
            cascade_preds_series = pd.Series(cascade_preds, index=y_test_str.index)
            baseline_preds_series = pd.Series(baseline_preds_str, index=y_test_str.index)
            
            cascade_correct = (cascade_preds_series == outcome)[outcome_mask].sum()
            baseline_correct = (baseline_preds_series == outcome)[outcome_mask].sum()
            total = outcome_mask.sum()
            
            cascade_recall = cascade_correct / total
            baseline_recall = baseline_correct / total
            
            logger.info(f"   {outcome}: Cascade {cascade_recall:.3f}, Baseline {baseline_recall:.3f} (Δ{cascade_recall - baseline_recall:+.3f})")
    
    # 5. COMPARAISON AVEC BASELINES NAÏFS
    logger.info("\\n🏆 COMPARAISON BASELINES NAÏFS")
    logger.info("=" * 35)
    
    majority_acc = np.mean(y_test == 0)  # Always predict Home
    random_acc = 1/3
    
    logger.info(f"   Random (33.3%):       {random_acc:.3f}")
    logger.info(f"   Majority class Home:  {majority_acc:.3f}")
    logger.info(f"   Cascade optimal:      {cascade_accuracy:.3f}")
    logger.info(f"   Baseline optimal:     {baseline_accuracy:.3f}")
    
    # Boosts
    cascade_boost_maj = cascade_accuracy - majority_acc
    baseline_boost_maj = baseline_accuracy - majority_acc
    cascade_boost_rand = cascade_accuracy - random_acc
    baseline_boost_rand = baseline_accuracy - random_acc
    
    logger.info(f"\\n   Boost vs Majority:")
    logger.info(f"     Cascade:  {cascade_boost_maj:+.3f}")
    logger.info(f"     Baseline: {baseline_boost_maj:+.3f}")
    
    logger.info(f"\\n   Boost vs Random:")
    logger.info(f"     Cascade:  {cascade_boost_rand:+.3f}")
    logger.info(f"     Baseline: {baseline_boost_rand:+.3f}")
    
    # 6. VERDICT FINAL
    logger.info("\\n🏆 VERDICT FINAL EPL 2025-26")
    logger.info("=" * 35)
    
    if cascade_accuracy > baseline_accuracy + 0.02:
        winner = "✅ CASCADE GAGNE"
        recommendation = "UTILISER CASCADE POUR EPL 2025-26"
    elif baseline_accuracy > cascade_accuracy + 0.02:
        winner = "✅ BASELINE GAGNE"
        recommendation = "UTILISER BASELINE POUR EPL 2025-26"
    else:
        winner = "⚖️ ÉGALITÉ"
        recommendation = "PERFORMANCE ÉQUIVALENTE"
    
    # Production readiness
    if max(cascade_accuracy, baseline_accuracy) > majority_acc:
        production_status = "✅ PRODUCTION VIABLE"
    else:
        production_status = "❌ NON PRODUCTION (< majority class)"
    
    logger.info(f"   Gagnant: {winner}")
    logger.info(f"   Recommandation: {recommendation}")
    logger.info(f"   Production: {production_status}")
    
    # Tableau synthèse final
    print(f"\\n🎯 TEST CASCADE vs BASELINE - 50 MATCHS EPL 2025-26")
    print(f"\\n{'Modèle':<15} {'Accuracy':<10} {'vs Majority':<12} {'vs Random':<10}")
    print(f"{'='*15} {'='*10} {'='*12} {'='*10}")
    print(f"{'Cascade':<15} {cascade_accuracy:.3f}      {cascade_boost_maj:+.3f}        {cascade_boost_rand:+.3f}")
    print(f"{'Baseline':<15} {baseline_accuracy:.3f}      {baseline_boost_maj:+.3f}        {baseline_boost_rand:+.3f}")
    print(f"{'Majority':<15} {majority_acc:.3f}      {0:+.3f}        {majority_acc - random_acc:+.3f}")
    print(f"{'Random':<15} {random_acc:.3f}      {random_acc - majority_acc:+.3f}        {0:+.3f}")
    print(f"\\n🏆 VERDICT: {winner}")
    print(f"🚀 PRODUCTION: {production_status}")
    
    return {
        'cascade_accuracy': cascade_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'majority_accuracy': majority_acc,
        'winner': winner,
        'production_status': production_status,
        'cascade_distribution': cascade_dist.to_dict(),
        'baseline_distribution': baseline_pred_dist.to_dict(),
        'test_samples': len(test_data)
    }

if __name__ == "__main__":
    results = test_cascade_optimal_50_matchs()