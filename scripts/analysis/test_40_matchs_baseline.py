#!/usr/bin/env python3
"""
🎯 TEST BASELINE SUR 40 MATCHS EPL 2025-26
=========================================
Test du baseline RandomForest sur les 40 premiers matchs de la saison 2025-26.
Comparaison avec performance historique pour validation production.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_40_matchs")

def test_baseline_40_matchs():
    """Test baseline sur 40 matchs EPL 2025-26."""
    logger.info("🎯 TEST BASELINE SUR 40 MATCHS EPL 2025-26")
    logger.info("=" * 50)
    
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
    
    # Préparation target
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    data['target'] = data['FullTimeResult'].map(target_mapping)
    
    # Filtrage données valides
    valid_mask = data['target'].notna()
    data = data[valid_mask].sort_values('Date').reset_index(drop=True)
    
    logger.info(f"   Total échantillons: {len(data)}")
    
    # Identification EPL 2025-26 (derniers matchs)
    epl_2025_start = pd.to_datetime('2025-08-01')
    epl_2025_mask = data['Date'] >= epl_2025_start
    epl_2025_count = epl_2025_mask.sum()
    
    logger.info(f"   Matchs EPL 2025-26: {epl_2025_count}")
    
    if epl_2025_count < 30:
        logger.warning(f"⚠️ Seulement {epl_2025_count} matchs EPL 2025-26 détectés")
        # Prendre les derniers matchs comme proxy
        test_data = data.tail(40)
        logger.info(f"   Utilisation des 40 derniers matchs comme test")
    else:
        # Prendre les 40 premiers matchs EPL 2025-26
        epl_data = data[epl_2025_mask].head(40)
        test_data = epl_data
        logger.info(f"   Test sur 40 premiers matchs EPL 2025-26")
    
    # Split train/test temporel
    test_start_idx = data.index[data['Date'] == test_data.iloc[0]['Date']].min()
    train_data = data.iloc[:test_start_idx]
    
    logger.info(f"   Train: {len(train_data)} échantillons (jusqu'à {train_data.iloc[-1]['Date'].strftime('%Y-%m-%d')})")
    logger.info(f"   Test:  {len(test_data)} échantillons (depuis {test_data.iloc[0]['Date'].strftime('%Y-%m-%d')})")
    
    # Préparation features
    X_train = train_data[features].fillna(0)
    y_train = train_data['target'].astype(int)
    X_test = test_data[features].fillna(0)
    y_test = test_data['target'].astype(int)
    
    # Distribution train
    train_dist = y_train.value_counts(normalize=True).sort_index() * 100
    logger.info(f"   Train distribution: H={train_dist.get(0, 0):.1f}% D={train_dist.get(1, 0):.1f}% A={train_dist.get(2, 0):.1f}%")
    
    # Distribution test (vérité terrain)
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100
    logger.info(f"   Test distribution:  H={test_dist.get(0, 0):.1f}% D={test_dist.get(1, 0):.1f}% A={test_dist.get(2, 0):.1f}%")
    
    # Entraînement baseline
    logger.info("\n🔧 ENTRAÎNEMENT BASELINE")
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
    logger.info("\n🎯 PRÉDICTIONS SUR 40 MATCHS")
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"   Accuracy: {accuracy:.3f}")
    
    # Distribution prédictions
    pred_dist = pd.Series(y_pred).value_counts(normalize=True).sort_index() * 100
    logger.info(f"   Pred distribution:  H={pred_dist.get(0, 0):.1f}% D={pred_dist.get(1, 0):.1f}% A={pred_dist.get(2, 0):.1f}%")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n📊 MATRICE DE CONFUSION:")
    logger.info(f"         Pred: H    D    A")
    for i, (real_label, row) in enumerate(zip(['H', 'D', 'A'], cm)):
        logger.info(f"   Real {real_label}:    {row[0]:2d}   {row[1]:2d}   {row[2]:2d}")
    
    # Métriques par classe
    logger.info(f"\n🎯 MÉTRIQUES PAR CLASSE:")
    class_names = ['H', 'D', 'A']
    for i, class_name in enumerate(class_names):
        if i < len(cm) and i < len(cm[0]):
            tp = cm[i, i] if i < len(cm) and i < len(cm[0]) else 0
            fp = cm[:, i].sum() - tp if i < len(cm[0]) else 0
            fn = cm[i, :].sum() - tp if i < len(cm) else 0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f"   {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Comparaison avec baselines
    logger.info(f"\n🏆 COMPARAISON BASELINES:")
    majority_acc = np.mean(y_test == 0)  # Always predict Home
    random_acc = 1/3
    
    logger.info(f"   Random (33.3%):       {random_acc:.3f}")
    logger.info(f"   Majority class Home:  {majority_acc:.3f}")
    logger.info(f"   Notre baseline:       {accuracy:.3f}")
    logger.info(f"   Boost vs majority:    {accuracy - majority_acc:+.3f}")
    logger.info(f"   Boost vs random:      {accuracy - random_acc:+.3f}")
    
    # Analyse confidence
    logger.info(f"\n🎲 ANALYSE CONFIDENCE:")
    max_probas = np.max(y_proba, axis=1)
    avg_confidence = np.mean(max_probas)
    logger.info(f"   Confidence moyenne:   {avg_confidence:.3f}")
    logger.info(f"   Confidence min:       {np.min(max_probas):.3f}")
    logger.info(f"   Confidence max:       {np.max(max_probas):.3f}")
    
    # Verdict final
    logger.info(f"\n🏆 VERDICT FINAL")
    if accuracy > 0.55:
        verdict = "✅ EXCELLENT"
        production_ready = "🚀 PRODUCTION READY"
    elif accuracy > 0.50:
        verdict = "🎯 BON"
        production_ready = "✅ VALIDÉ PRODUCTION"
    elif accuracy > majority_acc:
        verdict = "⚠️ ACCEPTABLE"
        production_ready = "⚠️ PRODUCTION CONDITIONNELLE"
    else:
        verdict = "❌ INSUFFISANT"
        production_ready = "❌ NON PRODUCTION"
    
    logger.info(f"   Performance 40 matchs: {verdict}")
    logger.info(f"   Statut production:     {production_ready}")
    
    # Tableau de synthèse
    print(f"\n🎯 TEST 40 MATCHS EPL 2025-26 - BASELINE RF")
    print(f"\n📊 RÉSULTATS:")
    print(f"   Accuracy:              {accuracy:.3f}")
    print(f"   Confidence moyenne:    {avg_confidence:.3f}")
    print(f"   Boost vs majority:     {accuracy - majority_acc:+.3f}")
    print(f"\n📈 DISTRIBUTIONS:")
    print(f"   Réel:  H={test_dist.get(0, 0):4.1f}% D={test_dist.get(1, 0):4.1f}% A={test_dist.get(2, 0):4.1f}%")
    print(f"   Pred:  H={pred_dist.get(0, 0):4.1f}% D={pred_dist.get(1, 0):4.1f}% A={pred_dist.get(2, 0):4.1f}%")
    print(f"\n🏆 VERDICT: {verdict}")
    print(f"🚀 PRODUCTION: {production_ready}")
    
    return {
        'accuracy': accuracy,
        'verdict': verdict,
        'production_ready': production_ready,
        'confidence': avg_confidence,
        'test_distribution': test_dist.to_dict(),
        'pred_distribution': pred_dist.to_dict()
    }

if __name__ == "__main__":
    results = test_baseline_40_matchs()