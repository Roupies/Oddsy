#!/usr/bin/env python3
"""
🔍 DIAGNOSTIC OVERFITTING TEMPOREL
==================================
Analyse du gap performance entre CV historique et test EPL 2025-26.
Identification des causes profondes de l'échec de généralisation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import logging
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnostic")

def diagnostic_overfitting_temporel():
    """Diagnostic complet de l'overfitting temporel."""
    logger.info("🔍 DIAGNOSTIC OVERFITTING TEMPOREL")
    logger.info("=" * 50)
    
    # Chargement données
    dataset_path = "data/processed/v_auto_update_20250916_110247.csv"
    data = pd.read_csv(dataset_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    # Target mapping
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    data['target'] = data['FullTimeResult'].map(target_mapping)
    
    # Filtrage et tri
    valid_mask = data['target'].notna()
    data = data[valid_mask].sort_values('Date').reset_index(drop=True)
    
    logger.info(f"📊 Dataset: {len(data)} échantillons")
    logger.info(f"   Période: {data['Date'].min().strftime('%Y-%m-%d')} → {data['Date'].max().strftime('%Y-%m-%d')}")
    
    # Split temporel
    train_cutoff = pd.to_datetime('2025-05-25')
    test_start = pd.to_datetime('2025-08-15')
    
    train_mask = data['Date'] <= train_cutoff
    test_mask = data['Date'] >= test_start
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    logger.info(f"   Train: {len(train_data)} échantillons (≤ {train_cutoff.strftime('%Y-%m-%d')})")
    logger.info(f"   Test:  {len(test_data)} échantillons (≥ {test_start.strftime('%Y-%m-%d')})")
    
    # 1. ANALYSE DISTRIBUTIONS
    logger.info("\\n📊 ANALYSE DISTRIBUTIONS H/D/A")
    logger.info("=" * 35)
    
    train_dist = train_data['target'].value_counts(normalize=True).sort_index() * 100
    test_dist = test_data['target'].value_counts(normalize=True).sort_index() * 100
    
    logger.info(f"   Train:  H={train_dist.get(0, 0):5.1f}% D={train_dist.get(1, 0):5.1f}% A={train_dist.get(2, 0):5.1f}%")
    logger.info(f"   Test:   H={test_dist.get(0, 0):5.1f}% D={test_dist.get(1, 0):5.1f}% A={test_dist.get(2, 0):5.1f}%")
    logger.info(f"   Δ:      H={test_dist.get(0, 0) - train_dist.get(0, 0):+5.1f}% D={test_dist.get(1, 0) - train_dist.get(1, 0):+5.1f}% A={test_dist.get(2, 0) - train_dist.get(2, 0):+5.1f}%")
    
    # 2. ANALYSE FEATURES DRIFT
    logger.info("\\n🌊 ANALYSE DRIFT FEATURES")
    logger.info("=" * 25)
    
    feature_drifts = []
    
    for feature in features:
        train_values = train_data[feature].fillna(0)
        test_values = test_data[feature].fillna(0)
        
        train_mean = train_values.mean()
        test_mean = test_values.mean()
        train_std = train_values.std()
        
        # Drift normalisé (en écarts-types)
        normalized_drift = abs(test_mean - train_mean) / (train_std + 1e-8)
        
        feature_drifts.append({
            'feature': feature,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'drift_abs': abs(test_mean - train_mean),
            'drift_normalized': normalized_drift
        })
        
        logger.info(f"   {feature[:20]:<20}: {train_mean:6.3f} → {test_mean:6.3f} (Δ{test_mean - train_mean:+6.3f}, {normalized_drift:.2f}σ)")
    
    # Top drifts
    feature_drifts.sort(key=lambda x: x['drift_normalized'], reverse=True)
    logger.info(f"\\n   🚨 TOP DRIFTS:")
    for i, drift in enumerate(feature_drifts[:3]):
        logger.info(f"     {i+1}. {drift['feature']}: {drift['drift_normalized']:.2f}σ")
    
    # 3. PERFORMANCE TEMPORELLE
    logger.info("\\n📈 PERFORMANCE PAR ANNÉE")
    logger.info("=" * 25)
    
    # Split par année
    data['year'] = data['Date'].dt.year
    years = sorted(data['year'].unique())
    
    yearly_performance = []
    
    for year in years[-5:]:  # 5 dernières années
        year_data = data[data['year'] == year]
        if len(year_data) < 50:  # Pas assez de données
            continue
            
        # Split avant cette année (train) vs cette année (test)
        before_year = data[data['year'] < year]
        
        if len(before_year) < 500:  # Pas assez d'historique
            continue
        
        X_before = before_year[features].fillna(0)
        y_before = before_year['target'].astype(int)
        X_year = year_data[features].fillna(0)
        y_year = year_data['target'].astype(int)
        
        # Entraînement
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        )
        
        calibrated_model = CalibratedClassifierCV(model, cv=3)
        calibrated_model.fit(X_before, y_before)
        
        # Test
        predictions = calibrated_model.predict(X_year)
        accuracy = accuracy_score(y_year, predictions)
        
        yearly_performance.append({
            'year': year,
            'accuracy': accuracy,
            'samples': len(year_data),
            'train_samples': len(before_year)
        })
        
        logger.info(f"   {year}: {accuracy:.3f} ({len(year_data)} échantillons)")
    
    # 4. DIAGNOSTIC EARLY SEASON
    logger.info("\\n🎯 DIAGNOSTIC EARLY SEASON (J1-J4)")
    logger.info("=" * 35)
    
    # Seulement début de saison
    early_season_mask = data['matchday_normalized'] <= 0.15
    early_data = data[early_season_mask]
    
    early_train = early_data[early_data['Date'] <= train_cutoff]
    early_test = early_data[early_data['Date'] >= test_start]
    
    logger.info(f"   Early season train: {len(early_train)} échantillons")
    logger.info(f"   Early season test:  {len(early_test)} échantillons")
    
    if len(early_test) > 5:
        # Distribution early season
        early_train_dist = early_train['target'].value_counts(normalize=True).sort_index() * 100
        early_test_dist = early_test['target'].value_counts(normalize=True).sort_index() * 100
        
        logger.info(f"   Early Train: H={early_train_dist.get(0, 0):5.1f}% D={early_train_dist.get(1, 0):5.1f}% A={early_train_dist.get(2, 0):5.1f}%")
        logger.info(f"   Early Test:  H={early_test_dist.get(0, 0):5.1f}% D={early_test_dist.get(1, 0):5.1f}% A={early_test_dist.get(2, 0):5.1f}%")
    
    # 5. CONCLUSION DIAGNOSTIC
    logger.info("\\n🎯 CONCLUSION DIAGNOSTIC")
    logger.info("=" * 25)
    
    # Problèmes identifiés
    problemes = []
    
    # Distribution shift
    home_shift = abs(test_dist.get(0, 0) - train_dist.get(0, 0))
    if home_shift > 5:
        problemes.append(f"Distribution shift majeur (H: {home_shift:+.1f}%)")
    
    # Feature drift
    max_drift = max(feature_drifts, key=lambda x: x['drift_normalized'])
    if max_drift['drift_normalized'] > 1.0:
        problemes.append(f"Feature drift: {max_drift['feature']} ({max_drift['drift_normalized']:.1f}σ)")
    
    # Overfitting temporel
    recent_performance = [p['accuracy'] for p in yearly_performance[-2:]]  # 2 dernières années
    if recent_performance and np.mean(recent_performance) < 0.48:
        problemes.append(f"Dégradation temporelle continue ({np.mean(recent_performance):.3f})")
    
    # Sample size
    if len(test_data) < 40:
        problemes.append(f"Test set trop petit ({len(test_data)} échantillons)")
    
    logger.info(f"   🚨 PROBLÈMES IDENTIFIÉS:")
    for i, probleme in enumerate(problemes, 1):
        logger.info(f"     {i}. {probleme}")
    
    # Recommandations
    logger.info(f"\\n💡 RECOMMANDATIONS:")
    if home_shift > 5:
        logger.info(f"     1. Recalibrer pour distribution EPL 2025-26 (plus de Home wins)")
    if max_drift['drift_normalized'] > 1.0:
        logger.info(f"     2. Re-engineering feature {max_drift['feature']}")
    if len(test_data) < 40:
        logger.info(f"     3. Attendre plus de matchs EPL 2025-26 pour validation fiable")
    
    logger.info(f"     4. Considérer approche conservative (majority class baseline)")
    
    # Verdict final
    if len(problemes) >= 3:
        verdict = "❌ OVERFITTING TEMPOREL MAJEUR"
        recommendation = "ATTENDRE PLUS DE DONNÉES EPL 2025-26"
    elif home_shift > 7:
        verdict = "⚠️ DISTRIBUTION SHIFT CRITIQUE"
        recommendation = "RECALIBRATION URGENTE REQUISE"
    else:
        verdict = "⚠️ PROBLÈMES TEMPORELS MODÉRÉS"
        recommendation = "OPTIMISATION CIBLÉE POSSIBLE"
    
    logger.info(f"\\n🏆 VERDICT: {verdict}")
    logger.info(f"🚀 RECOMMANDATION: {recommendation}")
    
    # Tableau final
    print(f"\\n🔍 DIAGNOSTIC OVERFITTING TEMPOREL - SYNTHÈSE")
    print(f"\\n📊 Distributions:")
    print(f"   Train:  H={train_dist.get(0, 0):5.1f}% D={train_dist.get(1, 0):5.1f}% A={train_dist.get(2, 0):5.1f}%")
    print(f"   Test:   H={test_dist.get(0, 0):5.1f}% D={test_dist.get(1, 0):5.1f}% A={test_dist.get(2, 0):5.1f}%")
    print(f"\\n🌊 Top Feature Drifts:")
    for i, drift in enumerate(feature_drifts[:3]):
        print(f"   {i+1}. {drift['feature']:<25}: {drift['drift_normalized']:5.2f}σ")
    print(f"\\n🎯 VERDICT: {verdict}")
    print(f"🚀 ACTION: {recommendation}")
    
    return {
        'verdict': verdict,
        'recommendation': recommendation,
        'problemes': problemes,
        'feature_drifts': feature_drifts,
        'yearly_performance': yearly_performance
    }

if __name__ == "__main__":
    results = diagnostic_overfitting_temporel()