#!/usr/bin/env python3
"""
🎯 TEST CASCADE EXACT 52.5%
===========================
Reproduction exacte du cascade qui avait donné 52.5% sur 40 matchs EPL 2025-26.
Paramètres: draw_weight=2.5, draw_threshold=0.40
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cascade_525")

class CascadeModel525:
    """Reproduction exacte du cascade 52.5%."""
    
    def __init__(self, draw_weight=2.5, draw_threshold=0.40):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight={0: 1, 1: draw_weight}, random_state=42
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )
        self.draw_threshold = draw_threshold
    
    def fit(self, X, y):
        # Conversion vers classes string
        if y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        logger.info(f"🔧 Entraînement cascade 52.5% sur {len(X)} échantillons")
        logger.info(f"   draw_weight: {self.clf_draw.class_weight[1]}")
        logger.info(f"   draw_threshold: {self.draw_threshold}")
        
        # 1. Draw Forest
        y_draw = (y_str == 'D').astype(int)
        draw_dist = y_draw.value_counts(normalize=True) * 100
        logger.info(f"   Distribution draws: {draw_dist.get(1, 0):.1f}% draws")
        
        self.clf_draw.fit(X, y_draw)
        
        # 2. Home/Away Forest
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            # Filtrage des NaN dans y_homeaway
            valid_homeaway = y_homeaway.notna()
            if valid_homeaway.sum() > 5:
                logger.info(f"   Home/Away entraîné sur {valid_homeaway.sum()} échantillons")
                self.clf_homeaway.fit(X_notdraw[valid_homeaway], y_homeaway[valid_homeaway])
    
    def predict(self, X):
        # 1. Prédiction draws
        draw_proba = self.clf_draw.predict_proba(X)[:, 1]
        
        # 2. Prédiction home/away
        homeaway_proba = self.clf_homeaway.predict_proba(X)[:, 1]  # Proba Home
        
        # 3. Logique cascade
        predictions = []
        for i in range(len(X)):
            if draw_proba[i] > self.draw_threshold:
                predictions.append('D')
            else:
                if homeaway_proba[i] > 0.5:
                    predictions.append('H')
                else:
                    predictions.append('A')
        
        return np.array(predictions)

def test_cascade_525_exact():
    """Test exact du cascade 52.5%."""
    logger.info("🎯 TEST CASCADE EXACT 52.5%")
    logger.info("=" * 40)
    
    # Même dataset que le test original
    dataset_path = "data/processed/v_auto_update_20250916_110247.csv"
    logger.info(f"📊 Dataset: {dataset_path}")
    
    data = pd.read_csv(dataset_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Features identiques
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
    
    # Split temporel strict (même que test original)
    train_cutoff = pd.to_datetime('2025-05-25')
    test_start = pd.to_datetime('2025-08-15')
    
    train_mask = data['Date'] <= train_cutoff
    test_mask = data['Date'] >= test_start
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    logger.info(f"   Train: {len(train_data)} échantillons")
    logger.info(f"   Test EPL 2025-26: {len(test_data)} échantillons")
    
    # Préparation données
    X_train = train_data[features].fillna(0)
    y_train = train_data['target'].astype(int)
    X_test = test_data[features].fillna(0)
    y_test = test_data['target'].astype(int)
    
    # Distribution test
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100
    logger.info(f"   Distribution test: H={test_dist.get(0, 0):.1f}% D={test_dist.get(1, 0):.1f}% A={test_dist.get(2, 0):.1f}%")
    
    # 1. TEST CASCADE 52.5% EXACT
    logger.info("\\n🎯 CASCADE 52.5% (draw_weight=2.5, threshold=0.40)")
    
    cascade_525 = CascadeModel525(draw_weight=2.5, draw_threshold=0.40)
    cascade_525.fit(X_train, y_train)
    
    # Prédictions
    cascade_preds = cascade_525.predict(X_test)
    cascade_accuracy = accuracy_score(y_test.map({0: 'H', 1: 'D', 2: 'A'}), cascade_preds)
    
    # Distribution prédictions
    cascade_dist = pd.Series(cascade_preds).value_counts(normalize=True).sort_index() * 100
    
    logger.info(f"\\n   📊 RÉSULTATS CASCADE 52.5%:")
    logger.info(f"     Accuracy: {cascade_accuracy:.3f}")
    logger.info(f"     Pred: H={cascade_dist.get('H', 0):.1f}% D={cascade_dist.get('D', 0):.1f}% A={cascade_dist.get('A', 0):.1f}%")
    
    # 2. COMPARAISON AVEC BASELINE
    logger.info("\\n📊 BASELINE POUR COMPARAISON")
    
    baseline_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )
    
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    
    # Conversion baseline pour comparaison
    baseline_preds_str = pd.Series(baseline_preds).map({0: 'H', 1: 'D', 2: 'A'})
    baseline_dist = baseline_preds_str.value_counts(normalize=True).sort_index() * 100
    
    logger.info(f"\\n   📊 RÉSULTATS BASELINE:")
    logger.info(f"     Accuracy: {baseline_accuracy:.3f}")
    logger.info(f"     Pred: H={baseline_dist.get('H', 0):.1f}% D={baseline_dist.get('D', 0):.1f}% A={baseline_dist.get('A', 0):.1f}%")
    
    # 3. COMPARAISON DÉTAILLÉE
    logger.info("\\n⚖️  COMPARAISON FINALE")
    logger.info("=" * 25)
    
    accuracy_diff = cascade_accuracy - baseline_accuracy
    logger.info(f"   Cascade 52.5%: {cascade_accuracy:.3f}")
    logger.info(f"   Baseline:      {baseline_accuracy:.3f}")
    logger.info(f"   Différence:    {accuracy_diff:+.3f}")
    
    # Matrice confusion cascade
    y_test_str = y_test.map({0: 'H', 1: 'D', 2: 'A'})
    cm_cascade = confusion_matrix(y_test_str, cascade_preds, labels=['H', 'D', 'A'])
    
    logger.info(f"\\n   📊 MATRICE CONFUSION CASCADE:")
    logger.info(f"         Pred: H    D    A")
    for i, (real_label, row) in enumerate(zip(['H', 'D', 'A'], cm_cascade)):
        logger.info(f"   Real {real_label}:    {row[0]:2d}   {row[1]:2d}   {row[2]:2d}")
    
    # Comparaison avec majority class
    majority_acc = np.mean(y_test == 0)  # Always predict Home
    
    logger.info(f"\\n🏆 COMPARAISON BASELINES:")
    logger.info(f"   Majority class: {majority_acc:.3f}")
    logger.info(f"   Cascade 52.5%:  {cascade_accuracy:.3f} ({cascade_accuracy - majority_acc:+.3f})")
    logger.info(f"   Baseline:       {baseline_accuracy:.3f} ({baseline_accuracy - majority_acc:+.3f})")
    
    # Verdict final
    if cascade_accuracy > baseline_accuracy + 0.02:
        winner = "✅ CASCADE 52.5% GAGNE"
    elif baseline_accuracy > cascade_accuracy + 0.02:
        winner = "✅ BASELINE GAGNE"
    else:
        winner = "⚖️ ÉGALITÉ"
    
    if max(cascade_accuracy, baseline_accuracy) > majority_acc:
        production_status = "✅ PRODUCTION VIABLE"
    else:
        production_status = "❌ NON PRODUCTION"
    
    logger.info(f"\\n🎯 VERDICT: {winner}")
    logger.info(f"🚀 PRODUCTION: {production_status}")
    
    # Tableau final
    print(f"\\n🎯 CASCADE 52.5% vs BASELINE - EPL 2025-26")
    print(f"\\n{'Modèle':<15} {'Accuracy':<10} {'vs Majority':<12}")
    print(f"{'='*15} {'='*10} {'='*12}")
    print(f"{'Cascade 52.5%':<15} {cascade_accuracy:.3f}      {cascade_accuracy - majority_acc:+.3f}")
    print(f"{'Baseline':<15} {baseline_accuracy:.3f}      {baseline_accuracy - majority_acc:+.3f}")
    print(f"{'Majority':<15} {majority_acc:.3f}      {0:+.3f}")
    print(f"\\n🏆 WINNER: {winner}")
    
    return {
        'cascade_accuracy': cascade_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'majority_accuracy': majority_acc,
        'winner': winner,
        'production_status': production_status
    }

if __name__ == "__main__":
    results = test_cascade_525_exact()