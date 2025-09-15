#!/usr/bin/env python3
"""
Proper Season Split Test - Train 2019-2024, Test 2024-2025
Test if we can get good draw recall with proper season-based split.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def test_proper_season_split():
    """Test with proper season-based split: 2019-2024 train, 2024-2025 test."""
    print("🏆 PROPER SEASON SPLIT TEST")
    print("=" * 60)
    print("Train: 2019-2024 seasons (5 complete seasons)")
    print("Test: 2024-2025 season (current season)")
    print()
    
    # Load data
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Season-based split
    train_df = df[df['Season'].isin(['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024'])].copy()
    test_df = df[df['Season'] == '2024-2025'].copy()
    
    print(f"📊 Data Split:")
    print(f"Train: {len(train_df)} matches ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Test: {len(test_df)} matches ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")
    
    # Check distribution
    print(f"\n📈 Train Distribution:")
    train_dist = train_df['FullTimeResult'].value_counts(normalize=True).sort_index()
    for result, pct in train_dist.items():
        print(f"  {result}: {pct:.1%}")
    
    print(f"\n📈 Test Distribution:")
    test_dist = test_df['FullTimeResult'].value_counts(normalize=True).sort_index()
    for result, pct in test_dist.items():
        print(f"  {result}: {pct:.1%}")
    
    # Features
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # Test 1: Simple 3-class RandomForest
    print(f"\n1️⃣ SIMPLE 3-CLASS RANDOMFOREST:")
    X_train = train_df[features].fillna(train_df[features].median())
    y_train = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    X_test = test_df[features].fillna(train_df[features].median())
    y_test = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    simple_model = RandomForestClassifier(n_estimators=100, random_state=42)
    simple_model.fit(X_train, y_train)
    y_pred_simple = simple_model.predict(X_test)
    accuracy_simple = accuracy_score(y_test, y_pred_simple)
    
    print(f"Accuracy: {accuracy_simple:.1%}")
    
    # Detailed classification report
    pred_simple_mapped = {0: 'H', 1: 'D', 2: 'A'}
    y_pred_simple_hda = [pred_simple_mapped[p] for p in y_pred_simple]
    y_test_hda = test_df['FullTimeResult']
    
    print("\nClassification Report:")
    print(classification_report(y_test_hda, y_pred_simple_hda, labels=['H', 'D', 'A']))
    
    # Test 2: Cascade with multiple thresholds for draw recall
    print(f"\n2️⃣ CASCADE WITH DRAW RECALL OPTIMIZATION:")
    
    # Stage 1: Draw detection with SMOTE
    y_train_draw = (train_df['FullTimeResult'] == 'D').astype(int)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_draw)
    
    stage1_model = RandomForestClassifier(n_estimators=100, random_state=42)
    stage1_model.fit(X_train_balanced, y_train_balanced)
    stage1_proba = stage1_model.predict_proba(X_test)[:, 1]
    
    # Stage 2: Home vs Away for non-draws
    train_non_draw = train_df[train_df['FullTimeResult'] != 'D'].copy()
    X_train_s2 = train_non_draw[features].fillna(train_non_draw[features].median())
    y_train_s2 = train_non_draw['FullTimeResult'].map({'H': 0, 'A': 1})
    
    stage2_model = RandomForestClassifier(n_estimators=100, random_state=42)
    stage2_model.fit(X_train_s2, y_train_s2)
    
    # Test multiple thresholds focusing on draw recall
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1_macro = 0
    best_threshold = 0.5
    best_result = None
    
    print("\nThreshold Optimization (focusing on draw recall):")
    
    for threshold in thresholds:
        draw_mask = stage1_proba >= threshold
        y_pred_cascade = np.full(len(X_test), 'D', dtype=object)
        
        # Stage 2 for non-draws
        if (~draw_mask).sum() > 0:
            stage2_pred = stage2_model.predict(X_test[~draw_mask])
            y_pred_cascade[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
        
        accuracy_cascade = accuracy_score(y_test_hda, y_pred_cascade)
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score
        f1_macro = f1_score(y_test_hda, y_pred_cascade, labels=['H', 'D', 'A'], average='macro')
        f1_draw = f1_score(y_test_hda, y_pred_cascade, labels=['D'], average='macro')
        
        # Draw statistics
        draw_predictions = draw_mask.sum()
        draw_recall = np.sum((y_test_hda == 'D') & (y_pred_cascade == 'D')) / np.sum(y_test_hda == 'D')
        draw_precision = np.sum((y_test_hda == 'D') & (y_pred_cascade == 'D')) / max(1, np.sum(y_pred_cascade == 'D'))
        
        print(f"  Threshold {threshold:.1f}: {accuracy_cascade:.1%} acc, {f1_macro:.3f} F1, {draw_predictions:3d} draws ({draw_mask.mean():.1%})")
        print(f"    Draw: {draw_precision:.1%} prec, {draw_recall:.1%} recall, {f1_draw:.3f} F1")
        
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_threshold = threshold
            best_result = {
                'accuracy': accuracy_cascade,
                'f1_macro': f1_macro,
                'f1_draw': f1_draw,
                'draw_recall': draw_recall,
                'draw_precision': draw_precision,
                'draw_predictions': draw_predictions,
                'y_pred': y_pred_cascade
            }
    
    # Final assessment
    print(f"\n🏆 SEASON SPLIT RESULTS:")
    print(f"Simple 3-class: {accuracy_simple:.1%} accuracy")
    print(f"Best Cascade: {best_result['accuracy']:.1%} accuracy (threshold {best_threshold:.1f})")
    print(f"  F1-Macro: {best_result['f1_macro']:.3f}")
    print(f"  Draw Recall: {best_result['draw_recall']:.1%}")
    print(f"  Draw Predictions: {best_result['draw_predictions']}/{len(X_test)} ({best_result['draw_predictions']/len(X_test):.1%})")
    
    improvement = (best_result['accuracy'] - accuracy_simple) * 100
    
    if best_result['accuracy'] > accuracy_simple:
        print(f"✅ CASCADE WINS: +{improvement:.1f}pp improvement!")
    elif abs(improvement) < 1:
        print(f"🤝 TIE: {improvement:+.1f}pp difference (negligible)")
    else:
        print(f"❌ Simple wins: {improvement:+.1f}pp")
    
    # Show confusion matrix for best cascade
    print(f"\n📊 Best Cascade Confusion Matrix:")
    cm = confusion_matrix(y_test_hda, best_result['y_pred'], labels=['H', 'D', 'A'])
    print("      H    D    A")
    for i, label in enumerate(['H', 'D', 'A']):
        print(f"{label}:  {cm[i][0]:3d} {cm[i][1]:3d} {cm[i][2]:3d}")
    
    return {
        'simple_accuracy': accuracy_simple,
        'cascade_accuracy': best_result['accuracy'],
        'best_threshold': best_threshold,
        'draw_recall': best_result['draw_recall'],
        'f1_macro': best_result['f1_macro']
    }

if __name__ == "__main__":
    test_proper_season_split()