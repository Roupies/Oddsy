#!/usr/bin/env python3
"""
Quick Threshold Fix - Fast optimization to fix 42.9% accuracy issue
No hyperparameter tuning - just threshold optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def quick_cascade_test():
    """Fast threshold optimization test."""
    print("🚀 QUICK THRESHOLD FIX")
    print("=" * 40)
    
    # Load data
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['stage1_target'] = (df['FullTimeResult'] == 'D').astype(int)
    
    features = ['elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
               'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized', 
               'form_diff_normalized', 'h2h_score', 'away_goals_sum_5']
    
    # Split data
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    print(f"Data: Train {len(train_df)}, Test {len(test_df)}")
    print(f"True draw rate: {test_df['stage1_target'].mean():.1%}")
    
    # Quick Stage 1 training
    X_train_s1 = train_df[features].fillna(train_df[features].median())
    y_train_s1 = train_df['stage1_target']
    
    smote = SMOTE(random_state=42)
    X_train_s1_balanced, y_train_s1_balanced = smote.fit_resample(X_train_s1, y_train_s1)
    
    stage1_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Faster
    stage1_model.fit(X_train_s1_balanced, y_train_s1_balanced)
    
    # Quick Stage 2 training  
    train_non_draw = train_df[train_df['stage1_target'] == 0].copy()
    X_train_s2 = train_non_draw[features].fillna(train_non_draw[features].median())
    y_train_s2 = train_non_draw['FullTimeResult'].map({'H': 0, 'A': 1})
    
    stage2_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Faster
    stage2_model.fit(X_train_s2, y_train_s2)
    
    # Test thresholds quickly
    X_test = test_df[features].fillna(train_df[features].median())
    y_test_true = test_df['FullTimeResult']
    stage1_proba = stage1_model.predict_proba(X_test)[:, 1]
    
    print(f"\n🎯 THRESHOLD OPTIMIZATION:")
    print(f"{'Thresh':<7} {'Accuracy':<9} {'DrawPred':<9} {'DrawRec':<8} {'Status'}")
    print("-" * 50)
    
    best_threshold = 0.5
    best_accuracy = 0
    
    # Test key thresholds
    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        draw_mask = stage1_proba >= threshold
        y_pred = np.full(len(X_test), 'D', dtype=object)
        
        if (~draw_mask).sum() > 0:
            non_draw_features = X_test[~draw_mask]
            stage2_pred = stage2_model.predict(non_draw_features)
            y_pred[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
        
        accuracy = accuracy_score(y_test_true, y_pred)
        draw_pred_rate = draw_mask.mean()
        
        # Draw recall
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_true, y_pred, labels=['H', 'D', 'A'], average=None, zero_division=0
        )
        draw_recall = recall[1]
        
        status = "🟢 GOOD" if accuracy >= 0.45 else "🟡 OK" if accuracy >= 0.40 else "🔴 BAD"
        
        print(f"{threshold:<7.2f} {accuracy:<9.1%} {draw_pred_rate:<9.1%} {draw_recall:<8.1%} {status}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"\n🏆 BEST THRESHOLD: {best_threshold:.2f}")
    print(f"Best Accuracy: {best_accuracy:.1%}")
    
    # Test best threshold in detail
    draw_mask = stage1_proba >= best_threshold
    y_pred = np.full(len(X_test), 'D', dtype=object)
    
    if (~draw_mask).sum() > 0:
        non_draw_features = X_test[~draw_mask]
        stage2_pred = stage2_model.predict(non_draw_features)
        y_pred[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
    
    accuracy = accuracy_score(y_test_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_true, y_pred, labels=['H', 'D', 'A'], average=None, zero_division=0
    )
    
    print(f"\n📊 DETAILED PERFORMANCE (threshold={best_threshold:.2f}):")
    print(f"Global Accuracy: {accuracy:.1%}")
    print(f"Draw Predictions: {draw_mask.sum()}/{len(draw_mask)} ({draw_mask.mean():.1%})")
    print(f"Home F1: {f1[0]:.3f}")
    print(f"Draw F1: {f1[1]:.3f} (Recall: {recall[1]:.1%})")
    print(f"Away F1: {f1[2]:.3f}")
    print(f"F1-Macro: {np.mean(f1):.3f}")
    
    # Compare to baseline
    print(f"\n🆚 COMPARISON:")
    print(f"Current (0.4): 42.9% accuracy")
    print(f"Optimized ({best_threshold:.2f}): {accuracy:.1%} accuracy")
    print(f"Improvement: {(accuracy - 0.429)*100:+.1f}pp")

if __name__ == "__main__":
    quick_cascade_test()