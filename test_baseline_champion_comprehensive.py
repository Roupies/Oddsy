#!/usr/bin/env python3
"""
Comprehensive Testing of Baseline Champion v2.3 on Both Scenarios
Focus on the working production model first
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def load_baseline_model():
    """Load Baseline Champion v2.3"""
    model_path = 'models/production/baseline_champion_v23.joblib'
    try:
        model = joblib.load(model_path)
        print(f"✓ Loaded Baseline Champion v2.3")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

def load_data():
    """Load and prepare dataset"""
    data_path = "data/processed/v_auto_update_20250922_093416.csv"
    df = pd.read_csv(data_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Exact feature order from metadata
    feature_cols = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Filter complete data
    mask = df[feature_cols + ['FullTimeResult']].notna().all(axis=1)
    df_clean = df[mask].copy()
    
    print(f"Dataset: {len(df)} total → {len(df_clean)} complete")
    print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    
    return df_clean, feature_cols

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    # Convert predictions from numbers to labels (model predicts 0,1,2)
    label_map = {0: 'H', 1: 'D', 2: 'A'}
    y_pred_labels = [label_map[pred] for pred in y_pred]
    
    accuracy = accuracy_score(y_test, y_pred_labels)
    labels = ['H', 'D', 'A']
    
    precision = precision_score(y_test, y_pred_labels, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred_labels, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_labels, labels=labels)
    
    return {
        'accuracy': accuracy,
        'precision': dict(zip(labels, precision)),
        'recall': dict(zip(labels, recall)),
        'confusion_matrix': cm,
        'labels': labels
    }

def plot_confusion_matrix(cm, labels, save_path, title):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title}\nAccuracy: {np.trace(cm)/np.sum(cm):.3f}')
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight='bold')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_phase1(model, df, feature_cols):
    """Phase 1: Original 1,900/380 split"""
    print("\n=== PHASE 1: Original Split (1,900 train / 380 test) ===")
    
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    train_df = df_sorted.iloc[:1900]
    test_df = df_sorted.iloc[1900:2280]
    
    print(f"Train: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test: {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    X_test = test_df[feature_cols].fillna(0.5)
    y_test = test_df['FullTimeResult']
    
    print(f"Test distribution: {y_test.value_counts().to_dict()}")
    
    metrics = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    os.makedirs('outputs', exist_ok=True)
    plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'], 
                         'outputs/cm_phase1_baseline.png', 'Baseline Champion v2.3 - Phase 1')
    
    return metrics

def test_phase2(model, df, feature_cols):
    """Phase 2: Extended training 2,280/50 split"""
    print("\n=== PHASE 2: Extended Training (2,280 train / 50 test) ===")
    
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    # Use more training data
    train_df = df_sorted.iloc[:2280] if len(df_sorted) > 2280 else df_sorted.iloc[:-50]
    test_df = df_sorted.iloc[2280:2330] if len(df_sorted) > 2330 else df_sorted.iloc[-50:]
    
    print(f"Train: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test: {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    X_test = test_df[feature_cols].fillna(0.5)
    y_test = test_df['FullTimeResult']
    
    print(f"Test distribution: {y_test.value_counts().to_dict()}")
    
    metrics = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'], 
                         'outputs/cm_phase2_baseline.png', 'Baseline Champion v2.3 - Phase 2')
    
    return metrics

def print_results(phase_name, metrics):
    """Print formatted results"""
    print(f"\n📊 {phase_name} RESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Precision by class:")
    for label in ['H', 'D', 'A']:
        print(f"  {label}: {metrics['precision'][label]:.3f}")
    print("Recall by class:")
    for label in ['H', 'D', 'A']:
        print(f"  {label}: {metrics['recall'][label]:.3f}")

def main():
    """Main execution"""
    print("🏆 BASELINE CHAMPION v2.3 COMPREHENSIVE TESTING")
    print("=" * 60)
    
    # Load model and data
    model = load_baseline_model()
    if not model:
        return
    
    df, feature_cols = load_data()
    
    # Test both phases
    phase1_metrics = test_phase1(model, df, feature_cols)
    print_results("PHASE 1", phase1_metrics)
    
    phase2_metrics = test_phase2(model, df, feature_cols)
    print_results("PHASE 2", phase2_metrics)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("PHASE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Phase 1 (1,900/380): {phase1_metrics['accuracy']:.4f}")
    print(f"Phase 2 (2,280/50):  {phase2_metrics['accuracy']:.4f}")
    
    improvement = phase2_metrics['accuracy'] - phase1_metrics['accuracy']
    print(f"Improvement: {improvement:+.4f}")
    
    # Save results
    results = {
        'baseline_champion_v23': {
            'phase1': phase1_metrics,
            'phase2': phase2_metrics,
            'improvement': improvement
        }
    }
    
    with open('outputs/baseline_comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to outputs/")

if __name__ == '__main__':
    main()