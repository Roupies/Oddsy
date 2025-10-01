#!/usr/bin/env python3
"""
Comprehensive Model Testing Script for Oddsy Football Prediction Project
Adapted from base script to match Oddsy's project structure and data format

Tests all production models on two scenarios:
1. Phase 1: Original split (1,900 train / 380 test) 
2. Phase 2: Extended training (2,280 train / 50 test EPL 25-26)

Usage:
    python test_all_production_models_comprehensive.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Import cascade model classes
sys.path.append('.')
try:
    from cascade_champion_v20_production import CascadeChampionV20
    from final_cascade_champion_v21 import CascadeChampionV21
    print("✓ Imported cascade model classes")
except ImportError as e:
    print(f"⚠️ Warning: Could not import cascade classes: {e}")
    CascadeChampionV20 = None
    CascadeChampionV21 = None


def find_oddsy_models(models_dir="models/production"):
    """Find all production models in Oddsy project structure"""
    models = {}
    p = Path(models_dir)
    if not p.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    for f in p.iterdir():
        if f.suffix == '.joblib' and 'champion' in f.name.lower():
            models[f.stem] = str(f)
    
    print(f"Found {len(models)} production models:")
    for name, path in models.items():
        print(f"  - {name}: {path}")
    
    return models


def load_model_safe(path):
    """Safely load a model with error handling and CASCADE FRESH CREATION for anti-leakage"""
    model_name = Path(path).stem
    
    # FORCE FRESH CASCADE MODELS to prevent data leakage from pre-trained models
    if 'cascade_champion_v20' in model_name.lower() and CascadeChampionV20:
        print(f"🔄 Creating FRESH Cascade Champion v2.0 (anti-leakage)...")
        return CascadeChampionV20()
    elif 'cascade_champion_v21' in model_name.lower() and CascadeChampionV21:
        print(f"🔄 Creating FRESH Cascade Champion v2.1 (anti-leakage)...")
        return CascadeChampionV21()
    
    # For non-cascade models, try normal loading
    try:
        model = joblib.load(path)
        print(f"✓ Loaded: {Path(path).name}")
        return model
    except Exception as e:
        print(f"✗ Failed to load {Path(path).name}: {e}")
        return None


def load_oddsy_dataset(dataset_path="data/processed/v_auto_update_20250922_093416.csv"):
    """Load Oddsy dataset with proper column handling"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use FullTimeResult as target (H/D/A format)
    target_col = 'FullTimeResult'
    
    # Define feature columns in EXACT order from baseline_champion_v23 metadata
    feature_cols = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Filter to only include rows with all required features and target
    mask = df[feature_cols + [target_col]].notna().all(axis=1)
    df_clean = df[mask].copy()
    
    print(f"Dataset loaded: {len(df)} total rows, {len(df_clean)} with complete data")
    print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    print(f"Target distribution: {df_clean[target_col].value_counts().to_dict()}")
    
    return df_clean, feature_cols, target_col


def evaluate_model_oddsy(model, X_test, y_test):
    """Evaluate model with Oddsy-specific metrics"""
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None
    
    labels = ['H', 'D', 'A']  # Home, Draw, Away
    
    # Handle label conversion - models may predict numbers or strings
    if hasattr(y_pred, 'dtype') and np.issubdtype(y_pred.dtype, np.number):
        # Model predicts numbers, convert to labels
        label_map = {0: 'H', 1: 'D', 2: 'A'}
        y_pred_labels = [label_map.get(pred, 'H') for pred in y_pred]
    else:
        # Model already predicts labels
        y_pred_labels = y_pred
    
    # Ensure y_test is string labels
    if hasattr(y_test, 'dtype') and np.issubdtype(y_test.dtype, np.number):
        label_map = {0: 'H', 1: 'D', 2: 'A'}
        y_test_labels = [label_map.get(true, 'H') for true in y_test]
    else:
        y_test_labels = y_test
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_test_labels, y_pred_labels, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=labels)
    
    # Classification report
    report = classification_report(y_test_labels, y_pred_labels, labels=labels, zero_division=0, output_dict=True)
    
    return {
        'accuracy': float(accuracy),
        'precision_per_class': dict(zip(labels, precision)),
        'recall_per_class': dict(zip(labels, recall)),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'labels': labels
    }


def plot_confusion_matrix_oddsy(cm, labels, save_path, title):
    """Plot confusion matrix with Oddsy styling"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title}\nAccuracy: {np.trace(cm)/np.sum(cm):.3f}')
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    
    # Add text annotations
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


def run_phase1_oddsy(models, df, feature_cols, target_col, out_dir):
    """Phase 1: Test on original 1,900/380 split"""
    print("\n=== PHASE 1: Original Split (1,900 train / 380 test) ===")
    
    # Sort by date and use first 2,280 rows (1,900 + 380)
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    if len(df_sorted) < 2280:
        print(f"Warning: Dataset has only {len(df_sorted)} rows, using all available")
        train_size = min(1900, len(df_sorted) - 50)
        test_size = min(380, len(df_sorted) - train_size)
    else:
        train_size = 1900
        test_size = 380
    
    train_df = df_sorted.iloc[:train_size]
    test_df = df_sorted.iloc[train_size:train_size + test_size]
    
    print(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    return evaluate_models_on_split(models, train_df, test_df, feature_cols, target_col, out_dir, "phase1")


def run_phase2_oddsy(models, df, feature_cols, target_col, out_dir):
    """Phase 2: Test with extended training (2,280 train / ~50 test)"""
    print("\n=== PHASE 2: Extended Training (2,280 train / 50 test) ===")
    
    # Sort by date
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    if len(df_sorted) < 2330:
        print(f"Warning: Dataset has only {len(df_sorted)} rows")
        train_size = min(2280, len(df_sorted) - 50)
        test_size = len(df_sorted) - train_size
    else:
        train_size = 2280
        test_size = 50
    
    train_df = df_sorted.iloc[:train_size]
    test_df = df_sorted.iloc[train_size:train_size + test_size]
    
    print(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    return evaluate_models_on_split(models, train_df, test_df, feature_cols, target_col, out_dir, "phase2")


def evaluate_models_on_split(models, train_df, test_df, feature_cols, target_col, out_dir, phase_name):
    """Evaluate all models on a train/test split"""
    results = []
    
    # Prepare features
    X_train = train_df[feature_cols].fillna(0.5)  # Fill NaN with neutral value
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0.5)
    y_test = test_df[target_col]
    
    # Convert targets to numeric for cascade models
    target_map = {'H': 0, 'D': 1, 'A': 2}
    y_train_numeric = y_train.map(target_map)
    y_test_numeric = y_test.map(target_map)
    
    print(f"\nTrain set: {len(X_train)} matches")
    print(f"Test set: {len(X_test)} matches")
    print(f"Train distribution: {y_train.value_counts().to_dict()}")
    print(f"Test distribution: {y_test.value_counts().to_dict()}")
    
    # Evaluate each model
    for model_name, model_path in models.items():
        print(f"\n--- Evaluating {model_name} ---")
        
        model = load_model_safe(model_path)
        if model is None:
            continue
        
        # Check if model needs training (cascade models) - FIX DATA LEAKAGE
        if hasattr(model, 'is_fitted') and not model.is_fitted:
            print(f"🏗️ Training {model_name}...")
            try:
                # CRITICAL FIX: Pass only features, not full DataFrame
                # This prevents data leakage from target/date columns
                model.fit(X_train, y_train_numeric)
            except Exception as e:
                print(f"✗ Training failed: {e}")
                continue
        
        # Get predictions and metrics
        metrics = evaluate_model_oddsy(model, X_test, y_test)
        if metrics is None:
            continue
        
        # Plot confusion matrix
        cm_path = os.path.join(out_dir, f"cm_{phase_name}_{model_name}.png")
        plot_confusion_matrix_oddsy(
            np.array(metrics['confusion_matrix']), 
            metrics['labels'], 
            cm_path, 
            f"{model_name} - {phase_name.upper()}"
        )
        
        # Store results
        result = {
            'phase': phase_name,
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': metrics['accuracy'],
            'precision_H': metrics['precision_per_class']['H'],
            'precision_D': metrics['precision_per_class']['D'],
            'precision_A': metrics['precision_per_class']['A'],
            'recall_H': metrics['recall_per_class']['H'],
            'recall_D': metrics['recall_per_class']['D'],
            'recall_A': metrics['recall_per_class']['A'],
            'confusion_matrix': metrics['confusion_matrix'],
            'confusion_matrix_file': cm_path,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        results.append(result)
        
        # Print summary
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision - H: {metrics['precision_per_class']['H']:.3f}, D: {metrics['precision_per_class']['D']:.3f}, A: {metrics['precision_per_class']['A']:.3f}")
        print(f"Recall - H: {metrics['recall_per_class']['H']:.3f}, D: {metrics['recall_per_class']['D']:.3f}, A: {metrics['recall_per_class']['A']:.3f}")
    
    return results


def generate_summary_report(all_results, out_dir):
    """Generate comprehensive summary report"""
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Phase': result['phase'],
            'Model': result['model_name'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision_H': f"{result['precision_H']:.3f}",
            'Precision_D': f"{result['precision_D']:.3f}",
            'Precision_A': f"{result['precision_A']:.3f}",
            'Recall_H': f"{result['recall_H']:.3f}",
            'Recall_D': f"{result['recall_D']:.3f}",
            'Recall_A': f"{result['recall_A']:.3f}",
            'Train_Size': result['train_size'],
            'Test_Size': result['test_size']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary CSV
    summary_path = os.path.join(out_dir, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed results JSON
    results_path = os.path.join(out_dir, 'detailed_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Find best models
    phase1_results = [r for r in all_results if r['phase'] == 'phase1']
    phase2_results = [r for r in all_results if r['phase'] == 'phase2']
    
    if phase1_results:
        best_phase1 = max(phase1_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 BEST PHASE 1 MODEL: {best_phase1['model_name']} - {best_phase1['accuracy']:.4f}")
    
    if phase2_results:
        best_phase2 = max(phase2_results, key=lambda x: x['accuracy'])
        print(f"🏆 BEST PHASE 2 MODEL: {best_phase2['model_name']} - {best_phase2['accuracy']:.4f}")
    
    print(f"\n📊 Results saved to:")
    print(f"  - Summary: {summary_path}")
    print(f"  - Details: {results_path}")
    print(f"  - Confusion matrices: {out_dir}/cm_*.png")
    
    return summary_df


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Comprehensive Oddsy Model Testing')
    parser.add_argument('--models_dir', default='models/production', help='Directory containing model files')
    parser.add_argument('--dataset', default='data/processed/v_auto_update_20250922_093416.csv', help='Path to dataset')
    parser.add_argument('--out_dir', default='outputs/model_testing', help='Output directory')
    parser.add_argument('--phase', choices=['phase1', 'phase2', 'all'], default='all', help='Which phase to run')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("🚀 ODDSY COMPREHENSIVE MODEL TESTING")
    print("="*50)
    
    try:
        # Load models and dataset
        models = find_oddsy_models(args.models_dir)
        if not models:
            print(f"❌ No models found in {args.models_dir}")
            return
        
        df, feature_cols, target_col = load_oddsy_dataset(args.dataset)
        
        # Run testing phases
        all_results = []
        
        if args.phase in ['phase1', 'all']:
            phase1_results = run_phase1_oddsy(models, df, feature_cols, target_col, args.out_dir)
            all_results.extend(phase1_results)
        
        if args.phase in ['phase2', 'all']:
            phase2_results = run_phase2_oddsy(models, df, feature_cols, target_col, args.out_dir)
            all_results.extend(phase2_results)
        
        # Generate summary
        if all_results:
            generate_summary_report(all_results, args.out_dir)
        else:
            print("❌ No results generated")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()