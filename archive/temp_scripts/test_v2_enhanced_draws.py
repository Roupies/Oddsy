#!/usr/bin/env python3
"""
Test v2.0 Enhanced Draw Model
============================

Compare v1.6 SMOTE baseline vs v2.0 with enhanced draw odds features.
Focus on draw prediction improvement while maintaining overall accuracy.

Expected improvement: 50.79% → 55%+ accuracy with better draw detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import sys
import os

# Add project root to path  
sys.path.append('.')
from utils import setup_logging

def load_enhanced_dataset():
    """Load the v2.0 dataset with enhanced draw features"""
    logger = setup_logging()
    
    # Load the enhanced dataset created by enhanced_draw_odds_v2.py
    file_path = "data/processed/premier_league_enhanced_draws_v2_2025_09_02_105034.csv"
    
    if not os.path.exists(file_path):
        logger.error(f"Enhanced dataset not found: {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    logger.info(f"✅ Loaded enhanced dataset: {df.shape[0]} matches, {df.shape[1]} features")
    
    return df

def compare_feature_sets():
    """
    Compare 3 model configurations:
    1. v1.5 Baseline (traditional features only)
    2. v1.6 SMOTE (traditional + SMOTE)  
    3. v2.0 Enhanced (traditional + SMOTE + draw features)
    """
    
    logger = setup_logging()
    logger.info("=== v2.0 Enhanced Draw Model Comparison ===")
    
    df = load_enhanced_dataset()
    if df is None:
        return
        
    # Define feature sets
    traditional_features = [
        'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'corners_diff_normalized', 'market_entropy_norm'
    ]
    
    enhanced_draw_features = [
        'draw_value_indicator', 'bookmaker_draw_disagreement', 
        'market_draw_bias', 'draw_odds_instability'
    ]
    
    # Check feature availability
    available_traditional = [f for f in traditional_features if f in df.columns]
    available_enhanced = [f for f in enhanced_draw_features if f in df.columns]
    
    logger.info(f"Traditional features available: {len(available_traditional)}/6")
    logger.info(f"Enhanced draw features available: {len(available_enhanced)}/4")
    
    if len(available_traditional) < 5:
        logger.error("Insufficient traditional features for comparison")
        return
        
    # Prepare data - create target from FullTimeResult
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping)
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data (temporal split for realistic evaluation)
    # Use last 380 matches as test set (roughly 1 season)
    train_idx = df.index[:-380]
    test_idx = df.index[-380:]
    
    X_train_base = df.loc[train_idx]
    X_test_base = df.loc[test_idx] 
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    logger.info(f"📊 Data Split: Train={len(y_train)}, Test={len(y_test)}")
    
    results = {}
    
    # ===== Model 1: v1.5 Baseline (No SMOTE) =====
    logger.info("\n--- Model 1: v1.5 Baseline (Traditional Features Only) ---")
    
    X_train_trad = X_train_base[available_traditional]
    X_test_trad = X_test_base[available_traditional]
    
    rf_baseline = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        class_weight='balanced'  # Handle imbalance without SMOTE
    )
    
    rf_baseline.fit(X_train_trad, y_train)
    y_pred_baseline = rf_baseline.predict(X_test_trad)
    
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    logger.info(f"🎯 Accuracy (v1.5 Baseline): {acc_baseline:.4f}")
    
    print("Classification Report (v1.5 Baseline):")
    report_baseline = classification_report(y_test, y_pred_baseline, 
                                          target_names=['Home', 'Draw', 'Away'],
                                          output_dict=True)
    print(classification_report(y_test, y_pred_baseline, target_names=['Home', 'Draw', 'Away']))
    
    results['v1.5_baseline'] = {
        'accuracy': acc_baseline,
        'draw_f1': report_baseline['Draw']['f1-score'],
        'draw_recall': report_baseline['Draw']['recall']
    }
    
    # ===== Model 2: v1.6 SMOTE =====
    logger.info("\n--- Model 2: v1.6 SMOTE (Traditional + SMOTE) ---")
    
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_trad, y_train)
    
    logger.info(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
    
    rf_smote = RandomForestClassifier(
        n_estimators=100,
        max_depth=10, 
        random_state=42
    )
    
    rf_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = rf_smote.predict(X_test_trad)
    
    acc_smote = accuracy_score(y_test, y_pred_smote)
    logger.info(f"🎯 Accuracy (v1.6 SMOTE): {acc_smote:.4f}")
    
    print("Classification Report (v1.6 SMOTE):")
    report_smote = classification_report(y_test, y_pred_smote,
                                       target_names=['Home', 'Draw', 'Away'],
                                       output_dict=True)
    print(classification_report(y_test, y_pred_smote, target_names=['Home', 'Draw', 'Away']))
    
    results['v1.6_smote'] = {
        'accuracy': acc_smote,
        'draw_f1': report_smote['Draw']['f1-score'],
        'draw_recall': report_smote['Draw']['recall']
    }
    
    # ===== Model 3: v2.0 Enhanced =====
    logger.info("\n--- Model 3: v2.0 Enhanced (Traditional + SMOTE + Draw Features) ---")
    
    all_features = available_traditional + available_enhanced
    X_train_enhanced = X_train_base[all_features]
    X_test_enhanced = X_test_base[all_features]
    
    # Apply SMOTE to enhanced feature set
    X_train_enh_smote, y_train_enh_smote = smote.fit_resample(X_train_enhanced, y_train)
    
    rf_enhanced = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,  # Slightly deeper for more features
        random_state=42
    )
    
    rf_enhanced.fit(X_train_enh_smote, y_train_enh_smote)
    y_pred_enhanced = rf_enhanced.predict(X_test_enhanced)
    
    acc_enhanced = accuracy_score(y_test, y_pred_enhanced)
    logger.info(f"🎯 Accuracy (v2.0 Enhanced): {acc_enhanced:.4f}")
    
    print("Classification Report (v2.0 Enhanced):")
    report_enhanced = classification_report(y_test, y_pred_enhanced,
                                          target_names=['Home', 'Draw', 'Away'],
                                          output_dict=True)
    print(classification_report(y_test, y_pred_enhanced, target_names=['Home', 'Draw', 'Away']))
    
    results['v2.0_enhanced'] = {
        'accuracy': acc_enhanced,
        'draw_f1': report_enhanced['Draw']['f1-score'],
        'draw_recall': report_enhanced['Draw']['recall']
    }
    
    # ===== Feature Importance Analysis =====
    logger.info("\n--- v2.0 Enhanced Feature Importance ---")
    
    feature_importance = list(zip(all_features, rf_enhanced.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance, 1):
        draw_marker = " 🎯" if feature in enhanced_draw_features else ""
        market_marker = " 📈" if "market" in feature else ""
        logger.info(f"{i:2d}. {feature:<25} {importance:.4f}{draw_marker}{market_marker}")
    
    # ===== Final Comparison =====
    logger.info("\n" + "="*70)
    logger.info("🏆 FINAL COMPARISON - v2.0 Enhanced Draw Model")
    logger.info("="*70)
    
    print(f"{'Model':<20} {'Accuracy':<12} {'Draw F1':<12} {'Draw Recall':<12} {'Improvement'}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        acc = metrics['accuracy']
        draw_f1 = metrics['draw_f1'] 
        draw_recall = metrics['draw_recall']
        
        if model_name == 'v1.5_baseline':
            improvement = "Baseline"
        else:
            acc_diff = acc - results['v1.5_baseline']['accuracy']
            improvement = f"+{acc_diff:+.1%}"
            
        print(f"{model_name:<20} {acc:<12.1%} {draw_f1:<12.1%} {draw_recall:<12.1%} {improvement}")
    
    # Highlight best performing model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_acc = results[best_model]['accuracy']
    
    logger.info(f"\n🎯 Best Model: {best_model} with {best_acc:.1%} accuracy")
    
    if best_acc >= 0.55:
        logger.info("🏆 EXCELLENT TARGET ACHIEVED! (55%+ accuracy)")
    elif best_acc >= 0.52:
        logger.info("✅ GOOD TARGET ACHIEVED! (52%+ accuracy)")
    else:
        logger.info("⚠️  Still below good target (52%)")
    
    # Check draw improvement specifically
    baseline_draw_f1 = results['v1.5_baseline']['draw_f1']
    enhanced_draw_f1 = results['v2.0_enhanced']['draw_f1']
    draw_improvement = enhanced_draw_f1 - baseline_draw_f1
    
    logger.info(f"📊 Draw F1-Score Improvement: {baseline_draw_f1:.1%} → {enhanced_draw_f1:.1%} ({draw_improvement:+.1%})")
    
    return results

if __name__ == "__main__":
    results = compare_feature_sets()
    
    if results:
        print("\n🎉 v2.0 Enhanced Draw Model Testing Complete!")
        print("Ready to proceed with best configuration for production.")
    else:
        print("❌ Testing failed - check data and features.")