#!/usr/bin/env python3
"""
Test v3.0 Clean Market Model
===========================

Test the simplified market approach:
- Clean market probabilities (H/D/A) instead of derived features
- Class weight balancing instead of SMOTE
- Optional two-stage classification (Draw vs Non-Draw)

Target: Beat v1.5 baseline (53.4%) with better draw detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# Add project root to path
sys.path.append('.')
from utils import setup_logging

def load_v3_dataset():
    """Load v3.0 dataset with clean market features"""
    logger = setup_logging()
    
    file_path = "data/processed/premier_league_market_v3_2025_09_02_105923.csv"
    
    if not os.path.exists(file_path):
        logger.error(f"v3.0 dataset not found: {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    logger.info(f"✅ Loaded v3.0 dataset: {df.shape[0]} matches, {df.shape[1]} features")
    
    return df

def test_single_stage_models():
    """
    Compare baseline vs v3.0 clean market features
    """
    logger = setup_logging()
    logger.info("=== v3.0 Clean Market Model Test ===")
    
    df = load_v3_dataset()
    if df is None:
        return None
        
    # Create target variable
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping)
    
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Feature sets
    traditional_features = [
        'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'corners_diff_normalized', 'market_entropy_norm'
    ]
    
    v3_market_features = [
        'market_home_prob_norm', 'market_draw_prob_norm', 'market_away_prob_norm', 'draw_opportunity'
    ]
    
    # Check availability
    available_trad = [f for f in traditional_features if f in df.columns]
    available_v3 = [f for f in v3_market_features if f in df.columns]
    
    logger.info(f"Traditional features: {len(available_trad)}/6")
    logger.info(f"v3.0 market features: {len(available_v3)}/4")
    
    # Temporal split - last 380 matches as test
    train_idx = df.index[:-380]
    test_idx = df.index[-380:]
    
    X_train_base = df.loc[train_idx]
    X_test_base = df.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    logger.info(f"📊 Split: Train={len(y_train)}, Test={len(y_test)}")
    
    results = {}
    
    # ===== Model 1: v1.5 Baseline =====
    logger.info("\n--- Model 1: v1.5 Baseline ---")
    
    X_train_trad = X_train_base[available_trad]
    X_test_trad = X_test_base[available_trad]
    
    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_baseline.fit(X_train_trad, y_train)
    y_pred_baseline = rf_baseline.predict(X_test_trad)
    
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    logger.info(f"🎯 Accuracy (Baseline): {acc_baseline:.4f}")
    
    print("Classification Report (v1.5 Baseline):")
    report_baseline = classification_report(y_test, y_pred_baseline,
                                          target_names=['Home', 'Draw', 'Away'],
                                          output_dict=True)
    print(classification_report(y_test, y_pred_baseline, target_names=['Home', 'Draw', 'Away']))
    
    results['baseline'] = {
        'accuracy': acc_baseline,
        'draw_f1': report_baseline['Draw']['f1-score'],
        'draw_recall': report_baseline['Draw']['recall'],
        'draw_precision': report_baseline['Draw']['precision']
    }
    
    # ===== Model 2: v3.0 Traditional + Market =====
    logger.info("\n--- Model 2: v3.0 Traditional + Clean Market ---")
    
    combined_features = available_trad + available_v3
    X_train_v3 = X_train_base[combined_features]
    X_test_v3 = X_test_base[combined_features]
    
    rf_v3 = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,  # Slightly deeper for more features
        random_state=42,
        class_weight='balanced'
    )
    
    rf_v3.fit(X_train_v3, y_train)
    y_pred_v3 = rf_v3.predict(X_test_v3)
    
    acc_v3 = accuracy_score(y_test, y_pred_v3)
    logger.info(f"🎯 Accuracy (v3.0): {acc_v3:.4f}")
    
    print("Classification Report (v3.0 Clean Market):")
    report_v3 = classification_report(y_test, y_pred_v3,
                                    target_names=['Home', 'Draw', 'Away'],
                                    output_dict=True)
    print(classification_report(y_test, y_pred_v3, target_names=['Home', 'Draw', 'Away']))
    
    results['v3_market'] = {
        'accuracy': acc_v3,
        'draw_f1': report_v3['Draw']['f1-score'],
        'draw_recall': report_v3['Draw']['recall'],
        'draw_precision': report_v3['Draw']['precision']
    }
    
    # ===== Model 3: Market Only (Test market signal strength) =====
    logger.info("\n--- Model 3: Market Features Only ---")
    
    X_train_market = X_train_base[available_v3]
    X_test_market = X_test_base[available_v3]
    
    rf_market = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_market.fit(X_train_market, y_train)
    y_pred_market = rf_market.predict(X_test_market)
    
    acc_market = accuracy_score(y_test, y_pred_market)
    logger.info(f"🎯 Accuracy (Market Only): {acc_market:.4f}")
    
    print("Classification Report (Market Only):")
    report_market = classification_report(y_test, y_pred_market,
                                        target_names=['Home', 'Draw', 'Away'],
                                        output_dict=True)
    print(classification_report(y_test, y_pred_market, target_names=['Home', 'Draw', 'Away']))
    
    results['market_only'] = {
        'accuracy': acc_market,
        'draw_f1': report_market['Draw']['f1-score'],
        'draw_recall': report_market['Draw']['recall'],
        'draw_precision': report_market['Draw']['precision']
    }
    
    # ===== Feature Importance Analysis =====
    logger.info("\n--- v3.0 Feature Importance ---")
    
    feature_importance = list(zip(combined_features, rf_v3.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance, 1):
        market_marker = " 📈" if "market" in feature else ""
        draw_marker = " 🎯" if "draw" in feature else ""
        logger.info(f"{i:2d}. {feature:<30} {importance:.4f}{market_marker}{draw_marker}")
    
    return results

def test_two_stage_model():
    """
    Test two-stage classification: 
    Stage 1: Draw vs Non-Draw
    Stage 2: Home vs Away (for non-draws)
    """
    logger = setup_logging()
    logger.info("\n=== Two-Stage Model Test ===")
    
    df = load_v3_dataset()
    if df is None:
        return None
        
    # Create targets
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping)
    
    # Stage 1 target: Draw (1) vs Non-Draw (0)
    y_stage1 = (y == 1).astype(int)
    
    # Stage 2 target: Home (0) vs Away (1) for non-draws only
    non_draw_mask = y != 1
    y_stage2 = y[non_draw_mask].map({0: 0, 2: 1})  # H->0, A->1
    
    # Features
    traditional_features = [
        'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'corners_diff_normalized', 'market_entropy_norm'
    ]
    v3_features = [
        'market_home_prob_norm', 'market_draw_prob_norm', 'market_away_prob_norm', 'draw_opportunity'
    ]
    
    all_features = [f for f in traditional_features + v3_features if f in df.columns]
    
    # Temporal split
    train_idx = df.index[:-380]
    test_idx = df.index[-380:]
    
    X_train = df.loc[train_idx, all_features]
    X_test = df.loc[test_idx, all_features] 
    y1_train = y_stage1.loc[train_idx]
    y1_test = y_stage1.loc[test_idx]
    
    logger.info(f"Stage 1 train distribution: {y1_train.value_counts().to_dict()}")
    
    # ===== Stage 1: Draw vs Non-Draw =====
    rf_stage1 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_stage1.fit(X_train, y1_train)
    y1_pred = rf_stage1.predict(X_test)
    
    stage1_acc = accuracy_score(y1_test, y1_pred)
    logger.info(f"🎯 Stage 1 Accuracy (Draw Detection): {stage1_acc:.4f}")
    
    print("Stage 1 Classification Report (Draw vs Non-Draw):")
    print(classification_report(y1_test, y1_pred, target_names=['Non-Draw', 'Draw']))
    
    # ===== Stage 2: Home vs Away (for predicted non-draws) =====
    non_draw_pred_mask = y1_pred == 0  # Predicted non-draws
    non_draw_test_indices = test_idx[non_draw_pred_mask]
    
    if len(non_draw_test_indices) > 0:
        X_stage2_train = df.loc[train_idx[y_stage1.loc[train_idx] == 0], all_features]
        y_stage2_train = y.loc[train_idx[y_stage1.loc[train_idx] == 0]].map({0: 0, 2: 1})
        
        X_stage2_test = df.loc[non_draw_test_indices, all_features]
        y_stage2_true = y.loc[non_draw_test_indices].map({0: 0, 2: 1})
        
        rf_stage2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_stage2.fit(X_stage2_train, y_stage2_train)
        y2_pred = rf_stage2.predict(X_stage2_test)
        
        stage2_acc = accuracy_score(y_stage2_true, y2_pred)
        logger.info(f"🎯 Stage 2 Accuracy (H vs A): {stage2_acc:.4f}")
        
        print("Stage 2 Classification Report (Home vs Away):")
        print(classification_report(y_stage2_true, y2_pred, target_names=['Home', 'Away']))
        
        # ===== Combine Stages for Overall Performance =====
        y_final_pred = np.full(len(y1_test), -1)  # Initialize
        y_final_pred[y1_pred == 1] = 1  # Predicted draws
        
        # Fill non-draws with stage 2 predictions
        non_draw_indices = np.where(y1_pred == 0)[0]
        if len(y2_pred) == len(non_draw_indices):
            y_final_pred[non_draw_indices] = np.where(y2_pred == 0, 0, 2)  # 0->H, 1->A
        
        # Calculate combined accuracy
        y_test_actual = y.loc[test_idx]
        combined_acc = accuracy_score(y_test_actual, y_final_pred)
        logger.info(f"🏆 Two-Stage Combined Accuracy: {combined_acc:.4f}")
        
        return {
            'stage1_accuracy': stage1_acc,
            'stage2_accuracy': stage2_acc,
            'combined_accuracy': combined_acc
        }
    else:
        logger.warning("No non-draws predicted in stage 1")
        return None

def main():
    """Run comprehensive v3.0 testing"""
    logger = setup_logging()
    logger.info("🚀 Starting v3.0 Clean Market Model Tests")
    
    # Test single-stage models
    single_results = test_single_stage_models()
    
    if single_results:
        print("\n" + "="*70)
        print("🏆 SINGLE-STAGE MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<20} {'Accuracy':<12} {'Draw F1':<12} {'Draw Recall':<12} {'Draw Precision'}")
        print("-" * 75)
        
        for model_name, metrics in single_results.items():
            acc = metrics['accuracy']
            f1 = metrics['draw_f1']
            recall = metrics['draw_recall'] 
            precision = metrics['draw_precision']
            
            print(f"{model_name:<20} {acc:<12.1%} {f1:<12.1%} {recall:<12.1%} {precision:<12.1%}")
        
        # Find best model
        best_model = max(single_results.keys(), key=lambda k: single_results[k]['accuracy'])
        best_acc = single_results[best_model]['accuracy']
        
        print(f"\n🎯 Best Single-Stage Model: {best_model} ({best_acc:.1%})")
        
        if best_acc >= 0.55:
            print("🏆 EXCELLENT TARGET ACHIEVED!")
        elif best_acc >= 0.52:
            print("✅ GOOD TARGET ACHIEVED!")
        else:
            print("⚠️  Below good target")
    
    # Test two-stage model
    print("\n" + "="*70)
    print("🔄 TWO-STAGE MODEL TEST")
    print("="*70)
    
    two_stage_results = test_two_stage_model()
    
    if two_stage_results:
        print(f"Two-Stage Combined: {two_stage_results['combined_accuracy']:.1%}")
    
    print("\n✅ v3.0 Testing Complete!")

if __name__ == "__main__":
    main()