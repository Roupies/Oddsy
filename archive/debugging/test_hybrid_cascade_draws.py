#!/usr/bin/env python3
"""
Hybrid Cascade Model for Draw Prediction
========================================

Instead of ClusterCentroids (which hurt overall accuracy), test a cascaded approach:

1. Model 1: Binary Draw detector (Draw vs Non-Draw) - can use oversampling here
2. Model 2: Home vs Away classifier (only when Model 1 says "Non-Draw")

This should maintain overall accuracy while improving draw detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Add project root to path
sys.path.append('.')
from utils import setup_logging

def prepare_data_for_cascade():
    """Prepare data for cascade approach"""
    logger = setup_logging()
    logger.info("=== PREPARING DATA FOR CASCADE MODEL ===")
    
    # Load corrected dataset
    corrected_file = "data/processed/v13_xg_corrected_features_latest.csv"
    df = pd.read_csv(corrected_file, parse_dates=['Date'])
    
    # Same 10 features as production
    model_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Same temporal split
    train_cutoff = '2024-05-19'
    test_start = '2024-08-16'
    
    valid_data = df.dropna(subset=model_features)
    train_data = valid_data[valid_data['Date'] <= train_cutoff].copy()
    test_data = valid_data[valid_data['Date'] >= test_start].copy()
    
    # Features
    X_train = train_data[model_features].fillna(0.5).values
    X_test = test_data[model_features].fillna(0.5).values
    
    # Targets for different models
    # Original 3-class
    y_train_3class = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    y_test_3class = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    # Binary: Draw vs Non-Draw
    y_train_binary = (y_train_3class == 1).astype(int)  # 1 if Draw, 0 if H or A
    y_test_binary = (y_test_3class == 1).astype(int)
    
    # Home vs Away (only for non-draws)
    non_draw_mask_train = y_train_3class != 1
    non_draw_mask_test = y_test_3class != 1
    
    X_train_ha = X_train[non_draw_mask_train]
    X_test_ha = X_test[non_draw_mask_test] 
    y_train_ha = y_train_3class[non_draw_mask_train]  # 0=H, 2=A
    y_test_ha = y_test_3class[non_draw_mask_test]
    
    # Convert A(2) to 1 for binary H vs A
    y_train_ha = (y_train_ha == 2).astype(int)  # 0=Home, 1=Away
    y_test_ha = (y_test_ha == 2).astype(int)
    
    logger.info(f"📊 Data prepared:")
    logger.info(f"  Full dataset: {len(X_train)} train, {len(X_test)} test")
    logger.info(f"  Draw detection: {sum(y_train_binary)} draws / {len(y_train_binary)} total")
    logger.info(f"  H vs A: {len(X_train_ha)} train, {len(X_test_ha)} test")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train_3class': y_train_3class, 'y_test_3class': y_test_3class,
        'y_train_binary': y_train_binary, 'y_test_binary': y_test_binary,
        'X_train_ha': X_train_ha, 'X_test_ha': X_test_ha,
        'y_train_ha': y_train_ha, 'y_test_ha': y_test_ha
    }

def train_baseline_model(data):
    """Baseline single model for comparison"""
    logger = setup_logging()
    logger.info("=== TRAINING BASELINE MODEL ===")
    
    # Standard RF with exact production config
    baseline_rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    
    # With calibration like production
    baseline_model = CalibratedClassifierCV(baseline_rf, method='isotonic', cv=3)
    baseline_model.fit(data['X_train'], data['y_train_3class'])
    
    # Evaluate
    y_pred_baseline = baseline_model.predict(data['X_test'])
    accuracy_baseline = accuracy_score(data['y_test_3class'], y_pred_baseline)
    f1_macro_baseline = f1_score(data['y_test_3class'], y_pred_baseline, average='macro')
    
    logger.info(f"🎯 Baseline Accuracy: {accuracy_baseline:.4f} ({accuracy_baseline*100:.2f}%)")
    logger.info(f"📊 Baseline F1-Macro: {f1_macro_baseline:.4f}")
    
    print("\n📊 Baseline Classification Report:")
    print(classification_report(data['y_test_3class'], y_pred_baseline, 
                              target_names=['Home', 'Draw', 'Away']))
    
    return {
        'model': baseline_model,
        'accuracy': accuracy_baseline,
        'f1_macro': f1_macro_baseline,
        'y_pred': y_pred_baseline
    }

class HybridCascadeModel:
    """
    Two-stage cascade model:
    1. Draw detector (binary, can use SMOTE)
    2. H vs A classifier (for non-draws)
    """
    
    def __init__(self, draw_threshold=0.5, use_smote=True):
        self.draw_threshold = draw_threshold
        self.use_smote = use_smote
        self.draw_detector = None
        self.ha_classifier = None
        
    def fit(self, data):
        logger = setup_logging()
        logger.info(f"=== TRAINING HYBRID CASCADE MODEL ===")
        logger.info(f"Draw threshold: {self.draw_threshold}")
        logger.info(f"Use SMOTE: {self.use_smote}")
        
        # Train Draw Detector
        if self.use_smote:
            # Use SMOTE to balance Draw vs Non-Draw
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(data['X_train'], data['y_train_binary'])
            logger.info(f"📊 SMOTE: {len(data['X_train'])} → {len(X_balanced)} samples")
            
            draw_rf = RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            self.draw_detector = CalibratedClassifierCV(draw_rf, method='isotonic', cv=3)
            self.draw_detector.fit(X_balanced, y_balanced)
        else:
            draw_rf = RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            self.draw_detector = CalibratedClassifierCV(draw_rf, method='isotonic', cv=3)
            self.draw_detector.fit(data['X_train'], data['y_train_binary'])
        
        # Train H vs A Classifier (no need for resampling, naturally balanced)
        ha_rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        self.ha_classifier = CalibratedClassifierCV(ha_rf, method='isotonic', cv=3)
        self.ha_classifier.fit(data['X_train_ha'], data['y_train_ha'])
        
        logger.info("✅ Cascade model trained")
        
    def predict(self, X):
        # Stage 1: Detect draws
        draw_proba = self.draw_detector.predict_proba(X)[:, 1]  # Probability of draw
        
        # Stage 2: For non-draws, predict H vs A
        ha_pred = self.ha_classifier.predict(X)  # 0=Home, 1=Away
        
        # Combine predictions
        final_pred = np.zeros(len(X))
        
        for i in range(len(X)):
            if draw_proba[i] >= self.draw_threshold:
                final_pred[i] = 1  # Draw
            else:
                if ha_pred[i] == 0:
                    final_pred[i] = 0  # Home
                else:
                    final_pred[i] = 2  # Away
                    
        return final_pred.astype(int), draw_proba

def test_cascade_model(data, draw_threshold=0.5, use_smote=True):
    """Test cascade model with given parameters"""
    logger = setup_logging()
    logger.info(f"=== TESTING CASCADE MODEL (thresh={draw_threshold}, SMOTE={use_smote}) ===")
    
    # Train cascade model
    cascade = HybridCascadeModel(draw_threshold=draw_threshold, use_smote=use_smote)
    cascade.fit(data)
    
    # Predict
    y_pred_cascade, draw_probabilities = cascade.predict(data['X_test'])
    
    # Evaluate
    accuracy_cascade = accuracy_score(data['y_test_3class'], y_pred_cascade)
    f1_macro_cascade = f1_score(data['y_test_3class'], y_pred_cascade, average='macro')
    
    logger.info(f"🎯 Cascade Accuracy: {accuracy_cascade:.4f} ({accuracy_cascade*100:.2f}%)")
    logger.info(f"📊 Cascade F1-Macro: {f1_macro_cascade:.4f}")
    
    # Draw statistics
    draws_predicted = (y_pred_cascade == 1).sum()
    draws_actual = (data['y_test_3class'] == 1).sum()
    draw_recall = (y_pred_cascade[data['y_test_3class'] == 1] == 1).sum() / draws_actual
    
    logger.info(f"📊 Draw Stats: {draws_predicted} predicted, {draws_actual} actual")
    logger.info(f"📊 Draw Recall: {draw_recall:.1%}")
    
    print(f"\n📊 Cascade Model (thresh={draw_threshold}) Classification Report:")
    print(classification_report(data['y_test_3class'], y_pred_cascade,
                              target_names=['Home', 'Draw', 'Away']))
    
    return {
        'model': cascade,
        'accuracy': accuracy_cascade,
        'f1_macro': f1_macro_cascade,
        'y_pred': y_pred_cascade,
        'draw_recall': draw_recall,
        'draw_probabilities': draw_probabilities
    }

def main():
    """Test cascade approach vs baseline"""
    logger = setup_logging()
    logger.info("🚀 TESTING HYBRID CASCADE MODEL FOR DRAW PREDICTION")
    logger.info("="*70)
    
    # Prepare data
    data = prepare_data_for_cascade()
    
    # Test baseline
    logger.info("\n" + "="*50)
    baseline_results = train_baseline_model(data)
    
    # Test cascade with different thresholds
    cascade_results = {}
    
    thresholds_to_test = [0.3, 0.4, 0.5]
    
    for threshold in thresholds_to_test:
        logger.info("\n" + "="*50)
        key = f"cascade_{threshold}_smote"
        cascade_results[key] = test_cascade_model(data, 
                                                draw_threshold=threshold, 
                                                use_smote=True)
    
    # Test without SMOTE
    logger.info("\n" + "="*50)
    cascade_results['cascade_0.4_no_smote'] = test_cascade_model(data,
                                                               draw_threshold=0.4,
                                                               use_smote=False)
    
    # Final comparison
    logger.info("\n" + "="*70)
    logger.info("🏆 FINAL COMPARISON")
    logger.info("="*70)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'Approach':<25} {'Accuracy':<10} {'F1-Macro':<10} {'Draw Recall':<12}")
    print("-" * 60)
    print(f"{'Baseline':<25} {baseline_results['accuracy']:<10.3f} {baseline_results['f1_macro']:<10.3f} {'-':<12}")
    
    for name, result in cascade_results.items():
        print(f"{name:<25} {result['accuracy']:<10.3f} {result['f1_macro']:<10.3f} {result['draw_recall']:<12.1%}")
    
    # Find best approach
    best_accuracy = baseline_results['accuracy']
    best_f1 = baseline_results['f1_macro']
    best_name = "Baseline"
    
    for name, result in cascade_results.items():
        if result['f1_macro'] > best_f1:
            best_f1 = result['f1_macro']
            best_name = name
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
    
    logger.info(f"\n🏆 Best F1-Macro: {best_name} ({best_f1:.4f})")
    logger.info(f"🎯 Best Accuracy: {best_accuracy:.3f}")
    
    # Recommendation
    logger.info(f"\n📋 RECOMMENDATION:")
    if best_name != "Baseline":
        logger.info(f"✅ UPGRADE to {best_name} - Better balanced performance")
    else:
        logger.info("🤔 Baseline still best - Cascade needs refinement")
    
    logger.info("="*70)

if __name__ == "__main__":
    main()
    print("\n🎯 Hybrid cascade experiment complete!")
    print("Check logs for detailed analysis.")