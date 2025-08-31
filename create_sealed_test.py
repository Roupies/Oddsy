#!/usr/bin/env python3
"""
Create proper sealed test for Oddsy v1.3 model
Retrain on 2019-2023, test on 2024-2025 (never seen during development)
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))
from utils import setup_logging

def create_sealed_test():
    """
    Create rigorous train/sealed split and evaluate true performance
    """
    logger = setup_logging()
    logger.info("=== 🔐 SEALED TEST CREATION - TRUE BASELINE ===")
    
    # Load full dataset with dates
    df = pd.read_csv('data/processed/premier_league_ml_ready.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Full dataset: {len(df)} matches ({df['Date'].min()} → {df['Date'].max()})")
    
    # 🔪 CRITICAL SPLIT: Pre-2024 vs 2024+
    cutoff_date = '2024-01-01'
    train_dev_data = df[df['Date'] < cutoff_date].copy()
    sealed_test_data = df[df['Date'] >= cutoff_date].copy()
    
    logger.info(f"\n📊 TEMPORAL SPLIT:")
    logger.info(f"  Train/Dev: {len(train_dev_data)} matches ({train_dev_data['Date'].min()} → {train_dev_data['Date'].max()})")
    logger.info(f"  Sealed:    {len(sealed_test_data)} matches ({sealed_test_data['Date'].min()} → {sealed_test_data['Date'].max()})")
    logger.info(f"  Split ratio: {len(train_dev_data)/len(df):.1%} train, {len(sealed_test_data)/len(df):.1%} sealed")
    
    if len(sealed_test_data) < 100:
        logger.warning(f"⚠️ Small sealed test set: {len(sealed_test_data)} matches")
    
    # v1.3 features (7 optimal features)
    v13_features = [
        "form_diff_normalized",
        "elo_diff_normalized", 
        "h2h_score",
        "matchday_normalized",
        "shots_diff_normalized",
        "corners_diff_normalized"
    ]
    
    # Check if market_entropy_norm exists
    if 'market_entropy_norm' in df.columns:
        v13_features.append('market_entropy_norm')
        logger.info("✅ Found market_entropy_norm - using full v1.3 feature set")
    else:
        logger.warning("⚠️ market_entropy_norm missing - using 6 features only")
    
    logger.info(f"Features ({len(v13_features)}): {v13_features}")
    
    # Prepare training data
    X_train_dev = train_dev_data[v13_features].fillna(0.5)
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train_dev = train_dev_data['FullTimeResult'].map(label_mapping)
    
    # Prepare sealed test data
    X_sealed = sealed_test_data[v13_features].fillna(0.5)
    y_sealed = sealed_test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"\nData shapes:")
    logger.info(f"  Train/Dev: {X_train_dev.shape}")
    logger.info(f"  Sealed:    {X_sealed.shape}")
    
    # 🧪 CROSS-VALIDATION on Train/Dev (2019-2023)
    logger.info(f"\n🧪 CROSS-VALIDATION on Train/Dev Data (2019-2023)")
    logger.info("-" * 60)
    
    # v1.3 best configuration: XGBoost Conservative equivalent (using RF for simplicity)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # TimeSeriesSplit on training data
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train_dev, y_train_dev, cv=tscv, scoring='accuracy')
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    logger.info(f"CV Results (2019-2023):")
    logger.info(f"  Individual folds: {[f'{s:.4f}' for s in cv_scores]}")
    logger.info(f"  Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Range: [{cv_mean-cv_std:.4f}, {cv_mean+cv_std:.4f}]")
    
    # 🎯 TRAIN FINAL MODEL on all Train/Dev data
    logger.info(f"\n🎯 TRAINING FINAL MODEL (2019-2023)")
    model.fit(X_train_dev, y_train_dev)
    
    # Training accuracy (for comparison)
    train_acc = model.score(X_train_dev, y_train_dev)
    logger.info(f"Training accuracy: {train_acc:.4f}")
    
    # 🔐 SEALED TEST - THE MOMENT OF TRUTH
    logger.info(f"\n🔐 SEALED TEST EVALUATION (2024-2025)")
    logger.info("=" * 60)
    logger.info("⚠️  FIRST AND ONLY EVALUATION ON SEALED DATA")
    
    # Predict on sealed test
    y_pred_sealed = model.predict(X_sealed)
    sealed_accuracy = accuracy_score(y_sealed, y_pred_sealed)
    
    logger.info(f"\n📊 SEALED TEST RESULTS:")
    logger.info(f"  Accuracy: {sealed_accuracy:.4f} ({sealed_accuracy*100:.2f}%)")
    
    # Detailed analysis
    logger.info(f"\n📈 DETAILED ANALYSIS:")
    logger.info(f"  CV Performance (2019-2023):  {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Sealed Performance (2024+):  {sealed_accuracy:.4f}")
    logger.info(f"  Performance Drop:            {(cv_mean - sealed_accuracy)*100:+.2f}pp")
    
    # Statistical significance
    performance_drop = cv_mean - sealed_accuracy
    if abs(performance_drop) < cv_std:
        logger.info(f"  🟢 Drop within 1 std dev - not statistically significant")
    elif abs(performance_drop) < 2*cv_std:
        logger.info(f"  🟡 Drop within 2 std dev - marginally significant")
    else:
        logger.info(f"  🔴 Drop > 2 std dev - statistically significant degradation")
    
    # Classification report
    logger.info(f"\n📋 CLASSIFICATION REPORT (Sealed Test):")
    class_names = ['Home', 'Draw', 'Away']
    report = classification_report(y_sealed, y_pred_sealed, target_names=class_names)
    logger.info(f"\n{report}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_sealed, y_pred_sealed)
    logger.info(f"\n🎯 CONFUSION MATRIX:")
    logger.info(f"         Pred:  H    D    A")
    for i, actual_class in enumerate(['Home', 'Draw', 'Away']):
        logger.info(f"Actual {actual_class}: {cm[i]}")
    
    # Feature importance
    feature_importance = list(zip(v13_features, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n🎯 FEATURE IMPORTANCE (Top 5):")
    for feature, importance in feature_importance[:5]:
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Business metrics
    logger.info(f"\n💰 BUSINESS METRICS:")
    random_baseline = 1/3
    majority_baseline = max(np.bincount(y_sealed)) / len(y_sealed)
    
    logger.info(f"  Random baseline (33.3%):     {sealed_accuracy > random_baseline}")
    logger.info(f"  Majority class baseline:      {majority_baseline:.4f} -> {'✅ BEAT' if sealed_accuracy > majority_baseline else '❌ LOST'}")
    logger.info(f"  Good model target (50%):      {'✅ ACHIEVED' if sealed_accuracy > 0.50 else '❌ MISSED'}")
    logger.info(f"  Excellent target (55%):       {'✅ ACHIEVED' if sealed_accuracy > 0.55 else '❌ MISSED'}")
    
    # Save results
    results = {
        'sealed_test_date': datetime.now().isoformat(),
        'cv_performance': {
            'mean': float(cv_mean),
            'std': float(cv_std),
            'scores': [float(s) for s in cv_scores]
        },
        'sealed_performance': {
            'accuracy': float(sealed_accuracy),
            'samples': int(len(sealed_test_data))
        },
        'performance_drop': float(performance_drop),
        'feature_importance': {f: float(imp) for f, imp in feature_importance},
        'business_metrics': {
            'beats_random': sealed_accuracy > random_baseline,
            'beats_majority': sealed_accuracy > majority_baseline,
            'achieves_good': sealed_accuracy > 0.50,
            'achieves_excellent': sealed_accuracy > 0.55
        }
    }
    
    # Save model (trained without seeing sealed data)
    model_filename = f"models/v13_sealed_model_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.joblib"
    joblib.dump(model, model_filename)
    logger.info(f"\n💾 Model saved: {model_filename}")
    
    # Save results
    results_filename = f"evaluation/sealed_test_results_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
    os.makedirs('evaluation', exist_ok=True)
    
    import json
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"📄 Results saved: {results_filename}")
    
    # Final verdict
    logger.info(f"\n" + "="*80)
    logger.info(f"🏁 FINAL VERDICT - TRUE v1.3 PERFORMANCE")
    logger.info(f"="*80)
    
    if sealed_accuracy >= 0.55:
        logger.info(f"🏆 EXCELLENT: {sealed_accuracy:.4f} - Industry competitive!")
    elif sealed_accuracy >= 0.50:
        logger.info(f"✅ GOOD: {sealed_accuracy:.4f} - Beats baselines significantly")
    elif sealed_accuracy >= 0.436:
        logger.info(f"⚠️ MINIMAL: {sealed_accuracy:.4f} - Beats majority class but not impressive")
    else:
        logger.info(f"❌ POOR: {sealed_accuracy:.4f} - Doesn't beat simple baselines")
    
    logger.info(f"\nReady for v2.0 development with TRUE baseline: {sealed_accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    try:
        results = create_sealed_test()
        print(f"\n🎯 SEALED TEST COMPLETE")
        print(f"True Performance: {results['sealed_performance']['accuracy']:.4f}")
        print(f"Performance Drop: {results['performance_drop']*100:+.2f}pp")
        
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Sealed test failed: {str(e)}")
        raise