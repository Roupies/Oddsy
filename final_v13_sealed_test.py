#!/usr/bin/env python3
"""
FINAL v1.3 sealed test with proper dataset
Real dates + market_entropy_norm feature
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

def final_v13_sealed_test():
    """
    Final sealed test with perfect v1.3 setup
    """
    logger = setup_logging()
    logger.info("=== 🏆 FINAL v1.3 SEALED TEST (PROPER DATASET) ===")
    
    # Load the properly combined dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Perfect v1.3 dataset loaded: {len(df)} matches ({df['Date'].min()} → {df['Date'].max()})")
    
    # 🔪 PROPER TEMPORAL SPLIT: Pre-2024 vs 2024+
    cutoff_date = '2024-01-01'
    train_dev_data = df[df['Date'] < cutoff_date].copy()
    sealed_test_data = df[df['Date'] >= cutoff_date].copy()
    
    logger.info(f"\n📊 PROPER TEMPORAL SPLIT:")
    logger.info(f"  Train/Dev: {len(train_dev_data)} matches ({train_dev_data['Date'].min()} → {train_dev_data['Date'].max()})")
    logger.info(f"  Sealed:    {len(sealed_test_data)} matches ({sealed_test_data['Date'].min()} → {sealed_test_data['Date'].max()})")
    logger.info(f"  Split ratio: {len(train_dev_data)/len(df):.1%} train, {len(sealed_test_data)/len(df):.1%} test")
    
    # v1.3 COMPLETE feature set (7 features)
    v13_complete_features = [
        "form_diff_normalized",
        "elo_diff_normalized", 
        "h2h_score",
        "matchday_normalized",
        "shots_diff_normalized",
        "corners_diff_normalized",
        "market_entropy_norm"  # 🎯 The key 7th feature
    ]
    
    logger.info(f"v1.3 COMPLETE features: {v13_complete_features}")
    
    # Prepare data
    X_train_dev = train_dev_data[v13_complete_features].fillna(0.5)
    X_sealed = sealed_test_data[v13_complete_features].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train_dev = train_dev_data['FullTimeResult'].map(label_mapping)
    y_sealed = sealed_test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"\nData shapes:")
    logger.info(f"  Train/Dev: {X_train_dev.shape}")
    logger.info(f"  Sealed:    {X_sealed.shape}")
    
    # Data quality verification
    logger.info(f"\n🔍 DATA QUALITY (MARKET INTELLIGENCE):")
    market_stats = X_train_dev['market_entropy_norm']
    logger.info(f"  market_entropy_norm: min={market_stats.min():.3f}, max={market_stats.max():.3f}, mean={market_stats.mean():.3f}")
    logger.info(f"  Missing values: {X_train_dev.isnull().sum().sum()}")
    logger.info(f"  Zero values: {(X_train_dev == 0).sum().sum()}")
    
    # 🧪 CROSS-VALIDATION on Train/Dev (2019-2023)
    logger.info(f"\n🧪 CROSS-VALIDATION (2019-2023 ONLY)")
    logger.info("=" * 60)
    
    # v1.3 optimal model (RandomForest equivalent to XGBoost Conservative)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Proper TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train_dev, y_train_dev, cv=tscv, scoring='accuracy')
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    logger.info(f"CV Results (7 features, 2019-2023):")
    logger.info(f"  Individual folds: {[f'{s:.4f}' for s in cv_scores]}")
    logger.info(f"  Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  95% CI: [{cv_mean-1.96*cv_std:.4f}, {cv_mean+1.96*cv_std:.4f}]")
    
    # Compare with reported v1.3 performance
    reported_v13 = 0.5305
    cv_vs_reported = cv_mean - reported_v13
    logger.info(f"\n📊 COMPARISON WITH REPORTED v1.3:")
    logger.info(f"  Reported v1.3 performance: {reported_v13:.4f}")
    logger.info(f"  Our CV performance:        {cv_mean:.4f}")
    logger.info(f"  Difference:                {cv_vs_reported*100:+.2f}pp")
    
    # 🎯 TRAIN FINAL MODEL
    logger.info(f"\n🎯 TRAINING FINAL MODEL (2019-2023)")
    model.fit(X_train_dev, y_train_dev)
    train_acc = model.score(X_train_dev, y_train_dev)
    logger.info(f"Training accuracy: {train_acc:.4f}")
    
    # 🔐 THE MOMENT OF TRUTH - SEALED TEST
    logger.info(f"\n🔐 SEALED TEST - FINAL v1.3 PERFORMANCE")
    logger.info("=" * 70)
    logger.info("⚠️  FIRST AND ONLY EVALUATION ON 2024-2025")
    
    # Predict
    y_pred_sealed = model.predict(X_sealed)
    sealed_accuracy = accuracy_score(y_sealed, y_pred_sealed)
    
    logger.info(f"\n🎯 SEALED TEST RESULTS:")
    logger.info(f"  Accuracy: {sealed_accuracy:.4f} ({sealed_accuracy*100:.2f}%)")
    
    # Performance analysis
    performance_drop = cv_mean - sealed_accuracy
    logger.info(f"\n📈 PERFORMANCE ANALYSIS:")
    logger.info(f"  CV Performance (2019-2023):    {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Sealed Performance (2024-25):  {sealed_accuracy:.4f}")
    logger.info(f"  Generalization Gap:            {performance_drop*100:+.2f}pp")
    
    # Statistical significance of the gap
    if abs(performance_drop) < cv_std:
        significance = "🟢 Excellent generalization (< 1σ)"
    elif abs(performance_drop) < 2*cv_std:
        significance = "🟡 Good generalization (< 2σ)"
    else:
        significance = "🔴 Poor generalization (> 2σ)"
    logger.info(f"  Significance:                  {significance}")
    
    # Business performance
    logger.info(f"\n💰 BUSINESS METRICS:")
    baselines = {
        'Random (33.3%)': 0.333,
        'Majority Class': max(np.bincount(y_sealed)) / len(y_sealed),
        'Good Target (50%)': 0.50,
        'Excellent Target (55%)': 0.55,
        'Industry Elite (60%)': 0.60
    }
    
    for name, threshold in baselines.items():
        status = "✅ ACHIEVED" if sealed_accuracy > threshold else "❌ MISSED"
        gap = (sealed_accuracy - threshold) * 100
        logger.info(f"  {name:20s}: {status} ({gap:+.1f}pp)")
    
    # Feature importance with market intelligence
    feature_importance = list(zip(v13_complete_features, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n🎯 FEATURE IMPORTANCE RANKING:")
    for i, (feature, importance) in enumerate(feature_importance, 1):
        market_flag = " 🎯" if feature == "market_entropy_norm" else ""
        logger.info(f"  {i}. {feature:25s}: {importance:.4f}{market_flag}")
    
    # Market intelligence analysis
    market_importance = dict(feature_importance)['market_entropy_norm']
    market_rank = [f[0] for f in feature_importance].index('market_entropy_norm') + 1
    logger.info(f"\n📊 MARKET INTELLIGENCE ANALYSIS:")
    logger.info(f"  market_entropy_norm importance: {market_importance:.4f}")
    logger.info(f"  Ranking: #{market_rank} out of 7 features")
    
    if market_rank <= 3:
        logger.info(f"  🏆 TOP-3 FEATURE - Critical for performance")
    elif market_rank <= 5:
        logger.info(f"  ✅ VALUABLE FEATURE - Contributes significantly")
    else:
        logger.info(f"  ⚠️ MINOR FEATURE - Limited contribution")
    
    # Detailed classification performance
    logger.info(f"\n📋 CLASSIFICATION BREAKDOWN:")
    class_names = ['Home', 'Draw', 'Away']
    report = classification_report(y_sealed, y_pred_sealed, target_names=class_names, output_dict=True)
    
    for class_name in class_names:
        metrics = report[class_name]
        logger.info(f"  {class_name:4s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # v2.0 roadmap implications
    logger.info(f"\n🚀 v2.0 DEVELOPMENT IMPLICATIONS:")
    if sealed_accuracy >= 0.55:
        logger.info(f"  ✅ v1.3 achieves EXCELLENT performance")
        logger.info(f"  🎯 v2.0 target: 58-62% (industry leading)")
        logger.info(f"  💡 Focus: Advanced market features (line movement, sharp money)")
    elif sealed_accuracy >= 0.52:
        logger.info(f"  ✅ v1.3 provides solid foundation")
        logger.info(f"  🎯 v2.0 target: 55-58% (industry competitive)")
        logger.info(f"  💡 Priority: Market intelligence expansion + xG data")
    else:
        logger.info(f"  ⚠️ v1.3 below expectations")
        logger.info(f"  🔍 Need: Fundamental feature engineering review")
    
    # Final verdict
    logger.info(f"\n" + "="*80)
    logger.info(f"🏆 FINAL v1.3 SEALED TEST VERDICT")
    logger.info(f"="*80)
    
    if sealed_accuracy >= 0.55:
        verdict = f"🏆 BREAKTHROUGH: {sealed_accuracy:.4f} - Industry competitive!"
        grade = "A"
    elif sealed_accuracy >= 0.52:
        verdict = f"✅ EXCELLENT: {sealed_accuracy:.4f} - Strong foundation for v2.0"
        grade = "B+"
    elif sealed_accuracy >= 0.50:
        verdict = f"✅ GOOD: {sealed_accuracy:.4f} - Beats all baselines"
        grade = "B"
    elif sealed_accuracy >= 0.45:
        verdict = f"⚠️ ACCEPTABLE: {sealed_accuracy:.4f} - Needs improvement"
        grade = "C+"
    else:
        verdict = f"❌ POOR: {sealed_accuracy:.4f} - Below expectations"
        grade = "F"
    
    logger.info(f"{verdict}")
    logger.info(f"Grade: {grade}")
    logger.info(f"Ready for v2.0 development: {'YES' if sealed_accuracy >= 0.50 else 'NEEDS WORK'}")
    
    # Save final results
    results = {
        'test_type': 'final_v1.3_sealed_test',
        'dataset': 'v13_complete_with_dates.csv',
        'features': v13_complete_features,
        'cv_performance': {
            'mean': float(cv_mean),
            'std': float(cv_std),
            'individual_scores': [float(s) for s in cv_scores]
        },
        'sealed_performance': float(sealed_accuracy),
        'generalization_gap': float(performance_drop),
        'market_intelligence': {
            'importance': float(market_importance),
            'rank': int(market_rank)
        },
        'business_metrics': {
            name.split()[0].lower(): sealed_accuracy > threshold 
            for name, threshold in baselines.items()
        },
        'grade': grade,
        'ready_for_v2': sealed_accuracy >= 0.50
    }
    
    return results

if __name__ == "__main__":
    try:
        results = final_v13_sealed_test()
        print(f"\n🏆 FINAL v1.3 SEALED TEST COMPLETE")
        print(f"Performance: {results['sealed_performance']:.4f}")
        print(f"Grade: {results['grade']}")
        print(f"Market Intelligence Rank: #{results['market_intelligence']['rank']}")
        print(f"Ready for v2.0: {results['ready_for_v2']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()