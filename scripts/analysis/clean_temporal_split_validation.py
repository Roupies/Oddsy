#!/usr/bin/env python3
"""
CLEAN TEMPORAL SPLIT VALIDATION
Address Gemini's critical feedback: Fix the "Frankenstein" test set
Use proper football season boundaries without complex feature recalculation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def create_clean_temporal_split():
    """
    Create proper temporal split based on football seasons - NO COMPLEX FIXES
    Train: 2019-2024 (5 complete seasons), Test: 2024-2025 (1 complete season)
    """
    logger = setup_logging()
    logger.info("=== ⚽ CLEAN FOOTBALL SEASON SPLIT ===")
    
    # Load original dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} matches")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Seasons: {sorted(df['Season'].unique())}")
    
    # Season-based split (no calendar date confusion)
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']
    
    train_data = df[df['Season'].isin(train_seasons)].copy()
    test_data = df[df['Season'].isin(test_seasons)].copy()
    
    logger.info(f"\n📊 CLEAN TEMPORAL SPLIT:")
    logger.info(f"Training seasons: {train_seasons} ({len(train_seasons)} seasons)")
    logger.info(f"Testing seasons: {test_seasons} ({len(test_seasons)} season)")
    logger.info(f"Training matches: {len(train_data)}")
    logger.info(f"Testing matches: {len(test_data)}")
    
    # Verify split integrity
    logger.info(f"\n🔍 SPLIT INTEGRITY CHECK:")
    
    # 1. No season overlap
    train_season_set = set(train_data['Season'])
    test_season_set = set(test_data['Season'])
    overlap = train_season_set.intersection(test_season_set)
    
    if len(overlap) > 0:
        logger.error(f"❌ Season overlap: {overlap}")
        return None, None
    else:
        logger.info("✅ No season overlap")
    
    # 2. Temporal boundary
    latest_train_date = train_data['Date'].max()
    earliest_test_date = test_data['Date'].min()
    
    logger.info(f"Latest training date: {latest_train_date}")
    logger.info(f"Earliest test date: {earliest_test_date}")
    
    if latest_train_date >= earliest_test_date:
        logger.error("❌ Temporal violation")
        return None, None
    else:
        logger.info("✅ Proper temporal separation")
    
    # 3. Complete seasons
    expected_matches = 380  # Premier League standard
    
    for season in train_seasons:
        actual = len(train_data[train_data['Season'] == season])
        if actual != expected_matches:
            logger.warning(f"⚠️  {season}: {actual}/{expected_matches} matches")
        else:
            logger.info(f"✅ {season}: Complete ({actual} matches)")
    
    if len(test_data) != expected_matches:
        logger.warning(f"⚠️  Test season incomplete: {len(test_data)}/{expected_matches}")
    else:
        logger.info(f"✅ Test season complete: {len(test_data)} matches")
    
    # 4. Data quality check
    train_missing = train_data['FullTimeResult'].isnull().sum()
    test_missing = test_data['FullTimeResult'].isnull().sum()
    
    logger.info(f"Missing results: Train={train_missing}, Test={test_missing}")
    
    if train_missing > 0 or test_missing > 0:
        logger.error("❌ Missing target values")
        return None, None
    else:
        logger.info("✅ No missing targets")
    
    return train_data, test_data

def validate_clean_split_performance():
    """
    Validate performance with clean temporal split
    """
    logger = setup_logging()
    logger.info("=== 🏆 CLEAN SPLIT PERFORMANCE VALIDATION ===")
    
    # Get clean split
    train_data, test_data = create_clean_temporal_split()
    if train_data is None:
        logger.error("❌ Failed to create clean split")
        return None
    
    # Same features as v1.4 breakthrough
    features = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    # Prepare data
    X_train = train_data[features].fillna(0.5)
    X_test = test_data[features].fillna(0.5)
    
    # Encode targets
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_data['FullTimeResult'].map(label_mapping)
    y_test = test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"\n📋 VALIDATION SETUP:")
    logger.info(f"Features: {features}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Training seasons: {len(train_data['Season'].unique())}")
    logger.info(f"Test seasons: {len(test_data['Season'].unique())}")
    
    # Check data quality
    logger.info(f"\n🔍 DATA QUALITY:")
    logger.info(f"Train missing: {X_train.isnull().sum().sum()}")
    logger.info(f"Test missing: {X_test.isnull().sum().sum()}")
    logger.info(f"Train target NaN: {y_train.isnull().sum()}")
    logger.info(f"Test target NaN: {y_test.isnull().sum()}")
    
    # Train model with breakthrough configuration
    model_config = {
        'n_estimators': 300,
        'max_depth': 18,  # Key breakthrough parameter
        'max_features': 'log2',
        'min_samples_leaf': 2,
        'min_samples_split': 15,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    logger.info(f"\n🚀 TRAINING MODEL:")
    logger.info(f"Algorithm: RandomForest")
    logger.info(f"Breakthrough config: max_depth={model_config['max_depth']}")
    
    # Train
    model = RandomForestClassifier(**model_config)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    logger.info(f"\n📊 CLEAN SPLIT PERFORMANCE:")
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Overfitting gap: {(train_accuracy - test_accuracy)*100:.1f}pp")
    
    # Detailed predictions
    y_pred = model.predict(X_test)
    
    # Performance breakdown
    report = classification_report(y_test, y_pred, target_names=['Home', 'Draw', 'Away'], output_dict=True)
    
    logger.info(f"\n🎯 PERFORMANCE BREAKDOWN:")
    logger.info(f"Home: {report['Home']['precision']:.3f}P {report['Home']['recall']:.3f}R {report['Home']['f1-score']:.3f}F1")
    logger.info(f"Draw: {report['Draw']['precision']:.3f}P {report['Draw']['recall']:.3f}R {report['Draw']['f1-score']:.3f}F1") 
    logger.info(f"Away: {report['Away']['precision']:.3f}P {report['Away']['recall']:.3f}R {report['Away']['f1-score']:.3f}F1")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n📈 CONFUSION MATRIX:")
    logger.info(f"         H    D    A")
    logger.info(f"True H: {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
    logger.info(f"True D: {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
    logger.info(f"True A: {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")
    
    # Feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n🎯 FEATURE IMPORTANCE:")
    for i, (feature, importance) in enumerate(sorted_importance, 1):
        logger.info(f"  {i}. {feature}: {importance:.3f}")
    
    # Performance targets
    logger.info(f"\n🏆 PERFORMANCE TARGETS:")
    targets = [
        ("Random", 0.333, "🎲"),
        ("Majority Class", 0.436, "📊"), 
        ("Good Model", 0.500, "✅"),
        ("Excellent", 0.550, "🏆"),
        ("Industry Leading", 0.580, "🚀")
    ]
    
    for name, threshold, emoji in targets:
        achieved = test_accuracy >= threshold
        status = "✅ ACHIEVED" if achieved else "❌ MISSED"
        margin = (test_accuracy - threshold) * 100
        logger.info(f"  {emoji} {name} ({threshold:.1%}): {status} ({margin:+.1f}pp)")
    
    # Historical comparison
    logger.info(f"\n📈 HISTORICAL COMPARISON:")
    comparisons = [
        ("Previous (mixed seasons)", 0.5621),
        ("v1.4 (calendar split)", 0.5567)
    ]
    
    for name, prev_score in comparisons:
        diff = (test_accuracy - prev_score) * 100
        logger.info(f"  {name}: {prev_score:.4f} → {test_accuracy:.4f} ({diff:+.2f}pp)")
    
    # Final assessment
    logger.info(f"\n" + "="*80)
    logger.info("🏁 DEFINITIVE ASSESSMENT")
    logger.info("="*80)
    
    if test_accuracy >= 0.58:
        status = "🚀 INDUSTRY LEADING"
        message = "Exceptional performance - Ready for commercial deployment"
    elif test_accuracy >= 0.55:
        status = "🏆 EXCELLENT"
        message = "Industry competitive - Strong commercial potential"
    elif test_accuracy >= 0.50:
        status = "✅ GOOD"
        message = "Solid baseline - Further development recommended"
    elif test_accuracy >= 0.436:
        status = "📊 MODERATE" 
        message = "Beats random but needs improvement"
    else:
        status = "❌ INSUFFICIENT"
        message = "Below baseline - Significant work needed"
    
    logger.info(f"STATUS: {status}")
    logger.info(f"SCORE: {test_accuracy:.4f} ({test_accuracy:.1%})")
    logger.info(f"ASSESSMENT: {message}")
    
    # Addressing Gemini's concerns
    logger.info(f"\n🎯 GEMINI'S FEEDBACK ADDRESSED:")
    logger.info("✅ Fixed 'Frankenstein' test set - Pure 2024-2025 season")
    logger.info("✅ Proper football season boundaries - No calendar confusion") 
    logger.info("✅ Clean temporal separation - 5 seasons train, 1 season test")
    logger.info("✅ No complex feature recalculation - Original features preserved")
    logger.info(f"✅ Definitive performance: {test_accuracy:.4f} on {len(test_data)} matches")
    
    # Save results
    results = {
        'test_accuracy': float(test_accuracy),
        'train_accuracy': float(train_accuracy), 
        'overfitting_gap': float(train_accuracy - test_accuracy),
        'clean_temporal_split': True,
        'train_seasons': 5,
        'test_seasons': 1,
        'train_matches': len(train_data),
        'test_matches': len(test_data),
        'feature_importance': dict(sorted_importance),
        'performance_targets': {target[0]: test_accuracy >= target[1] for target in targets},
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'addresses_gemini_feedback': True
    }
    
    # Save definitive results
    os.makedirs("evaluation", exist_ok=True)
    results_file = f"evaluation/definitive_clean_validation_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"💾 Definitive results saved: {results_file}")
    
    return results

def run_definitive_clean_validation():
    """
    Execute definitive validation addressing all of Gemini's concerns
    """
    logger = setup_logging()
    logger.info("🔬 DEFINITIVE CLEAN TEMPORAL VALIDATION")
    logger.info("=" * 80)
    logger.info("Addressing Gemini's Priority #1 Critical Feedback:")
    logger.info("• Fix 'Frankenstein' test set (mixed 2023-24 + 2024-25)")
    logger.info("• Use proper football season boundaries")  
    logger.info("• Validate on inattackable temporal split")
    logger.info("• Establish definitive baseline for v2 development")
    logger.info("=" * 80)
    
    results = validate_clean_split_performance()
    
    if results is None:
        logger.error("❌ Validation failed")
        return False
    
    # Success criteria
    success = results['test_accuracy'] >= 0.50
    
    logger.info(f"\n🏆 VALIDATION OUTCOME:")
    logger.info(f"Success: {'✅ PASSED' if success else '❌ FAILED'}")
    logger.info(f"Score: {results['test_accuracy']:.4f}")
    logger.info(f"Baseline Established: {'Yes' if success else 'No'}")
    
    return success

if __name__ == "__main__":
    try:
        success = run_definitive_clean_validation()
        
        print(f"\n🔬 DEFINITIVE CLEAN VALIDATION COMPLETE")
        
        if success:
            print("✅ Definitive baseline established with clean temporal split")
            print("🎯 Ready for v2 development with inattackable foundation")
        else:
            print("❌ Validation failed - Review results")
            
    except Exception as e:
        print(f"💥 Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)