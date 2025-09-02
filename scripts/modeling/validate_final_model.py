#!/usr/bin/env python3
"""
CANONICAL MODEL VALIDATION SCRIPT
Single source of truth for final model evaluation
Implements the definitive methodology discovered through rigorous testing

This script consolidates all validation learnings:
- Proper football season temporal split (not calendar dates)
- Clean feature integrity without complex recalculations  
- Comprehensive performance assessment
- Regression detection against established baselines

Usage:
    python3 scripts/modeling/validate_final_model.py
    python3 scripts/modeling/validate_final_model.py --model-config custom_config.json
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import json
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

# DEFINITIVE BASELINE ESTABLISHED THROUGH RIGOROUS TESTING
DEFINITIVE_BASELINE = 0.5184  # 51.84% - Clean temporal split validation
REGRESSION_TOLERANCE = 0.01   # 1pp tolerance for performance regression

def load_and_validate_data():
    """
    Load and validate the dataset with comprehensive integrity checks
    """
    logger = setup_logging()
    logger.info("=== 📊 LOADING AND VALIDATING DATA ===")
    
    data_path = 'data/processed/v13_complete_with_dates.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"❌ Dataset not found: {data_path}")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"✅ Dataset loaded: {len(df)} matches")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Seasons: {sorted(df['Season'].unique())}")
    
    # Validate data integrity
    validation_errors = []
    
    # 1. Check required columns
    required_columns = [
        'Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult',
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_errors.append(f"Missing columns: {missing_columns}")
    
    # 2. Check target variable integrity
    target_nulls = df['FullTimeResult'].isnull().sum()
    if target_nulls > 0:
        validation_errors.append(f"FullTimeResult has {target_nulls} null values")
    
    valid_results = {'H', 'D', 'A'}
    actual_results = set(df['FullTimeResult'].dropna().unique())
    if not actual_results.issubset(valid_results):
        validation_errors.append(f"Invalid FullTimeResult values: {actual_results - valid_results}")
    
    # 3. Check feature ranges
    feature_columns = [col for col in required_columns if col.endswith('_normalized') or col in ['h2h_score', 'market_entropy_norm']]
    
    for feature in feature_columns:
        if feature in df.columns:
            min_val = df[feature].min()
            max_val = df[feature].max()
            null_count = df[feature].isnull().sum()
            
            logger.info(f"  {feature}: [{min_val:.3f}, {max_val:.3f}] (nulls: {null_count})")
            
            if min_val < 0 or max_val > 1:
                validation_errors.append(f"{feature} outside [0,1] range: [{min_val:.3f}, {max_val:.3f}]")
    
    # 4. Check season completeness
    season_counts = df['Season'].value_counts()
    expected_matches_per_season = 380
    
    for season, count in season_counts.items():
        if count != expected_matches_per_season:
            logger.warning(f"⚠️  {season}: {count}/{expected_matches_per_season} matches (incomplete)")
    
    # 5. Check temporal order
    for season in df['Season'].unique():
        season_data = df[df['Season'] == season].sort_values('Date')
        if not season_data['Date'].is_monotonic_increasing:
            validation_errors.append(f"Temporal order violation in season {season}")
    
    # Report validation results
    if validation_errors:
        logger.error("❌ DATA VALIDATION FAILED:")
        for error in validation_errors:
            logger.error(f"  • {error}")
        return None
    else:
        logger.info("✅ Data validation passed - Dataset integrity confirmed")
        return df

def create_definitive_temporal_split(df):
    """
    Create the definitive temporal split using proper football seasons
    This methodology was established through rigorous testing and addresses
    the "Frankenstein test set" issue identified by expert review
    """
    logger = setup_logging()
    logger.info("=== ⚽ DEFINITIVE TEMPORAL SPLIT ===")
    
    # DEFINITIVE METHODOLOGY: Football season split (not calendar dates)
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']
    
    train_data = df[df['Season'].isin(train_seasons)].copy()
    test_data = df[df['Season'].isin(test_seasons)].copy()
    
    logger.info(f"Training seasons: {train_seasons} ({len(train_seasons)} seasons)")
    logger.info(f"Testing seasons: {test_seasons} ({len(test_seasons)} season)")
    logger.info(f"Training matches: {len(train_data)}")
    logger.info(f"Testing matches: {len(test_data)}")
    
    # Validate split integrity
    # 1. No season overlap
    train_season_set = set(train_data['Season'])
    test_season_set = set(test_data['Season'])
    overlap = train_season_set.intersection(test_season_set)
    
    if len(overlap) > 0:
        logger.error(f"❌ Season overlap detected: {overlap}")
        return None, None
    
    # 2. Proper temporal separation
    latest_train_date = train_data['Date'].max()
    earliest_test_date = test_data['Date'].min()
    
    logger.info(f"Temporal boundary: {latest_train_date} → {earliest_test_date}")
    
    if latest_train_date >= earliest_test_date:
        logger.error("❌ Temporal violation - Training data includes future information")
        return None, None
    
    # 3. Data completeness
    if len(test_data) == 0:
        logger.error("❌ Empty test set")
        return None, None
    
    if len(train_data) == 0:
        logger.error("❌ Empty training set")
        return None, None
    
    logger.info("✅ Temporal split validation passed")
    return train_data, test_data

def train_and_evaluate_model(train_data, test_data, model_config=None):
    """
    Train and evaluate model using definitive methodology
    """
    logger = setup_logging()
    logger.info("=== 🚀 MODEL TRAINING AND EVALUATION ===")
    
    # Features (definitive list established through testing)
    features = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized", 
        "market_entropy_norm"
    ]
    
    # Prepare data
    X_train = train_data[features].fillna(0.5)
    X_test = test_data[features].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_data['FullTimeResult'].map(label_mapping)
    y_test = test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Features: {len(features)}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Model configuration (definitive baseline established through testing)
    if model_config is None:
        model_config = {
            'n_estimators': 300,
            'max_depth': 18,      # Key parameter from breakthrough analysis
            'max_features': 'log2',
            'min_samples_leaf': 2,
            'min_samples_split': 15,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    logger.info(f"Model configuration: {model_config}")
    
    # Train model
    model = RandomForestClassifier(**model_config)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    overfitting_gap = train_accuracy - test_accuracy
    
    logger.info(f"\n📊 PERFORMANCE RESULTS:")
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Overfitting gap: {overfitting_gap*100:.1f}pp")
    
    # Detailed evaluation
    y_pred = model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Home', 'Draw', 'Away'], output_dict=True)
    
    logger.info(f"\n🎯 PERFORMANCE BREAKDOWN:")
    for outcome in ['Home', 'Draw', 'Away']:
        p = report[outcome]['precision']
        r = report[outcome]['recall'] 
        f1 = report[outcome]['f1-score']
        logger.info(f"{outcome:4s}: {p:.3f}P {r:.3f}R {f1:.3f}F1")
    
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
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'overfitting_gap': overfitting_gap,
        'y_pred': y_pred,
        'y_test': y_test,
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': dict(sorted_importance)
    }

def assess_performance_targets(test_accuracy):
    """
    Assess performance against established targets
    """
    logger = setup_logging()
    logger.info("=== 🏆 PERFORMANCE TARGET ASSESSMENT ===")
    
    targets = [
        ("Random Baseline", 0.333, "🎲"),
        ("Majority Class", 0.436, "📊"),
        ("Good Model", 0.500, "✅"),
        ("Excellent Model", 0.550, "🏆"),
        ("Industry Leading", 0.580, "🚀")
    ]
    
    achieved_targets = []
    
    for name, threshold, emoji in targets:
        achieved = test_accuracy >= threshold
        status = "✅ ACHIEVED" if achieved else "❌ MISSED"
        margin = (test_accuracy - threshold) * 100
        logger.info(f"  {emoji} {name} ({threshold:.1%}): {status} ({margin:+.1f}pp)")
        
        if achieved:
            achieved_targets.append(name)
    
    return achieved_targets

def check_performance_regression(test_accuracy):
    """
    Check for performance regression against definitive baseline
    """
    logger = setup_logging()
    logger.info("=== 📊 REGRESSION DETECTION ===")
    
    baseline = DEFINITIVE_BASELINE
    tolerance = REGRESSION_TOLERANCE
    
    difference = test_accuracy - baseline
    
    logger.info(f"Current performance: {test_accuracy:.4f}")
    logger.info(f"Definitive baseline: {baseline:.4f}")
    logger.info(f"Difference: {difference*100:+.2f}pp")
    logger.info(f"Tolerance: ±{tolerance*100:.1f}pp")
    
    if difference >= -tolerance:
        if difference > 0:
            logger.info("✅ PERFORMANCE IMPROVEMENT detected")
            return "improved"
        else:
            logger.info("✅ Performance maintained within tolerance")
            return "maintained" 
    else:
        logger.warning(f"❌ PERFORMANCE REGRESSION detected: {-difference*100:.2f}pp below baseline")
        return "regressed"

def save_validation_results(results, performance_status, achieved_targets):
    """
    Save validation results for CI/CD and monitoring
    """
    logger = setup_logging()
    
    # Prepare results summary
    validation_summary = {
        'timestamp': datetime.now().isoformat(),
        'methodology': 'definitive_clean_temporal_split',
        'baseline_version': 'v1.4_definitive',
        'test_accuracy': float(results['test_accuracy']),
        'train_accuracy': float(results['train_accuracy']),
        'overfitting_gap': float(results['overfitting_gap']),
        'definitive_baseline': DEFINITIVE_BASELINE,
        'performance_status': performance_status,
        'achieved_targets': achieved_targets,
        'feature_importance': results['feature_importance'],
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'validation_passed': performance_status != 'regressed'
    }
    
    # Save to evaluation directory
    os.makedirs('evaluation', exist_ok=True)
    
    # Save detailed results
    results_file = f"evaluation/canonical_validation_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    
    # Update latest results symlink
    latest_file = 'evaluation/latest_validation.json'
    with open(latest_file, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    
    logger.info(f"💾 Validation results saved: {results_file}")
    logger.info(f"💾 Latest results: {latest_file}")
    
    return validation_summary

def main():
    """
    Main validation pipeline
    """
    parser = argparse.ArgumentParser(description='Canonical Model Validation')
    parser.add_argument('--model-config', type=str, help='Path to custom model configuration JSON')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--regression-check', action='store_true', default=True, help='Enable regression checking')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🔬 CANONICAL MODEL VALIDATION PIPELINE")
    logger.info("=" * 80)
    logger.info("Definitive methodology established through rigorous testing")
    logger.info("Addresses 'Frankenstein test set' issue and ensures temporal integrity")
    logger.info("=" * 80)
    
    # Step 1: Load and validate data
    df = load_and_validate_data()
    if df is None:
        logger.error("❌ Data validation failed")
        return False
    
    # Step 2: Create definitive temporal split
    train_data, test_data = create_definitive_temporal_split(df)
    if train_data is None or test_data is None:
        logger.error("❌ Temporal split creation failed")
        return False
    
    # Step 3: Load custom model config if provided
    model_config = None
    if args.model_config:
        try:
            with open(args.model_config, 'r') as f:
                model_config = json.load(f)
            logger.info(f"✅ Custom model config loaded: {args.model_config}")
        except Exception as e:
            logger.error(f"❌ Failed to load model config: {e}")
            return False
    
    # Step 4: Train and evaluate model
    results = train_and_evaluate_model(train_data, test_data, model_config)
    
    # Step 5: Assess performance targets
    achieved_targets = assess_performance_targets(results['test_accuracy'])
    
    # Step 6: Check for regression
    performance_status = check_performance_regression(results['test_accuracy'])
    
    # Step 7: Save results
    validation_summary = save_validation_results(results, performance_status, achieved_targets)
    
    # Step 8: Save model if requested
    if args.save_model:
        model_file = f"models/canonical_model_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.joblib"
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(results['model'], model_file)
        logger.info(f"💾 Model saved: {model_file}")
    
    # Step 9: Final assessment
    logger.info("\n" + "=" * 80)
    logger.info("🏁 CANONICAL VALIDATION COMPLETE")
    logger.info("=" * 80)
    
    success = validation_summary['validation_passed']
    
    logger.info(f"✅ Validation Status: {'PASSED' if success else 'FAILED'}")
    logger.info(f"📊 Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']:.1%})")
    logger.info(f"📈 Performance: {performance_status.upper()}")
    logger.info(f"🎯 Targets Achieved: {len(achieved_targets)}/5")
    
    if success:
        logger.info("🎉 Model ready for deployment")
    else:
        logger.warning("⚠️ Performance regression detected - Review before deployment")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Canonical validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)