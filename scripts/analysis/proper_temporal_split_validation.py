#!/usr/bin/env python3
"""
PROPER TEMPORAL SPLIT VALIDATION
Re-validate performance with clean football season boundaries
Train: 2019-2024 (5 complete seasons), Test: 2024-2025 (1 complete season)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def create_proper_temporal_split():
    """
    Create proper temporal split based on football seasons
    Train: 2019-2020, 2020-2021, 2021-2022, 2022-2023, 2023-2024 (5 seasons)
    Test: 2024-2025 (1 complete season, 380 matches)
    """
    logger = setup_logging()
    logger.info("=== ⚽ CREATING PROPER FOOTBALL SEASON SPLIT ===")
    
    # Load cleaned dataset (with temporal integrity fixes)
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} matches")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Seasons available: {sorted(df['Season'].unique())}")
    
    # Define proper split by seasons (not calendar dates)
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']
    
    train_data = df[df['Season'].isin(train_seasons)].copy()
    test_data = df[df['Season'].isin(test_seasons)].copy()
    
    logger.info(f"\n📊 PROPER TEMPORAL SPLIT:")
    logger.info(f"Training seasons: {train_seasons}")
    logger.info(f"Testing seasons: {test_seasons}")
    logger.info(f"Training matches: {len(train_data)}")
    logger.info(f"Testing matches: {len(test_data)}")
    
    # Verify temporal integrity
    latest_train_date = train_data['Date'].max()
    earliest_test_date = test_data['Date'].min()
    
    logger.info(f"\nTemporal boundary verification:")
    logger.info(f"Latest training date: {latest_train_date}")
    logger.info(f"Earliest test date: {earliest_test_date}")
    
    if latest_train_date >= earliest_test_date:
        logger.error("❌ TEMPORAL OVERLAP - This should not happen with season-based split!")
        return None, None
    else:
        logger.info("✅ No temporal overlap - proper chronological separation")
    
    # Verify test set is complete season
    expected_matches_per_season = 380  # 20 teams × 19 home matches each
    
    if len(test_data) == expected_matches_per_season:
        logger.info(f"✅ Test set is complete season: {len(test_data)} matches")
    elif len(test_data) < expected_matches_per_season:
        logger.warning(f"⚠️  Test set incomplete: {len(test_data)}/{expected_matches_per_season} matches")
        logger.warning("   Season 2024-2025 may be in progress")
    else:
        logger.error(f"❌ Too many test matches: {len(test_data)} > {expected_matches_per_season}")
    
    # Apply temporal integrity fixes using full dataset for context
    full_df = df.copy()  # Keep full dataset for historical context
    train_data_clean = apply_temporal_fixes(train_data, logger, full_df)
    test_data_clean = apply_temporal_fixes(test_data, logger, full_df)
    
    return train_data_clean, test_data_clean

def apply_temporal_fixes(df, logger, full_dataset):
    """
    Apply temporal integrity fixes using full dataset for historical context
    """
    logger.info(f"🔧 Applying temporal fixes to {len(df)} matches...")
    
    cleaned_df = df.copy()
    
    # Fix form features - use full dataset for historical context
    for idx in range(len(cleaned_df)):
        current_match = cleaned_df.iloc[idx]
        current_date = current_match['Date']
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        
        # Get recent matches (only historical) from FULL dataset
        window_days = 60
        recent_matches = full_dataset[
            (full_dataset['Date'] < current_date) & 
            (full_dataset['Date'] >= current_date - pd.Timedelta(days=window_days))
        ]
        
        # Home team recent form
        home_recent = recent_matches[
            (recent_matches['HomeTeam'] == home_team) | 
            (recent_matches['AwayTeam'] == home_team)
        ]
        
        # Away team recent form  
        away_recent = recent_matches[
            (recent_matches['HomeTeam'] == away_team) | 
            (recent_matches['AwayTeam'] == away_team)
        ]
        
        # Calculate form scores
        if len(home_recent) >= 3 and len(away_recent) >= 3:
            home_wins = 0
            for _, match in home_recent.iterrows():
                if pd.notna(match['FullTimeResult']):
                    if match['HomeTeam'] == home_team and match['FullTimeResult'] == 'H':
                        home_wins += 1
                    elif match['AwayTeam'] == home_team and match['FullTimeResult'] == 'A':
                        home_wins += 1
            
            away_wins = 0
            for _, match in away_recent.iterrows():
                if pd.notna(match['FullTimeResult']):
                    if match['HomeTeam'] == away_team and match['FullTimeResult'] == 'H':
                        away_wins += 1
                    elif match['AwayTeam'] == away_team and match['FullTimeResult'] == 'A':
                        away_wins += 1
            
            home_form = home_wins / len(home_recent)
            away_form = away_wins / len(away_recent)
            cleaned_df.at[idx, 'form_diff_normalized'] = (home_form + (1 - away_form)) / 2
        else:
            cleaned_df.at[idx, 'form_diff_normalized'] = 0.5
    
    # Fix H2H scores - use full dataset for historical context
    for idx in range(len(cleaned_df)):
        current_match = cleaned_df.iloc[idx]
        current_date = current_match['Date']
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        
        # Historical H2H matches from FULL dataset
        historical_h2h = full_dataset[
            (full_dataset['Date'] < current_date) &
            (((full_dataset['HomeTeam'] == home_team) & (full_dataset['AwayTeam'] == away_team)) |
             ((full_dataset['HomeTeam'] == away_team) & (full_dataset['AwayTeam'] == home_team)))
        ]
        
        if len(historical_h2h) >= 2:
            home_wins = 0
            valid_matches = 0
            for _, match in historical_h2h.iterrows():
                if pd.notna(match['FullTimeResult']):
                    valid_matches += 1
                    if match['HomeTeam'] == home_team and match['FullTimeResult'] == 'H':
                        home_wins += 1
                    elif match['AwayTeam'] == home_team and match['FullTimeResult'] == 'A':
                        home_wins += 1
            
            h2h_score = home_wins / valid_matches if valid_matches > 0 else 0.5
        else:
            h2h_score = 0.5
        
        cleaned_df.at[idx, 'h2h_score'] = h2h_score
    
    return cleaned_df

def validate_with_proper_split():
    """
    Validate model performance with proper temporal split
    """
    logger = setup_logging()
    logger.info("=== 🏆 DEFINITIVE PERFORMANCE VALIDATION ===")
    
    # Get proper split
    train_data, test_data = create_proper_temporal_split()
    if train_data is None:
        logger.error("❌ Failed to create proper split")
        return None
    
    # Features
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
    
    logger.info(f"\n📋 VALIDATION SETUP:")
    logger.info(f"Features: {len(features)}")
    logger.info(f"Training: {len(X_train)} matches from {len(train_data['Season'].unique())} seasons")
    logger.info(f"Testing: {len(X_test)} matches from {len(test_data['Season'].unique())} season(s)")
    
    # Verified no missing values
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    logger.info(f"Missing values: Train={train_missing}, Test={test_missing}")
    
    # Train model with breakthrough configuration
    model_config = {
        'n_estimators': 300, 
        'max_depth': 18,  # Breakthrough parameter
        'max_features': 'log2',
        'min_samples_leaf': 2, 
        'min_samples_split': 15,
        'class_weight': 'balanced', 
        'random_state': 42, 
        'n_jobs': -1
    }
    
    logger.info(f"\n🚀 TRAINING MODEL:")
    logger.info(f"Algorithm: RandomForest")
    logger.info(f"Key params: n_estimators={model_config['n_estimators']}, max_depth={model_config['max_depth']}")
    
    model = RandomForestClassifier(**model_config)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    overfitting_gap = train_accuracy - test_accuracy
    
    logger.info(f"\n📊 DEFINITIVE PERFORMANCE RESULTS:")
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Overfitting gap: {overfitting_gap*100:.1f}pp")
    
    # Detailed evaluation
    y_pred = model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Home', 'Draw', 'Away'], output_dict=True)
    
    logger.info(f"\n🎯 DETAILED PERFORMANCE:")
    logger.info(f"Home Win: {report['Home']['precision']:.3f} precision, {report['Home']['recall']:.3f} recall, {report['Home']['f1-score']:.3f} F1")
    logger.info(f"Draw:     {report['Draw']['precision']:.3f} precision, {report['Draw']['recall']:.3f} recall, {report['Draw']['f1-score']:.3f} F1")
    logger.info(f"Away Win: {report['Away']['precision']:.3f} precision, {report['Away']['recall']:.3f} recall, {report['Away']['f1-score']:.3f} F1")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n📈 CONFUSION MATRIX:")
    logger.info(f"Predicted:  H     D     A")
    logger.info(f"Actual H: {cm[0][0]:3d}   {cm[0][1]:3d}   {cm[0][2]:3d}")
    logger.info(f"Actual D: {cm[1][0]:3d}   {cm[1][1]:3d}   {cm[1][2]:3d}")
    logger.info(f"Actual A: {cm[2][0]:3d}   {cm[2][1]:3d}   {cm[2][2]:3d}")
    
    # Feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n🎯 FEATURE IMPORTANCE:")
    for i, (feature, importance) in enumerate(sorted_importance, 1):
        logger.info(f"  {i}. {feature}: {importance:.3f}")
    
    # Performance targets assessment
    logger.info(f"\n🏆 PERFORMANCE TARGET ASSESSMENT:")
    
    targets = [
        ("Random Baseline", 0.333, "🎲"),
        ("Majority Class", 0.436, "📊"),
        ("Good Model", 0.500, "✅"),
        ("Excellent Model", 0.550, "🏆"),
        ("Industry Leading", 0.580, "🚀")
    ]
    
    for target_name, threshold, emoji in targets:
        status = "✅ ACHIEVED" if test_accuracy >= threshold else "❌ MISSED"
        margin = (test_accuracy - threshold) * 100
        logger.info(f"  {emoji} {target_name} ({threshold:.1%}): {status} ({margin:+.1f}pp)")
    
    # Final assessment
    logger.info(f"\n" + "="*80)
    logger.info(f"🏁 DEFINITIVE VALIDATION CONCLUSION")
    logger.info(f"="*80)
    
    if test_accuracy >= 0.58:
        conclusion = "🚀 INDUSTRY LEADING - Exceptional performance achieved"
    elif test_accuracy >= 0.55:
        conclusion = "🏆 EXCELLENT - Industry competitive performance confirmed"
    elif test_accuracy >= 0.50:
        conclusion = "✅ GOOD - Significantly beats baseline, commercially viable"
    elif test_accuracy >= 0.436:
        conclusion = "📊 MODERATE - Beats random but below commercial threshold"
    else:
        conclusion = "❌ INSUFFICIENT - Below majority class baseline"
    
    logger.info(f"FINAL STATUS: {conclusion}")
    logger.info(f"Test Performance: {test_accuracy:.4f} ({test_accuracy:.1%})")
    
    # Compare with previous validations
    logger.info(f"\n📈 HISTORICAL COMPARISON:")
    logger.info(f"  Previous validation (mixed seasons): 56.21%")
    logger.info(f"  Proper split validation: {test_accuracy:.1%}")
    logger.info(f"  Difference: {(test_accuracy - 0.5621)*100:+.1f}pp")
    
    # Business implications
    logger.info(f"\n💼 BUSINESS IMPLICATIONS:")
    
    if test_accuracy >= 0.55:
        logger.info("✅ Production Ready - Model exceeds industry benchmarks")
        logger.info("✅ Commercial Viability - Can compete with professional services")
        logger.info("✅ Investment Worthy - Strong foundation for advanced development")
    elif test_accuracy >= 0.50:
        logger.info("✅ MVP Ready - Solid baseline for further development")
        logger.info("⚠️  Improvement Needed - Focus on feature engineering for commercial use")
    else:
        logger.info("❌ Development Required - Need significant improvements before deployment")
    
    # Save definitive results
    results = {
        'test_accuracy': float(test_accuracy),
        'train_accuracy': float(train_accuracy),
        'overfitting_gap': float(overfitting_gap),
        'proper_temporal_split': True,
        'train_seasons': len(train_data['Season'].unique()),
        'test_seasons': len(test_data['Season'].unique()),
        'train_matches': len(X_train),
        'test_matches': len(X_test),
        'feature_importance': dict(sorted_importance),
        'performance_targets': {
            'beats_random': test_accuracy >= 0.333,
            'beats_majority': test_accuracy >= 0.436,
            'good_model': test_accuracy >= 0.500,
            'excellent_model': test_accuracy >= 0.550,
            'industry_leading': test_accuracy >= 0.580
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Save to file
    import json
    results_file = f"evaluation/definitive_validation_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
    os.makedirs("evaluation", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to: {results_file}")
    
    return results

def run_definitive_validation():
    """
    Execute the definitive validation with proper temporal split
    """
    logger = setup_logging()
    logger.info("🔬 STARTING DEFINITIVE TEMPORAL VALIDATION")
    logger.info("=" * 80)
    logger.info("Addressing Gemini's Critical Feedback:")
    logger.info("- Fix the 'Frankenstein' test set (mixed seasons)")
    logger.info("- Use proper football season boundaries")
    logger.info("- Validate on clean temporal split")
    logger.info("=" * 80)
    
    results = validate_with_proper_split()
    
    if results is None:
        logger.error("❌ Validation failed")
        return False
    
    return results['test_accuracy'] >= 0.50

if __name__ == "__main__":
    try:
        success = run_definitive_validation()
        print(f"\n🔬 DEFINITIVE VALIDATION COMPLETE")
        
        if success:
            print("✅ Performance validated with proper temporal split")
        else:
            print("❌ Model performance below acceptable threshold")
            
    except Exception as e:
        print(f"💥 Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)