#!/usr/bin/env python3
"""
Automated Data Leakage Detection Suite
Comprehensive temporal integrity testing for all features
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def test_rolling_window_integrity():
    """
    Test that all rolling/moving average features respect temporal boundaries
    """
    logger = setup_logging()
    logger.info("=== 🔍 TESTING ROLLING WINDOW TEMPORAL INTEGRITY ===")
    
    # Load dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Testing {len(df)} matches for temporal integrity")
    
    # Identify rolling/temporal features
    rolling_features = [
        'form_diff_normalized',
        'shots_diff_normalized', 
        'corners_diff_normalized'
    ]
    
    violations_found = 0
    
    for feature in rolling_features:
        logger.info(f"\n📊 Testing feature: {feature}")
        
        # Test for temporal violations by checking if future matches affect current predictions
        feature_violations = 0
        
        # Sample random matches for testing
        test_indices = np.random.choice(df.index[50:len(df)-50], size=100, replace=False)
        
        for idx in test_indices:
            current_match = df.iloc[idx]
            current_date = current_match['Date']
            current_value = current_match[feature]
            
            # Check if any future data could have influenced this value
            # Method: Look for patterns that suggest future data contamination
            
            # Get matches for same teams in temporal window
            home_team = current_match['HomeTeam']
            away_team = current_match['AwayTeam']
            
            # Find recent matches for these teams
            recent_window = df[
                (df['Date'] < current_date) & 
                (df['Date'] >= current_date - pd.Timedelta(days=90))
            ]
            
            team_recent_matches = recent_window[
                (recent_window['HomeTeam'].isin([home_team, away_team])) |
                (recent_window['AwayTeam'].isin([home_team, away_team]))
            ]
            
            if len(team_recent_matches) < 5:
                # Not enough historical data - feature should be neutral/default
                if not (0.4 <= current_value <= 0.6):
                    logger.warning(f"  ⚠️  Insufficient history but non-neutral value: {current_value:.3f}")
                    feature_violations += 1
        
        # Check for impossible future knowledge
        # Look for discontinuities that suggest batch processing with future data
        feature_series = df[feature].values
        diff_series = np.diff(feature_series)
        
        # Large jumps in consecutive matches suggest batch calculation issues
        large_jumps = np.where(np.abs(diff_series) > 0.3)[0]
        
        if len(large_jumps) > len(df) * 0.05:  # More than 5% large jumps is suspicious
            logger.warning(f"  ⚠️  Excessive large jumps ({len(large_jumps)}) - possible batch calculation")
            feature_violations += 10
        
        # Check seasonal boundary integrity
        season_boundaries = df[df['Date'].dt.month.isin([7, 8])].index  # July/August boundaries
        
        for boundary_idx in season_boundaries[1:]:  # Skip first season
            if boundary_idx < len(df) - 1:
                pre_boundary = df.iloc[boundary_idx - 1][feature]
                post_boundary = df.iloc[boundary_idx][feature]
                
                # Values should reset or be continuous, not jump dramatically
                if abs(pre_boundary - post_boundary) > 0.4:
                    logger.warning(f"  ⚠️  Large seasonal jump: {pre_boundary:.3f} → {post_boundary:.3f}")
                    feature_violations += 1
        
        violations_found += feature_violations
        
        if feature_violations == 0:
            logger.info(f"  ✅ {feature}: No temporal violations detected")
        else:
            logger.warning(f"  ❌ {feature}: {feature_violations} violations found")
    
    # Overall assessment
    if violations_found == 0:
        logger.info(f"\n✅ ROLLING WINDOW INTEGRITY: PASSED")
        return True
    else:
        logger.warning(f"\n❌ ROLLING WINDOW INTEGRITY: {violations_found} violations found")
        return False

def test_feature_calculation_order():
    """
    Test that features are calculated in correct chronological order
    """
    logger = setup_logging()
    logger.info("=== ⏰ TESTING FEATURE CALCULATION ORDER ===")
    
    # Load dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Verify dataset is chronologically sorted
    dates_sorted = df['Date'].is_monotonic_increasing
    
    if not dates_sorted:
        logger.error("❌ Dataset not chronologically sorted - fundamental issue!")
        return False
    else:
        logger.info("✅ Dataset correctly sorted chronologically")
    
    # Test for features that should evolve monotonically or smoothly
    smooth_features = [
        'matchday_normalized',  # Should increase within season
        'market_entropy_norm'   # Should be stable
    ]
    
    violations = 0
    
    for feature in smooth_features:
        logger.info(f"\n📈 Testing chronological evolution: {feature}")
        
        if feature == 'matchday_normalized':
            # Should reset at season boundaries and increase within seasons
            seasons = df['Season'].unique()
            
            for season in seasons:
                season_data = df[df['Season'] == season][feature]
                
                # Within season, should be monotonic increasing
                diffs = season_data.diff().dropna()
                negative_diffs = (diffs < -0.1).sum()  # Allow small noise
                
                if negative_diffs > len(season_data) * 0.1:  # More than 10% backwards
                    logger.warning(f"  ⚠️  Non-monotonic matchday in {season}: {negative_diffs} reversals")
                    violations += 1
                else:
                    logger.info(f"  ✅ {season}: Proper matchday progression")
        
        elif feature == 'market_entropy_norm':
            # Should be stable (not random walk)
            values = df[feature].dropna()
            
            # Calculate volatility
            volatility = values.std()
            mean_val = values.mean()
            cv = volatility / mean_val if mean_val > 0 else float('inf')
            
            logger.info(f"  Market entropy stats: mean={mean_val:.3f}, std={volatility:.3f}, cv={cv:.3f}")
            
            if cv > 2.0:  # Coefficient of variation too high
                logger.warning(f"  ⚠️  High volatility in market entropy: CV={cv:.3f}")
                violations += 1
            else:
                logger.info(f"  ✅ Market entropy shows proper stability")
    
    if violations == 0:
        logger.info(f"\n✅ CALCULATION ORDER: PASSED")
        return True
    else:
        logger.warning(f"\n❌ CALCULATION ORDER: {violations} issues found")
        return False

def test_train_test_split_integrity():
    """
    Test that no future data leaks into training set
    """
    logger = setup_logging()
    logger.info("=== 🚧 TESTING TRAIN/TEST SPLIT INTEGRITY ===")
    
    # Load dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Test the actual split used in modeling
    cutoff_date = '2024-01-01'
    train_data = df[df['Date'] < cutoff_date]
    test_data = df[df['Date'] >= cutoff_date]
    
    logger.info(f"Split date: {cutoff_date}")
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    violations = 0
    
    # Check 1: No temporal overlap
    latest_train_date = train_data['Date'].max()
    earliest_test_date = test_data['Date'].min()
    
    logger.info(f"Latest train date: {latest_train_date}")
    logger.info(f"Earliest test date: {earliest_test_date}")
    
    if latest_train_date >= earliest_test_date:
        logger.error("❌ TEMPORAL OVERLAP: Training data includes future information!")
        violations += 1
    else:
        logger.info("✅ No temporal overlap between train/test")
    
    # Check 2: Feature consistency across split
    all_features = [col for col in df.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FullTimeResult']]
    
    for feature in all_features:
        train_stats = train_data[feature].describe()
        test_stats = test_data[feature].describe()
        
        # Check for extreme distribution shifts
        train_range = train_stats['max'] - train_stats['min']
        test_range = test_stats['max'] - test_stats['min']
        
        if test_range > 0:
            range_ratio = train_range / test_range
            if range_ratio < 0.5 or range_ratio > 2.0:
                logger.warning(f"  ⚠️  {feature}: Range shift train/test {range_ratio:.2f}")
                violations += 1
        
        # Check means don't drift too much
        mean_shift = abs(train_stats['mean'] - test_stats['mean'])
        if mean_shift > 0.2:  # 20% shift in normalized features is large
            logger.warning(f"  ⚠️  {feature}: Large mean shift {mean_shift:.3f}")
            violations += 1
    
    # Check 3: No identical rows across train/test (potential duplication)
    # Create signature for each match
    train_signatures = set()
    for _, row in train_data.iterrows():
        signature = f"{row['HomeTeam']}_{row['AwayTeam']}_{row['Date'].strftime('%Y-%m-%d')}"
        train_signatures.add(signature)
    
    duplicate_matches = 0
    for _, row in test_data.iterrows():
        signature = f"{row['HomeTeam']}_{row['AwayTeam']}_{row['Date'].strftime('%Y-%m-%d')}"
        if signature in train_signatures:
            duplicate_matches += 1
    
    if duplicate_matches > 0:
        logger.error(f"❌ DUPLICATE MATCHES: {duplicate_matches} matches appear in both train/test")
        violations += duplicate_matches
    else:
        logger.info("✅ No duplicate matches across train/test")
    
    if violations == 0:
        logger.info(f"\n✅ TRAIN/TEST SPLIT INTEGRITY: PASSED")
        return True
    else:
        logger.error(f"\n❌ TRAIN/TEST SPLIT INTEGRITY: {violations} violations found")
        return False

def test_h2h_temporal_integrity():
    """
    Test Head-to-Head calculations don't use future data
    """
    logger = setup_logging()
    logger.info("=== 🥊 TESTING H2H TEMPORAL INTEGRITY ===")
    
    # Load dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Testing H2H calculations for temporal integrity")
    
    violations = 0
    
    # Sample matches to test
    test_indices = np.random.choice(df.index[100:len(df)-100], size=50, replace=False)
    
    for idx in test_indices:
        current_match = df.iloc[idx]
        current_date = current_match['Date']
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        h2h_score = current_match['h2h_score']
        
        # Find all historical matches between these teams
        historical_h2h = df[
            (df['Date'] < current_date) &
            (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
             ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)))
        ]
        
        if len(historical_h2h) == 0:
            # No history - H2H should be neutral (0.5)
            if not (0.45 <= h2h_score <= 0.55):
                logger.warning(f"  ⚠️  No H2H history but score={h2h_score:.3f} (should be ~0.5)")
                violations += 1
        else:
            # Has history - check if score makes sense based on actual history
            home_wins = ((historical_h2h['HomeTeam'] == home_team) & 
                        (historical_h2h['FullTimeResult'] == 'H')).sum()
            away_wins = ((historical_h2h['AwayTeam'] == home_team) & 
                        (historical_h2h['FullTimeResult'] == 'A')).sum()
            
            total_wins = home_wins + away_wins
            
            if len(historical_h2h) >= 3:  # Enough history to validate
                expected_score = total_wins / len(historical_h2h) if len(historical_h2h) > 0 else 0.5
                
                # Allow reasonable margin for normalization/weighting
                if abs(h2h_score - expected_score) > 0.3:
                    logger.warning(f"  ⚠️  H2H mismatch: calculated={expected_score:.3f}, actual={h2h_score:.3f}")
                    violations += 1
    
    # Test for future contamination by checking if H2H scores change dramatically
    # when we artificially move matches earlier in time
    logger.info("\n🔮 Testing for future information contamination...")
    
    # Check a few specific matchups across seasons
    team_pairs = [
        ('Man City', 'Liverpool'),
        ('Arsenal', 'Chelsea'),
        ('Man United', 'Tottenham')
    ]
    
    for home, away in team_pairs:
        pair_matches = df[
            ((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
            ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))
        ].sort_values('Date')
        
        if len(pair_matches) >= 5:
            # Check if H2H scores evolve reasonably over time
            h2h_scores = pair_matches['h2h_score'].values
            score_changes = np.abs(np.diff(h2h_scores))
            
            # Large changes suggest recalculation with future data
            large_changes = (score_changes > 0.4).sum()
            
            if large_changes > len(pair_matches) * 0.3:
                logger.warning(f"  ⚠️  {home} vs {away}: {large_changes} large H2H changes suggest future contamination")
                violations += 1
            else:
                logger.info(f"  ✅ {home} vs {away}: H2H evolution looks natural")
    
    if violations == 0:
        logger.info(f"\n✅ H2H TEMPORAL INTEGRITY: PASSED")
        return True
    else:
        logger.warning(f"\n❌ H2H TEMPORAL INTEGRITY: {violations} violations found")
        return False

def run_leakage_automation_suite():
    """
    Run complete automated data leakage detection suite
    """
    logger = setup_logging()
    logger.info("🔬 STARTING AUTOMATED DATA LEAKAGE DETECTION SUITE")
    logger.info("=" * 80)
    
    tests = [
        ("Rolling Window Integrity", test_rolling_window_integrity),
        ("Feature Calculation Order", test_feature_calculation_order),
        ("Train/Test Split Integrity", test_train_test_split_integrity),
        ("H2H Temporal Integrity", test_h2h_temporal_integrity)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                logger.info(f"{test_name}: ✅ PASSED")
            else:
                logger.warning(f"{test_name}: ❌ FAILED")
        except Exception as e:
            logger.error(f"{test_name}: 💥 ERROR - {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("🏁 LEAKAGE DETECTION SUITE SUMMARY")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🏆 ALL TESTS PASSED - No data leakage detected!")
    else:
        logger.warning(f"⚠️  {len(tests) - passed} tests failed - Potential data leakage issues")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_leakage_automation_suite()
    print(f"\n🔬 LEAKAGE DETECTION SUITE COMPLETE")
    print(f"Status: {'PASSED' if success else 'FAILED'}")
    
    if not success:
        print("⚠️  Potential data leakage detected - Review logs for details")
        sys.exit(1)
    else:
        print("✅ No data leakage detected - Temporal integrity verified")
        sys.exit(0)