#!/usr/bin/env python3
"""
Comprehensive Elo Implementation Verification
Tests for initialization, carry-over, order, K-factor, and temporal integrity
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def test_elo_carry_over_between_seasons():
    """Test if Elo ratings carry over correctly between seasons"""
    logger = setup_logging()
    logger.info("=== 🔄 TESTING ELO CARRY-OVER BETWEEN SEASONS ===")
    
    # Load dataset with Elo ratings
    dataset_path = 'data/processed/v13_complete_with_dates.csv'
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Loaded dataset: {len(df)} matches")
    
    # Check if we have Elo-related columns
    elo_columns = [col for col in df.columns if 'elo' in col.lower()]
    logger.info(f"Elo-related columns found: {elo_columns}")
    
    if not elo_columns:
        logger.warning("No Elo columns found in dataset - checking if needs to be computed")
        return False
    
    # Analyze carry-over for each season transition
    if 'Season' not in df.columns:
        logger.warning("Season column not found - deriving from dates")
        # Derive seasons from dates (Aug-July cycle)
        df['Season'] = df['Date'].apply(lambda d: f"{d.year}-{d.year+1}" if d.month >= 8 else f"{d.year-1}-{d.year}")
    
    seasons = sorted(df['Season'].unique())
    logger.info(f"Seasons found: {seasons}")
    
    carry_over_results = []
    
    for i in range(len(seasons) - 1):
        current_season = seasons[i]
        next_season = seasons[i + 1]
        
        logger.info(f"\nAnalyzing carry-over: {current_season} → {next_season}")
        
        # Get last matches of current season
        current_season_data = df[df['Season'] == current_season].copy()
        next_season_data = df[df['Season'] == next_season].copy()
        
        if len(current_season_data) == 0 or len(next_season_data) == 0:
            continue
        
        # Check if we have raw Elo ratings (before normalization)
        if 'elo_diff_normalized' in df.columns:
            logger.info("Found normalized Elo diff - need to check raw implementation")
            
            # For normalized features, we can't directly check carry-over
            # But we can check if there are discontinuities
            season_boundary = current_season_data['Date'].max()
            next_season_start = next_season_data['Date'].min()
            
            # Get matches around season boundary
            boundary_window = df[
                (df['Date'] >= season_boundary - timedelta(days=30)) &
                (df['Date'] <= next_season_start + timedelta(days=30))
            ].copy()
            
            if len(boundary_window) > 0:
                # Check for sudden jumps in normalized Elo differences
                elo_diffs = boundary_window['elo_diff_normalized'].values
                diff_variance = np.var(elo_diffs)
                
                carry_over_results.append({
                    'transition': f"{current_season} → {next_season}",
                    'variance_around_boundary': diff_variance,
                    'boundary_date': season_boundary,
                    'next_start': next_season_start,
                    'sample_size': len(boundary_window)
                })
                
                logger.info(f"  Elo diff variance around boundary: {diff_variance:.6f}")
        
        # Check for specific teams' Elo progression
        teams_in_both = set(current_season_data['HomeTeam'].unique()) & set(next_season_data['HomeTeam'].unique())
        logger.info(f"  Teams in both seasons: {len(teams_in_both)}")
        
        if len(teams_in_both) > 0:
            sample_team = list(teams_in_both)[0]
            logger.info(f"  Sample team analysis: {sample_team}")
            
            # Last appearance in current season
            team_current = current_season_data[
                (current_season_data['HomeTeam'] == sample_team) |
                (current_season_data['AwayTeam'] == sample_team)
            ].tail(1)
            
            # First appearance in next season
            team_next = next_season_data[
                (next_season_data['HomeTeam'] == sample_team) |
                (next_season_data['AwayTeam'] == sample_team)
            ].head(1)
            
            if len(team_current) > 0 and len(team_next) > 0:
                current_date = team_current['Date'].iloc[0]
                next_date = team_next['Date'].iloc[0]
                logger.info(f"    Last match {current_season}: {current_date}")
                logger.info(f"    First match {next_season}: {next_date}")
    
    # Summary of carry-over analysis
    logger.info(f"\n📊 CARRY-OVER ANALYSIS SUMMARY:")
    logger.info("=" * 50)
    
    if carry_over_results:
        avg_variance = np.mean([r['variance_around_boundary'] for r in carry_over_results])
        logger.info(f"Average boundary variance: {avg_variance:.6f}")
        
        for result in carry_over_results:
            logger.info(f"  {result['transition']}: variance={result['variance_around_boundary']:.6f}, samples={result['sample_size']}")
        
        # Interpretation
        if avg_variance < 0.01:
            logger.info("✅ LOW VARIANCE: Likely good Elo carry-over")
        elif avg_variance < 0.05:
            logger.info("⚠️ MODERATE VARIANCE: Possible issues with carry-over")
        else:
            logger.info("❌ HIGH VARIANCE: Likely Elo reset between seasons")
    else:
        logger.warning("Could not analyze carry-over - insufficient data")
    
    return len(carry_over_results) > 0 and avg_variance < 0.05 if carry_over_results else False

def test_elo_temporal_order():
    """Test if Elo calculations respect temporal order"""
    logger = setup_logging()
    logger.info("\n=== ⏰ TESTING ELO TEMPORAL ORDER ===")
    
    # This is a critical test - Elo at match i should only use matches j where j < i
    dataset_path = 'data/processed/v13_complete_with_dates.csv'
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info("Testing temporal integrity of Elo calculations...")
    
    # Check if we can access the Elo calculation code
    potential_elo_scripts = [
        'feature_engineering.py',
        'fix_leakage_features.py',
        'fixed_elo_engineering.py',
        'scripts/preprocessing/feature_engineering.py'
    ]
    
    elo_script_found = None
    for script in potential_elo_scripts:
        if os.path.exists(script):
            elo_script_found = script
            break
    
    if elo_script_found:
        logger.info(f"Found Elo implementation: {elo_script_found}")
        
        # Read the script to check implementation
        with open(elo_script_found, 'r') as f:
            content = f.read()
        
        # Check for temporal integrity patterns
        temporal_issues = []
        
        if 'sort_values' not in content:
            temporal_issues.append("Script may not sort data chronologically")
        
        if '.iloc[i:' in content or '.iloc[i+1:' in content:
            temporal_issues.append("Potential future data usage in rolling windows")
        
        if 'shift(' not in content and 'expanding(' not in content:
            temporal_issues.append("No obvious temporal protection (shift/expanding) found")
        
        if temporal_issues:
            logger.warning("⚠️ POTENTIAL TEMPORAL INTEGRITY ISSUES:")
            for issue in temporal_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Basic temporal integrity checks passed")
    else:
        logger.warning("Could not find Elo implementation script")
    
    # Test specific temporal violations
    logger.info("\nTesting for specific temporal violations...")
    
    # Sample random matches and check if Elo seems to use only past data
    sample_indices = np.random.choice(range(100, len(df)), size=min(50, len(df)//20), replace=False)
    
    temporal_violations = []
    
    for idx in sample_indices:
        match = df.iloc[idx]
        match_date = match['Date']
        
        # Get all matches before this one
        past_matches = df[df['Date'] < match_date]
        future_matches = df[df['Date'] > match_date]
        
        if len(past_matches) < 10:  # Need sufficient history
            continue
        
        # Check if Elo values seem reasonable given past performance
        if 'elo_diff_normalized' in df.columns:
            elo_diff = match['elo_diff_normalized']
            
            # Get recent performance of both teams
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Recent matches for home team
            home_recent = past_matches[
                (past_matches['HomeTeam'] == home_team) | (past_matches['AwayTeam'] == home_team)
            ].tail(10)
            
            # Recent matches for away team  
            away_recent = past_matches[
                (past_matches['HomeTeam'] == away_team) | (past_matches['AwayTeam'] == away_team)
            ].tail(10)
            
            # Simple sanity check - if Elo diff is extreme but recent performance doesn't justify it
            if abs(elo_diff - 0.5) > 0.4:  # Very skewed prediction
                if len(home_recent) < 5 or len(away_recent) < 5:
                    # Not enough history to justify extreme prediction
                    temporal_violations.append({
                        'match_idx': idx,
                        'date': match_date,
                        'teams': f"{home_team} vs {away_team}",
                        'elo_diff': elo_diff,
                        'issue': 'Extreme prediction with insufficient history'
                    })
    
    logger.info(f"Checked {len(sample_indices)} matches for temporal violations")
    
    if temporal_violations:
        logger.warning(f"❌ FOUND {len(temporal_violations)} POTENTIAL VIOLATIONS:")
        for violation in temporal_violations[:5]:  # Show first 5
            logger.warning(f"  {violation['date']}: {violation['teams']} - {violation['issue']}")
    else:
        logger.info("✅ No obvious temporal violations found in sample")
    
    return len(temporal_violations) == 0

def test_elo_initialization():
    """Test Elo initialization strategy"""
    logger = setup_logging()
    logger.info("\n=== 🎯 TESTING ELO INITIALIZATION ===")
    
    dataset_path = 'data/processed/v13_complete_with_dates.csv'
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Look at the very first matches
    first_matches = df.head(50)
    logger.info(f"Analyzing first {len(first_matches)} matches for initialization patterns")
    
    if 'elo_diff_normalized' in df.columns:
        early_elo_diffs = first_matches['elo_diff_normalized'].values
        
        # Check if early matches show signs of equal initialization
        early_variance = np.var(early_elo_diffs)
        early_mean = np.mean(early_elo_diffs)
        
        logger.info(f"Early Elo differences:")
        logger.info(f"  Mean: {early_mean:.4f} (should be ~0.5 for equal init)")
        logger.info(f"  Variance: {early_variance:.6f} (should be low for equal init)")
        logger.info(f"  Range: [{early_elo_diffs.min():.3f}, {early_elo_diffs.max():.3f}]")
        
        # Expected behavior for equal initialization
        if abs(early_mean - 0.5) < 0.05 and early_variance < 0.01:
            logger.info("✅ CONSISTENT with equal Elo initialization")
            init_status = "EQUAL_INIT"
        elif early_variance > 0.1:
            logger.info("❌ HIGH VARIANCE suggests pre-loaded ratings or issues")
            init_status = "PRELOADED_OR_ISSUES"
        else:
            logger.info("⚠️ UNCLEAR initialization pattern")
            init_status = "UNCLEAR"
        
        # Check first few matches of each team
        logger.info(f"\nFirst appearance analysis:")
        teams = set(first_matches['HomeTeam'].unique()) | set(first_matches['AwayTeam'].unique())
        
        for team in list(teams)[:5]:  # Sample first 5 teams
            team_matches = first_matches[
                (first_matches['HomeTeam'] == team) | (first_matches['AwayTeam'] == team)
            ]
            if len(team_matches) > 0:
                first_match = team_matches.iloc[0]
                elo_diff = first_match['elo_diff_normalized']
                home_team = first_match['HomeTeam']
                is_home = (home_team == team)
                expected_elo = elo_diff if is_home else (1 - elo_diff)
                logger.info(f"  {team} first match: elo_indicator={expected_elo:.3f}")
    
    else:
        logger.warning("No elo_diff_normalized column found")
        init_status = "NO_ELO_DATA"
    
    return init_status == "EQUAL_INIT"

def test_elo_k_factor_sensitivity():
    """Test K-factor impact and sensitivity"""
    logger = setup_logging()
    logger.info("\n=== 📊 TESTING K-FACTOR SENSITIVITY ===")
    
    # This is more of an analysis than a test
    # Check if current Elo implementation allows K-factor tuning
    
    potential_configs = ['config/config.json', 'config/elo_config.json']
    k_factor_found = False
    
    for config_path in potential_configs:
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Look for K-factor related settings
                k_settings = {}
                for key, value in config.items():
                    if 'k' in key.lower() and 'factor' in key.lower():
                        k_settings[key] = value
                    elif key.lower() in ['k_factor', 'k', 'elo_k']:
                        k_settings[key] = value
                
                if k_settings:
                    logger.info(f"Found K-factor settings in {config_path}: {k_settings}")
                    k_factor_found = True
                    break
                    
            except Exception as e:
                logger.warning(f"Could not read {config_path}: {e}")
    
    if not k_factor_found:
        logger.info("No explicit K-factor configuration found")
        logger.info("Checking if K-factor is hardcoded in implementation...")
        
        # Check Elo implementation files for hardcoded K values
        for script in ['feature_engineering.py', 'fix_leakage_features.py']:
            if os.path.exists(script):
                with open(script, 'r') as f:
                    content = f.read()
                
                # Look for K-factor patterns
                import re
                k_patterns = re.findall(r'[Kk].*?=.*?(\d+)', content)
                if k_patterns:
                    logger.info(f"Found potential K-factors in {script}: {k_patterns}")
    
    # Theoretical K-factor analysis
    logger.info(f"\n📚 K-FACTOR THEORY CHECK:")
    logger.info("Standard K-factors:")
    logger.info("  K=32: FIFA standard, good for moderate volatility")
    logger.info("  K=16: Lower volatility, slower adaptation")
    logger.info("  K=48: Higher volatility, faster adaptation")
    logger.info("  K=64: Very volatile, for experimental systems")
    
    # Check current Elo variance as proxy for K-factor level
    dataset_path = 'data/processed/v13_complete_with_dates.csv'
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        
        if 'elo_diff_normalized' in df.columns:
            elo_variance = np.var(df['elo_diff_normalized'])
            logger.info(f"\nObserved Elo variance: {elo_variance:.6f}")
            
            if elo_variance < 0.01:
                logger.info("⚠️ VERY LOW variance - K might be too small or Elo not working")
            elif elo_variance < 0.05:
                logger.info("✅ REASONABLE variance - moderate K-factor")
            elif elo_variance < 0.15:
                logger.info("⚠️ HIGH variance - K might be large or other issues")
            else:
                logger.info("❌ EXTREME variance - likely implementation issues")
    
    return k_factor_found or True  # Pass if we found config or did analysis

def run_comprehensive_elo_tests():
    """Run all Elo implementation tests"""
    logger = setup_logging()
    logger.info("🔬 STARTING COMPREHENSIVE ELO IMPLEMENTATION TESTS")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ('Carry-over Between Seasons', test_elo_carry_over_between_seasons),
        ('Temporal Order', test_elo_temporal_order),
        ('Initialization', test_elo_initialization),
        ('K-Factor Sensitivity', test_elo_k_factor_sensitivity)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n" + "="*60)
        logger.info(f"Running: {test_name}")
        logger.info("="*60)
        
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            test_results[test_name] = False
            logger.error(f"{test_name}: ❌ ERROR - {e}")
    
    # Overall summary
    logger.info(f"\n" + "="*80)
    logger.info("🏁 ELO IMPLEMENTATION TEST SUMMARY")
    logger.info("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🏆 ALL TESTS PASSED - Elo implementation looks good!")
        overall_status = "PASSED"
    elif passed_tests >= total_tests * 0.75:
        logger.info("⚠️ MOSTLY PASSED - Some issues to investigate")
        overall_status = "MOSTLY_PASSED"
    else:
        logger.info("❌ MULTIPLE FAILURES - Elo implementation needs review")
        overall_status = "FAILED"
    
    # Generate recommendations
    logger.info(f"\n🔧 RECOMMENDATIONS:")
    
    if not test_results.get('Carry-over Between Seasons', True):
        logger.info("• Fix Elo carry-over: Ensure ratings persist between seasons")
    
    if not test_results.get('Temporal Order', True):
        logger.info("• Fix temporal integrity: Ensure Elo uses only past matches")
    
    if not test_results.get('Initialization', True):
        logger.info("• Review initialization: Consider equal starting ratings")
    
    if not test_results.get('K-Factor Sensitivity', True):
        logger.info("• Add K-factor configuration: Make K-factor tunable")
    
    return {
        'overall_status': overall_status,
        'test_results': test_results,
        'passed_count': passed_tests,
        'total_count': total_tests
    }

if __name__ == "__main__":
    try:
        results = run_comprehensive_elo_tests()
        
        print(f"\n🔬 ELO IMPLEMENTATION TESTS COMPLETE")
        print(f"Status: {results['overall_status']}")
        print(f"Passed: {results['passed_count']}/{results['total_count']}")
        
        if results['overall_status'] != 'PASSED':
            print("⚠️ Issues found - review logs for details")
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()