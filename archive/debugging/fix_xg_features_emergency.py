#!/usr/bin/env python3
"""
EMERGENCY FIX: xG Features Data Leakage
======================================

Critical fix for xG efficiency features that have impossible values (>11.0).
This emergency patch corrects the calculation bugs that compromise model integrity.

Issues found:
- home_xg_eff_10 max = 11.119 (impossible - teams can't score 11x their xG over 10 games)
- Division by near-zero xG values creating infinite/extreme ratios
- No minimum threshold causing mathematical instabilities

Solutions implemented:
1. Recalculate xG efficiency with minimum xG threshold (0.5 over 10 games)
2. Apply realistic bounds [0.3, 3.0] for efficiency 
3. Handle edge cases properly
4. Validate all xG-related features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')
from utils import setup_logging

def load_corrupted_dataset():
    """Load the dataset with corrupted xG features"""
    logger = setup_logging()
    logger.info("=== LOADING CORRUPTED DATASET FOR EMERGENCY FIX ===")
    
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv', parse_dates=['Date'])
    logger.info(f"📊 Original dataset: {df.shape[0]} matches, {df.shape[1]} features")
    
    # Analyze corruption extent
    xg_eff_cols = ['home_xg_eff_10', 'away_xg_eff_10']
    
    for col in xg_eff_cols:
        if col in df.columns:
            col_stats = df[col].describe()
            extreme_values = (df[col] > 3.0).sum()
            impossible_values = (df[col] > 5.0).sum()
            
            logger.info(f"🔍 {col}:")
            logger.info(f"  Max value: {df[col].max():.3f}")
            logger.info(f"  Values > 3.0: {extreme_values} ({extreme_values/len(df)*100:.1f}%)")  
            logger.info(f"  Values > 5.0 (impossible): {impossible_values}")
    
    return df

def fix_xg_efficiency_calculation(df):
    """
    Emergency fix for xG efficiency calculation
    """
    logger = setup_logging()
    logger.info("=== EMERGENCY FIX: xG EFFICIENCY CALCULATION ===")
    
    # Available columns analysis
    xg_related_cols = [col for col in df.columns if 'xg' in col.lower()]
    logger.info(f"📊 Available xG columns: {len(xg_related_cols)}")
    for col in xg_related_cols:
        logger.info(f"  {col}")
    
    # Method 1: Apply strict bounds to existing efficiency features
    logger.info("🔧 Method 1: Applying safety bounds to existing features")
    
    efficiency_cols = ['home_xg_eff_10', 'away_xg_eff_10']
    
    for col in efficiency_cols:
        if col in df.columns:
            original_min = df[col].min()
            original_max = df[col].max()
            
            # Apply realistic bounds: efficiency between 0.3 and 3.0
            df[col] = df[col].clip(lower=0.3, upper=3.0)
            
            new_min = df[col].min()
            new_max = df[col].max()
            capped_values = ((df[col] == 0.3) | (df[col] == 3.0)).sum()
            
            logger.info(f"✅ {col}: [{original_min:.3f}, {original_max:.3f}] → [{new_min:.3f}, {new_max:.3f}]")
            logger.info(f"   Capped values: {capped_values} ({capped_values/len(df)*100:.1f}%)")
    
    # Method 2: Recalculate using available sum columns if present
    logger.info("🔧 Method 2: Attempting recalculation from sum columns")
    
    # Check if we have the raw data for recalculation
    sum_cols_available = []
    for side in ['home', 'away']:
        goals_col = f"{side}_goals_sum_10"
        xg_col = f"{side}_xg_sum_10"
        if goals_col in df.columns and xg_col in df.columns:
            sum_cols_available.append((side, goals_col, xg_col))
    
    if sum_cols_available:
        logger.info(f"✅ Found raw data columns for recalculation: {len(sum_cols_available)} sides")
        
        for side, goals_col, xg_col in sum_cols_available:
            eff_col = f"{side}_xg_eff_10"
            
            # Recalculate with safety measures
            logger.info(f"🔄 Recalculating {eff_col}")
            
            # Method: efficiency = goals / max(xG, 0.5)
            # This prevents division by near-zero values
            df[eff_col + '_recalc'] = df.apply(
                lambda row: row[goals_col] / max(row[xg_col], 0.5) if pd.notna(row[xg_col]) and pd.notna(row[goals_col]) else 0.95,
                axis=1
            )
            
            # Apply bounds
            df[eff_col + '_recalc'] = df[eff_col + '_recalc'].clip(0.3, 3.0)
            
            # Replace original column
            df[eff_col] = df[eff_col + '_recalc']
            df = df.drop(columns=[eff_col + '_recalc'])
            
            logger.info(f"✅ {eff_col} recalculated: range [{df[eff_col].min():.3f}, {df[eff_col].max():.3f}]")
    else:
        logger.warning("⚠️ No sum columns available for recalculation, using bounds only")
    
    return df

def validate_fixed_features(df):
    """
    Validate that all xG features are now within acceptable ranges
    """
    logger = setup_logging()
    logger.info("=== VALIDATING FIXED xG FEATURES ===")
    
    # Define acceptable ranges for all xG-related features
    feature_ranges = {
        'home_xg_eff_10': (0.25, 3.5),      # Slightly wider for validation
        'away_xg_eff_10': (0.25, 3.5),
        'home_xg_roll_5': (0, 5),           # Rolling averages should be reasonable
        'home_xg_roll_10': (0, 5),
        'away_xg_roll_5': (0, 5), 
        'away_xg_roll_10': (0, 5),
        'home_goals_sum_5': (0, 25),        # Maximum 5 goals per game * 5 games
        'home_goals_sum_10': (0, 50),       # Maximum 5 goals per game * 10 games
        'away_goals_sum_5': (0, 25),
        'away_goals_sum_10': (0, 50)
    }
    
    validation_passed = True
    issues_found = []
    
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in df.columns:
            out_of_range = df[(df[feature] < min_val) | (df[feature] > max_val)]
            
            if not out_of_range.empty:
                validation_passed = False
                issues_found.append({
                    'feature': feature,
                    'expected_range': (min_val, max_val),
                    'actual_range': (df[feature].min(), df[feature].max()),
                    'violations': len(out_of_range)
                })
                logger.error(f"❌ {feature}: {len(out_of_range)} values outside [{min_val}, {max_val}]")
                logger.error(f"   Actual range: [{df[feature].min():.3f}, {df[feature].max():.3f}]")
            else:
                logger.info(f"✅ {feature}: all values in [{min_val}, {max_val}]")
    
    if validation_passed:
        logger.info("🎉 ALL xG FEATURES VALIDATION: PASSED")
    else:
        logger.error(f"❌ VALIDATION FAILED: {len(issues_found)} features have issues")
    
    return validation_passed, issues_found

def save_corrected_dataset(df):
    """
    Save the corrected dataset with proper versioning
    """
    logger = setup_logging()
    logger.info("=== SAVING CORRECTED DATASET ===")
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = f"data/processed/v13_xg_corrected_features_{timestamp}.csv"
    
    # Save corrected dataset
    df.to_csv(output_file, index=False)
    logger.info(f"💾 Corrected dataset saved: {output_file}")
    
    # Create a symbolic link to latest
    latest_file = "data/processed/v13_xg_corrected_features_latest.csv"
    if os.path.exists(latest_file):
        os.remove(latest_file)
    
    # Create relative symlink
    os.symlink(os.path.basename(output_file), latest_file)
    logger.info(f"🔗 Latest link created: {latest_file}")
    
    # Log correction summary
    xg_eff_cols = ['home_xg_eff_10', 'away_xg_eff_10']
    logger.info("📊 CORRECTION SUMMARY:")
    for col in xg_eff_cols:
        if col in df.columns:
            logger.info(f"  {col}: range [{df[col].min():.3f}, {df[col].max():.3f}]")
            logger.info(f"    Mean: {df[col].mean():.3f}, Std: {df[col].std():.3f}")
    
    return output_file

def emergency_fix_pipeline():
    """
    Complete emergency fix pipeline
    """
    logger = setup_logging()
    logger.info("🚨 STARTING EMERGENCY xG FEATURES FIX")
    logger.info("="*70)
    
    # Step 1: Load corrupted dataset
    df = load_corrupted_dataset()
    
    # Step 2: Apply fixes
    df_fixed = fix_xg_efficiency_calculation(df)
    
    # Step 3: Validate fixes
    validation_passed, issues = validate_fixed_features(df_fixed)
    
    if not validation_passed:
        logger.error("❌ EMERGENCY FIX FAILED - validation issues remain")
        return None
    
    # Step 4: Save corrected dataset
    corrected_file = save_corrected_dataset(df_fixed)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("🎉 EMERGENCY FIX COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"✅ Corrected dataset: {corrected_file}")
    logger.info("✅ All xG features now within realistic ranges")
    logger.info("✅ Ready for model retraining with clean features")
    logger.info("="*70)
    
    return corrected_file

if __name__ == "__main__":
    corrected_file = emergency_fix_pipeline()
    
    if corrected_file:
        print(f"\n🎉 EMERGENCY FIX SUCCESS!")
        print(f"Corrected dataset: {corrected_file}")
        print("Next step: Retrain the 56% model with corrected features.")
    else:
        print("\n❌ EMERGENCY FIX FAILED!")
        print("Manual intervention required to resolve remaining issues.")