#!/usr/bin/env python3
"""
Enhanced Draw Odds v3.0 - Smart Market Integration
==================================================

Key lessons from v2.0 failure:
- Over-engineering with 4 correlated odds features hurt performance (48.2% vs 53.4% baseline)
- Market intelligence should complement, not replace, traditional features
- Simple market probability signals > complex derived features

v3.0 Strategy:
1. Clean market probability features (3 only: H/D/A implied probabilities)
2. Model vs market divergence detection
3. Two-stage classification option (Draw vs Non-Draw, then H vs A)
4. No SMOTE - use class_weight instead
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def calculate_market_probabilities(df):
    """
    Create clean market probability features from odds data
    
    Strategy: Use market consensus as a prior, then let model learn when market is wrong
    """
    logger = setup_logging()
    logger.info("=== v3.0 Clean Market Probability Features ===")
    
    # Use Bet365 as primary bookmaker (most liquid)
    required_cols = ['B365H', 'B365D', 'B365A']
    
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required odds columns: {required_cols}")
        return df
        
    logger.info("✅ Using Bet365 odds for market probabilities")
    
    # Convert odds to implied probabilities
    df['market_home_prob'] = 1 / df['B365H']
    df['market_draw_prob'] = 1 / df['B365D'] 
    df['market_away_prob'] = 1 / df['B365A']
    
    # Calculate overround (bookmaker margin)
    df['market_overround'] = df['market_home_prob'] + df['market_draw_prob'] + df['market_away_prob']
    
    # Normalize probabilities to sum to 1 (remove bookmaker margin)
    df['market_home_prob_norm'] = df['market_home_prob'] / df['market_overround']
    df['market_draw_prob_norm'] = df['market_draw_prob'] / df['market_overround']  
    df['market_away_prob_norm'] = df['market_away_prob'] / df['market_overround']
    
    # Verification
    prob_sum = df['market_home_prob_norm'] + df['market_draw_prob_norm'] + df['market_away_prob_norm']
    logger.info(f"✅ Normalized probabilities sum: {prob_sum.mean():.6f} (should be 1.000000)")
    
    logger.info("📊 Market Probability Statistics:")
    logger.info(f"  Home wins: {df['market_home_prob_norm'].mean():.3f} ± {df['market_home_prob_norm'].std():.3f}")
    logger.info(f"  Draws:     {df['market_draw_prob_norm'].mean():.3f} ± {df['market_draw_prob_norm'].std():.3f}")
    logger.info(f"  Away wins: {df['market_away_prob_norm'].mean():.3f} ± {df['market_away_prob_norm'].std():.3f}")
    
    # Clean up intermediate columns
    df = df.drop(columns=['market_home_prob', 'market_draw_prob', 'market_away_prob', 'market_overround'])
    
    return df

def create_market_divergence_features(df):
    """
    Create features that capture when our model might disagree with market
    
    These will be calculated AFTER model training using residuals
    """
    logger = setup_logging()
    logger.info("=== Market Divergence Feature Preparation ===")
    
    # For now, create placeholders - these will be filled during model training
    df['model_draw_prob'] = np.nan  # Will be filled with model predictions
    df['draw_value_signal'] = np.nan  # model_prob - market_prob
    
    # Market uncertainty indicator (simplified from v2.0)
    # High draw odds (low probability) in close matches = opportunity
    if all(col in df.columns for col in ['B365H', 'B365A', 'B365D']):
        # Close match indicator: when H and A odds are similar
        df['match_closeness'] = 1 - abs(df['B365H'] - df['B365A']) / (df['B365H'] + df['B365A'])
        
        # Draw opportunity: close match + high draw odds = potential value
        df['draw_opportunity'] = df['match_closeness'] * (df['B365D'] - 3.0) / 2.0  # Normalized
        df['draw_opportunity'] = np.clip(df['draw_opportunity'], 0, 1)
        
        logger.info(f"Draw opportunity feature: {df['draw_opportunity'].mean():.3f} ± {df['draw_opportunity'].std():.3f}")
    else:
        df['draw_opportunity'] = 0.5  # Neutral
        
    return df

def integrate_v3_features():
    """
    Create v3.0 dataset with clean market features
    """
    logger = setup_logging()
    logger.info("=== v3.0 Market Integration Pipeline ===")
    
    # Load base dataset (use v1.3 features as foundation)
    base_files = [
        "data/processed/v13_xg_safe_features.csv",
        "data/processed/premier_league_processed_v2_2024_08_30.csv"
    ]
    
    base_df = None
    for file in base_files:
        if os.path.exists(file):
            logger.info(f"Loading base dataset: {file}")
            base_df = pd.read_csv(file)
            break
            
    if base_df is None:
        logger.error("No suitable base dataset found")
        return None
        
    logger.info(f"Base dataset: {base_df.shape[0]} matches, {base_df.shape[1]} features")
    
    # Load and process odds data
    odds_files = [
        "data/raw/football_data_backup/football_data_2019_20.csv",
        "data/raw/football_data_backup/football_data_2020_21.csv",
        "data/raw/football_data_backup/football_data_2021_22.csv", 
        "data/raw/football_data_backup/football_data_2022_23.csv",
        "data/raw/football_data_backup/football_data_2023_24.csv",
        "data/raw/football_data_backup/football_data_2024_25.csv"
    ]
    
    all_odds = []
    for file in odds_files:
        if os.path.exists(file):
            df_odds = pd.read_csv(file)
            df_odds['Season'] = file.split('_')[-2] + '_' + file.split('_')[-1].replace('.csv', '')
            all_odds.append(df_odds)
    
    if not all_odds:
        logger.error("No odds files found")
        return None
        
    combined_odds = pd.concat(all_odds, ignore_index=True)
    logger.info(f"Loaded odds data: {combined_odds.shape[0]} matches")
    
    # Create clean market features
    enhanced_odds = calculate_market_probabilities(combined_odds)
    enhanced_odds = create_market_divergence_features(enhanced_odds)
    
    # Merge with base data
    if 'HomeTeam' in base_df.columns and 'AwayTeam' in base_df.columns:
        # Create merge keys
        enhanced_odds['merge_key'] = enhanced_odds['HomeTeam'] + "_vs_" + enhanced_odds['AwayTeam']
        base_df['merge_key'] = base_df['HomeTeam'] + "_vs_" + base_df['AwayTeam']
        
        # Select v3.0 features for merging
        v3_features = [
            'merge_key', 
            'market_home_prob_norm', 'market_draw_prob_norm', 'market_away_prob_norm',
            'draw_opportunity'
        ]
        
        merge_data = enhanced_odds[v3_features].drop_duplicates(subset=['merge_key'])
        
        # Merge
        final_df = base_df.merge(merge_data, on='merge_key', how='left')
        final_df = final_df.drop(columns=['merge_key'])
        
        # Fill missing values with neutral probabilities
        final_df['market_home_prob_norm'] = final_df['market_home_prob_norm'].fillna(0.436)  # Historical average
        final_df['market_draw_prob_norm'] = final_df['market_draw_prob_norm'].fillna(0.230)
        final_df['market_away_prob_norm'] = final_df['market_away_prob_norm'].fillna(0.334)
        final_df['draw_opportunity'] = final_df['draw_opportunity'].fillna(0.5)
        
        logger.info(f"Final dataset: {final_df.shape[0]} matches, {final_df.shape[1]} features")
        logger.info("✅ v3.0 features added: 4 clean market features")
        
        # Save v3.0 dataset
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output_file = f"data/processed/premier_league_market_v3_{timestamp}.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved v3.0 dataset: {output_file}")
        
        return output_file
    else:
        logger.error("Cannot merge - missing team columns")
        return None

if __name__ == "__main__":
    output_file = integrate_v3_features()
    if output_file:
        print(f"✅ v3.0 Clean Market Features Created: {output_file}")
        print("\n🎯 Key v3.0 Improvements:")
        print("  • Clean market probabilities (H/D/A) - no over-engineering")
        print("  • Draw opportunity signal for close matches")
        print("  • Ready for model vs market divergence detection")
        print("  • No SMOTE - will use class_weight='balanced'")
    else:
        print("❌ Failed to create v3.0 features")