#!/usr/bin/env python3
"""
Enhanced Draw Odds Features v2.0
================================

Create advanced odds-based features specifically designed to improve draw prediction.
Building on the successful market_entropy_norm from v1.3, this adds 4 new features
that focus on market inefficiencies and draw-specific signals.

v2.0 Target: 55%+ accuracy by exploiting draw mispricings in betting markets.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def calculate_implied_probability(odds):
    """Convert decimal odds to implied probability"""
    return 1 / odds if odds > 0 else 0

def calculate_draw_value_features(df):
    """
    Create 4 advanced draw-specific features:
    1. draw_value_indicator - Market underpricing of draws
    2. bookmaker_draw_disagreement - Variance in draw odds across bookmakers  
    3. draw_odds_movement - Movement patterns (requires temporal data)
    4. market_draw_bias - Systematic market bias against draws
    """
    logger = setup_logging()
    logger.info("=== Creating Enhanced Draw Features v2.0 ===")
    
    # Available bookmaker columns for draw odds
    draw_odds_cols = ['B365D', 'BWD', 'IWD', 'PSD', 'WHD', 'VCD']
    available_cols = [col for col in draw_odds_cols if col in df.columns]
    
    if len(available_cols) < 3:
        logger.error(f"Insufficient draw odds columns. Found: {available_cols}")
        return df
        
    logger.info(f"Using draw odds from: {available_cols}")
    
    # ===== FEATURE 1: Draw Value Indicator =====
    # When market consensus underprices draws vs fair probability
    
    # Calculate implied probabilities for each bookmaker
    for col in available_cols:
        prob_col = col.replace('D', '_draw_prob')
        df[prob_col] = df[col].apply(calculate_implied_probability)
    
    # Market consensus draw probability (average across bookmakers)
    prob_cols = [col.replace('D', '_draw_prob') for col in available_cols]
    df['market_draw_consensus'] = df[prob_cols].mean(axis=1)
    
    # Fair draw probability estimate (based on historical Premier League: ~23%)
    # But adjust based on match context (strong vs weak teams = lower draw prob)
    df['fair_draw_estimate'] = 0.23  # Base rate
    
    # Adjust fair estimate based on team strength difference
    # If we have odds for home/away, we can infer strength difference
    if 'B365H' in df.columns and 'B365A' in df.columns:
        df['strength_ratio'] = df['B365H'] / df['B365A']  
        # More even teams (ratio close to 1) = higher draw probability
        df['strength_balance'] = 1 - abs(df['strength_ratio'] - 1) / (df['strength_ratio'] + 1)
        df['fair_draw_estimate'] = 0.15 + 0.16 * df['strength_balance']  # 15-31% range
    
    # Draw value = fair probability - market consensus
    df['draw_value_raw'] = df['fair_draw_estimate'] - df['market_draw_consensus']
    df['draw_value_indicator'] = np.clip((df['draw_value_raw'] + 0.1) / 0.2, 0, 1)
    
    # ===== FEATURE 2: Bookmaker Draw Disagreement =====
    # High variance in draw odds suggests uncertainty -> opportunity
    
    df['draw_odds_std'] = df[available_cols].std(axis=1)
    df['draw_odds_mean'] = df[available_cols].mean(axis=1)
    df['draw_disagreement_raw'] = df['draw_odds_std'] / df['draw_odds_mean']  # Coefficient of variation
    
    # Normalize to 0-1 (higher = more disagreement)
    disagreement_99th = df['draw_disagreement_raw'].quantile(0.99)
    df['bookmaker_draw_disagreement'] = np.clip(df['draw_disagreement_raw'] / disagreement_99th, 0, 1)
    
    # ===== FEATURE 3: Market Draw Bias =====  
    # Systematic market tendency to under/over-price draws
    
    # Compare draw odds to "expected" based on home/away odds
    if 'B365H' in df.columns and 'B365A' in df.columns:
        # Simple model: if H and A odds suggest even match, draw should be ~3.5-4.0
        df['expected_draw_odds'] = 2 + 2 * abs(df['B365H'] - df['B365A']) / (df['B365H'] + df['B365A'])
        df['draw_bias_raw'] = df['B365D'] / df['expected_draw_odds']  # >1 = overpriced, <1 = underpriced
        
        # Convert to 0-1 indicator (0 = overpriced, 1 = underpriced)
        df['market_draw_bias'] = np.clip((2 - df['draw_bias_raw']) / 2, 0, 1)
    else:
        df['market_draw_bias'] = 0.5  # Neutral if we can't calculate
    
    # ===== FEATURE 4: Draw Odds Stability =====
    # Replace movement (needs temporal data) with stability indicator
    
    # Use the consistency of draw odds across bookmakers as proxy for market confidence  
    df['draw_odds_range'] = df[available_cols].max(axis=1) - df[available_cols].min(axis=1)
    df['draw_stability_raw'] = df['draw_odds_range'] / df['draw_odds_mean']
    
    # Normalize (lower range = more stable = lower opportunity)
    stability_99th = df['draw_stability_raw'].quantile(0.99) 
    df['draw_odds_instability'] = np.clip(df['draw_stability_raw'] / stability_99th, 0, 1)
    
    # ===== Summary Statistics =====
    
    logger.info("📊 Enhanced Draw Features Created:")
    logger.info(f"  draw_value_indicator: {df['draw_value_indicator'].mean():.3f} ± {df['draw_value_indicator'].std():.3f}")
    logger.info(f"  bookmaker_draw_disagreement: {df['bookmaker_draw_disagreement'].mean():.3f} ± {df['bookmaker_draw_disagreement'].std():.3f}")  
    logger.info(f"  market_draw_bias: {df['market_draw_bias'].mean():.3f} ± {df['market_draw_bias'].std():.3f}")
    logger.info(f"  draw_odds_instability: {df['draw_odds_instability'].mean():.3f} ± {df['draw_odds_instability'].std():.3f}")
    
    # Clean up intermediate columns
    cleanup_cols = prob_cols + ['market_draw_consensus', 'fair_draw_estimate', 'draw_value_raw',
                               'draw_odds_std', 'draw_odds_mean', 'draw_disagreement_raw',
                               'expected_draw_odds', 'draw_bias_raw', 'draw_odds_range', 'draw_stability_raw']
    
    if 'strength_ratio' in df.columns:
        cleanup_cols.extend(['strength_ratio', 'strength_balance'])
        
    df = df.drop(columns=[col for col in cleanup_cols if col in df.columns])
    
    return df

def integrate_with_existing_data():
    """
    Load existing processed data and add enhanced draw features
    """
    logger = setup_logging()
    logger.info("=== v2.0 Enhanced Draw Odds Integration ===")
    
    # Load the latest processed dataset (should have market_entropy_norm)
    processed_files = [
        "data/processed/v13_xg_safe_features.csv",  # Most recent clean dataset
        "data/processed/premier_league_xg_v21_2025_08_31_232209.csv",
        "data/processed/premier_league_processed_v2_2024_08_30.csv"
    ]
    
    base_df = None
    for file in processed_files:
        if os.path.exists(file):
            logger.info(f"Loading base dataset: {file}")
            base_df = pd.read_csv(file)
            break
    
    if base_df is None:
        logger.error("No suitable base dataset found")
        return None
        
    logger.info(f"Base dataset: {base_df.shape[0]} matches, {base_df.shape[1]} features")
    
    # Load odds data and merge
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
    
    # Create enhanced draw features
    enhanced_odds = calculate_draw_value_features(combined_odds)
    
    # Merge with base data on HomeTeam, AwayTeam, and Date/Season
    # This is tricky - need to match on team names and dates
    
    # For now, create a simple merge key
    if 'HomeTeam' in base_df.columns and 'AwayTeam' in base_df.columns:
        enhanced_odds['merge_key'] = enhanced_odds['HomeTeam'] + "_vs_" + enhanced_odds['AwayTeam']
        base_df['merge_key'] = base_df['HomeTeam'] + "_vs_" + base_df['AwayTeam'] 
        
        # Select only the new features for merging
        new_features = ['merge_key', 'draw_value_indicator', 'bookmaker_draw_disagreement', 
                       'market_draw_bias', 'draw_odds_instability']
        
        merge_data = enhanced_odds[new_features].drop_duplicates(subset=['merge_key'])
        
        # Merge
        final_df = base_df.merge(merge_data, on='merge_key', how='left')
        final_df = final_df.drop(columns=['merge_key'])
        
        # Fill missing values with neutral values
        for feature in ['draw_value_indicator', 'bookmaker_draw_disagreement', 
                       'market_draw_bias', 'draw_odds_instability']:
            final_df[feature] = final_df[feature].fillna(0.5)
        
        logger.info(f"Final dataset: {final_df.shape[0]} matches, {final_df.shape[1]} features")
        logger.info(f"New features added: {len(new_features)-1}")
        
        # Save enhanced dataset
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output_file = f"data/processed/premier_league_enhanced_draws_v2_{timestamp}.csv"
        final_df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved enhanced dataset: {output_file}")
        
        return output_file
    
    else:
        logger.error("Cannot merge - missing team columns")
        return None

if __name__ == "__main__":
    output_file = integrate_with_existing_data()
    if output_file:
        print(f"✅ v2.0 Enhanced Draw Features Created: {output_file}")
        print("\n🎯 Ready for v2.0 model training with draw-optimized features!")
    else:
        print("❌ Failed to create enhanced features")