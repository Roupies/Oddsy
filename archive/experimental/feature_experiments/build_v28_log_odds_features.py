#!/usr/bin/env python3
"""
v2.8 Log Odds Features Builder
Implements advanced market intelligence features based on thesis recommendations.
Focus: Log odds ratios for improved draw prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_log_odds_features(df):
    """Calculate advanced log odds ratio features for draw prediction."""
    
    # Convert market probabilities to odds (odds = 1/probability)
    df['odds_home'] = 1 / df['market_home_prob_norm']
    df['odds_draw'] = 1 / df['market_draw_prob_norm'] 
    df['odds_away'] = 1 / df['market_away_prob_norm']
    
    # Core log odds ratios (thesis recommendation)
    df['log_draw_home_ratio'] = np.log(df['odds_draw'] / df['odds_home'])
    df['log_draw_away_ratio'] = np.log(df['odds_draw'] / df['odds_away'])
    
    # Market balance indicator
    df['market_balance_score'] = np.abs(df['log_draw_home_ratio'] - df['log_draw_away_ratio'])
    
    # Draw favorability (positive = draw favored vs both outcomes)
    df['draw_favorability'] = (df['log_draw_home_ratio'] + df['log_draw_away_ratio']) / 2
    
    # Market efficiency gap (actual vs expected draw probability)
    actual_draw_rate = 0.23  # EPL historical draw rate
    df['market_efficiency_gap'] = df['market_draw_prob_norm'] - actual_draw_rate
    
    # Normalized versions for ML
    features_to_normalize = [
        'log_draw_home_ratio', 'log_draw_away_ratio', 'market_balance_score',
        'draw_favorability', 'market_efficiency_gap'
    ]
    
    for feature in features_to_normalize:
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[f'{feature}_normalized'] = (df[feature] - min_val) / (max_val - min_val)
    
    logger.info(f"Created {len(features_to_normalize)} log odds features")
    return df

def build_v28_dataset():
    """Build v2.8 dataset with log odds features."""
    
    # Load existing v2.4 baseline dataset
    input_path = Path('data/processed/premier_league_market_v3_2025_09_02_105923.csv')
    output_path = Path('data/processed/v28_log_odds_features_2025_09_06.csv')
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info(f"Input dataset shape: {df.shape}")
    
    # Calculate log odds features
    df = calculate_log_odds_features(df)
    
    # Verify no infinite/NaN values
    log_odds_features = [col for col in df.columns if 'log_' in col or 'market_balance' in col 
                        or 'draw_favorability' in col or 'market_efficiency' in col]
    
    for feature in log_odds_features:
        if df[feature].isna().any() or np.isinf(df[feature]).any():
            logger.warning(f"Found NaN/Inf values in {feature}, cleaning...")
            df[feature] = df[feature].fillna(df[feature].median())
            df[feature] = df[feature].replace([np.inf, -np.inf], [df[feature].max(), df[feature].min()])
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved v2.8 dataset to {output_path}")
    logger.info(f"Output dataset shape: {df.shape}")
    
    # Feature importance analysis
    print("\n🎯 New Log Odds Features:")
    for feature in log_odds_features:
        print(f"  • {feature}: mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")
    
    return df

if __name__ == "__main__":
    logger.info("🚀 Starting v2.8 Log Odds Features build...")
    df = build_v28_dataset()
    logger.info("✅ v2.8 Log Odds Features build complete!")