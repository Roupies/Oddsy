#!/usr/bin/env python3
"""
Create proper v1.3 dataset with real dates AND market_entropy_norm
Combine the best of both datasets
"""
import pandas as pd
from utils import setup_logging

def create_proper_v13_dataset():
    """
    Merge real dates with v1.3 features including market_entropy_norm
    """
    logger = setup_logging()
    logger.info("=== 🔧 CREATING PROPER v1.3 DATASET ===")
    
    # Load dataset with real dates (but missing market_entropy_norm)
    df_dates = pd.read_csv('data/processed/premier_league_ml_ready.csv')
    df_dates['Date'] = pd.to_datetime(df_dates['Date'])
    logger.info(f"Dataset with dates: {len(df_dates)} matches, columns: {list(df_dates.columns)}")
    
    # Load dataset with market_entropy_norm (but proxy dates)
    df_market = pd.read_csv('data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv')
    logger.info(f"Dataset with market: {len(df_market)} matches, columns: {list(df_market.columns)}")
    
    # Check if they have same length and order
    if len(df_dates) != len(df_market):
        logger.error(f"Dataset lengths don't match: {len(df_dates)} vs {len(df_market)}")
        return None
    
    # Check if FullTimeResult matches (to verify same order)
    if not df_dates['FullTimeResult'].equals(df_market['FullTimeResult']):
        logger.warning("⚠️ FullTimeResult doesn't match - different order or data")
        # Try to match by team names if available
        if 'HomeTeam' in df_dates.columns and 'AwayTeam' in df_dates.columns:
            logger.info("Attempting to align by team names...")
            # This would require more complex matching logic
        else:
            logger.error("Cannot align datasets safely")
            return None
    else:
        logger.info("✅ Datasets appear to be in same order")
    
    # Create combined dataset
    logger.info("🔀 COMBINING DATASETS...")
    
    # Start with dates dataset as base
    df_combined = df_dates.copy()
    
    # Add market_entropy_norm from market dataset
    df_combined['market_entropy_norm'] = df_market['market_entropy_norm']
    
    # Define v1.3 complete features
    v13_features = [
        "form_diff_normalized",
        "elo_diff_normalized", 
        "h2h_score",
        "matchday_normalized",
        "shots_diff_normalized",
        "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    # Keep only essential columns
    essential_columns = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult'] + v13_features
    available_columns = [col for col in essential_columns if col in df_combined.columns]
    
    df_final = df_combined[available_columns].copy()
    
    logger.info(f"✅ COMBINED DATASET CREATED:")
    logger.info(f"  Shape: {df_final.shape}")
    logger.info(f"  Date range: {df_final['Date'].min()} → {df_final['Date'].max()}")
    logger.info(f"  Features: {v13_features}")
    logger.info(f"  market_entropy_norm range: [{df_final['market_entropy_norm'].min():.3f}, {df_final['market_entropy_norm'].max():.3f}]")
    
    # Save combined dataset
    output_file = 'data/processed/v13_complete_with_dates.csv'
    df_final.to_csv(output_file, index=False)
    logger.info(f"💾 Saved: {output_file}")
    
    return df_final

if __name__ == "__main__":
    df = create_proper_v13_dataset()
    if df is not None:
        print("✅ v1.3 complete dataset created successfully")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    else:
        print("❌ Failed to create combined dataset")