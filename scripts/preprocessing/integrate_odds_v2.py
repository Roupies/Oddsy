import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def integrate_odds_with_baseline():
    """
    Integrate odds features with v1.2 baseline features.
    Create v2.0 dataset: 8 baseline features + 3 core odds features
    
    Phase 2: Start simple with core 3 features, target 53.0% accuracy
    """
    
    logger = setup_logging()
    logger.info("=== v2.0 Odds Integration with v1.2 Baseline ===")
    
    # Load v1.2 baseline ML-ready dataset
    baseline_file = "data/processed/premier_league_ml_ready.csv"
    logger.info(f"Loading v1.2 baseline: {baseline_file}")
    
    df_baseline = pd.read_csv(baseline_file)
    logger.info(f"Baseline loaded: {df_baseline.shape}")
    
    # Load extracted odds features
    odds_file = "data/processed/odds_features_extracted_2025_08_30_201650.csv"
    logger.info(f"Loading odds features: {odds_file}")
    
    df_odds = pd.read_csv(odds_file)
    logger.info(f"Odds loaded: {df_odds.shape}")
    
    # Convert dates for matching
    df_odds['Date'] = pd.to_datetime(df_odds['Date'])
    
    # Create matching keys
    logger.info("Creating match keys for alignment...")
    
    # For baseline dataset, we need to recreate Date from existing data
    # The baseline dataset should have Date info - let me check structure first
    logger.info("Baseline columns:")
    logger.info(f"  {list(df_baseline.columns)}")
    
    # Convert baseline Date to datetime
    df_baseline['Date'] = pd.to_datetime(df_baseline['Date'])
    
    # Create match keys
    df_baseline['match_key'] = (df_baseline['Date'].dt.strftime('%Y-%m-%d') + '_' + 
                               df_baseline['HomeTeam'] + '_' + df_baseline['AwayTeam'])
    
    df_odds['match_key'] = (df_odds['Date'].dt.strftime('%Y-%m-%d') + '_' + 
                           df_odds['HomeTeam'] + '_' + df_odds['AwayTeam'])
    
    logger.info("Match key examples:")
    logger.info(f"  Baseline: {df_baseline['match_key'].head(3).tolist()}")
    logger.info(f"  Odds: {df_odds['match_key'].head(3).tolist()}")
    
    # Merge datasets
    logger.info("Merging baseline and odds data...")
    
    # Select core odds features for Phase 2 (start simple)
    core_odds_features = [
        'market_home_advantage_norm',  # Core market consensus
        'market_entropy_norm',         # Market uncertainty
        'pinnacle_home_advantage_norm' # Sharp money indicator
    ]
    
    # Select odds columns for merge
    odds_merge_cols = ['match_key'] + core_odds_features
    df_odds_merge = df_odds[odds_merge_cols].copy()
    
    # Merge on match key
    df_merged = df_baseline.merge(df_odds_merge, on='match_key', how='left')
    logger.info(f"Merged dataset: {df_merged.shape}")
    
    # Check merge quality
    missing_odds = df_merged[core_odds_features].isnull().any(axis=1).sum()
    total_matches = len(df_merged)
    missing_pct = (missing_odds / total_matches) * 100
    
    logger.info(f"Merge quality:")
    logger.info(f"  Total matches: {total_matches}")
    logger.info(f"  Missing odds: {missing_odds} ({missing_pct:.1f}%)")
    
    if missing_pct > 5:
        logger.warning(f"High missing rate: {missing_pct:.1f}% > 5% threshold")
    else:
        logger.info(f"✅ Missing rate {missing_pct:.1f}% < 5% threshold")
    
    # Handle missing odds (fill with neutral values)
    for feature in core_odds_features:
        df_merged[feature] = df_merged[feature].fillna(0.5)  # Neutral
        df_merged[feature] = np.clip(df_merged[feature], 0, 1)  # Ensure [0,1]
    
    # Define v2.0 feature set
    v12_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized'
    ]
    
    v20_features = v12_features + core_odds_features
    
    logger.info(f"v1.2 baseline features ({len(v12_features)}): {v12_features}")
    logger.info(f"v2.0 odds features ({len(core_odds_features)}): {core_odds_features}")
    logger.info(f"v2.0 total features ({len(v20_features)}): {len(v20_features)}")
    
    # Verify all features exist
    missing_features = [f for f in v20_features if f not in df_merged.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        return None
    
    # Create final ML-ready dataset
    ml_columns = v20_features + ['FullTimeResult']
    df_v20 = df_merged[ml_columns].copy()
    
    # Data quality validation
    logger.info("Final dataset validation:")
    logger.info(f"  Shape: {df_v20.shape}")
    logger.info(f"  Missing values: {df_v20.isnull().sum().sum()}")
    logger.info(f"  Feature ranges:")
    
    for feature in core_odds_features:
        min_val = df_v20[feature].min()
        max_val = df_v20[feature].max()
        mean_val = df_v20[feature].mean()
        logger.info(f"    {feature}: [{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}")
    
    # Target distribution
    target_dist = df_v20['FullTimeResult'].value_counts(normalize=True).sort_index()
    logger.info(f"Target distribution:")
    for outcome, pct in target_dist.items():
        logger.info(f"  {outcome}: {pct:.3f}")
    
    # Save v2.0 dataset
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = f"data/processed/premier_league_ml_ready_v20_{timestamp}.csv"
    
    df_v20.to_csv(output_file, index=False)
    logger.info(f"v2.0 dataset saved: {output_file}")
    
    # Save feature configuration
    feature_config = {
        "version": "v2.0",
        "timestamp": timestamp,
        "baseline_features": v12_features,
        "odds_features": core_odds_features,
        "total_features": v20_features,
        "feature_count": {
            "baseline": len(v12_features),
            "odds": len(core_odds_features), 
            "total": len(v20_features)
        },
        "merge_quality": {
            "total_matches": int(total_matches),
            "missing_odds": int(missing_odds),
            "missing_percentage": float(missing_pct)
        }
    }
    
    import json
    config_file = f"config/features_v20_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    logger.info(f"Feature configuration saved: {config_file}")
    logger.info("=== v2.0 Integration Completed Successfully ===")
    
    return {
        'output_file': output_file,
        'config_file': config_file,
        'total_features': len(v20_features),
        'baseline_features': len(v12_features),
        'odds_features': len(core_odds_features),
        'merge_quality_pct': missing_pct,
        'total_matches': total_matches
    }

if __name__ == "__main__":
    result = integrate_odds_with_baseline()
    if result:
        print(f"\nv2.0 Integration SUCCESS!")
        print(f"Dataset: {result['output_file']}")
        print(f"Features: {result['baseline_features']} baseline + {result['odds_features']} odds = {result['total_features']} total")
        print(f"Merge quality: {result['merge_quality_pct']:.1f}% missing")
        print(f"Total matches: {result['total_matches']}")
    else:
        print("❌ Integration FAILED")