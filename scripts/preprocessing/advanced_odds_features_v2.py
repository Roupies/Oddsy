import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def engineer_advanced_odds_features():
    """
    PHASE 2.0.1 - Advanced Odds Features Engineering
    
    Focus on sophisticated market signals that go beyond basic consensus:
    1. Line Movement Analysis (early vs closing odds)
    2. Sharp vs Recreational Money (Pinnacle vs Bet365 divergence)
    3. Market Inefficiency Detection (cross-bookmaker disagreement)
    4. Odds Velocity Tracking (rate of change patterns)
    
    CRITICAL: Correlation control with existing features (especially Elo)
    Target: True v2.0 breakthrough to 55%+ accuracy
    """
    
    logger = setup_logging()
    logger.info("=== 🚀 PHASE 2.0.1 - Advanced Odds Features Engineering ===")
    
    # Load raw odds data directly from football_data_backup (need individual bookmaker columns)
    backup_dir = "data/raw/football_data_backup/"
    backup_files = [
        "football_data_2019_20.csv",
        "football_data_2020_21.csv", 
        "football_data_2021_22.csv",
        "football_data_2022_23.csv",
        "football_data_2023_24.csv",
        "football_data_2024_25.csv"
    ]
    
    logger.info("Loading raw odds data from football_data_backup...")
    all_odds_data = []
    
    for file in backup_files:
        filepath = os.path.join(backup_dir, file)
        logger.info(f"  Processing {file}...")
        
        df = pd.read_csv(filepath)
        season = file.replace('football_data_', '').replace('.csv', '')
        df['Season'] = season
        all_odds_data.append(df)
    
    # Combine all seasons
    df_odds = pd.concat(all_odds_data, ignore_index=True)
    df_odds['Date'] = pd.to_datetime(df_odds['Date'], format='%d/%m/%Y')
    df_odds = df_odds.sort_values('Date').reset_index(drop=True)
    logger.info(f"Raw odds dataset: {df_odds.shape}")
    
    # Rename to standard format
    if 'FTR' in df_odds.columns:
        df_odds['FullTimeResult'] = df_odds['FTR']
    
    # Load baseline for correlation control (need Date/team columns)
    baseline_file = "data/processed/premier_league_2019_2024_corrected_elo.csv"
    df_baseline = pd.read_csv(baseline_file)
    logger.info(f"Baseline for correlation control: {df_baseline.shape}")
    
    logger.info("Engineering advanced odds features...")
    
    # =========================
    # 1. LINE MOVEMENT ANALYSIS
    # =========================
    logger.info("\n🔍 1. LINE MOVEMENT ANALYSIS")
    
    # Early vs Closing odds spread (market movement pattern)
    # Using first available bookmaker as "early" proxy and Pinnacle as "closing"
    
    # Calculate line movement for Home odds
    df_odds['early_odds_home'] = df_odds['B365H']  # Bet365 as early indicator
    df_odds['closing_odds_home'] = df_odds['PSH']  # Pinnacle as closing (sharp)
    
    # Line movement = (closing - early) / early (percentage change)
    df_odds['line_movement_home'] = (df_odds['closing_odds_home'] - df_odds['early_odds_home']) / df_odds['early_odds_home']
    
    # Positive = odds increased (team became less favored)
    # Negative = odds decreased (team became more favored)
    
    # Same for Away teams
    df_odds['early_odds_away'] = df_odds['B365A']
    df_odds['closing_odds_away'] = df_odds['PSA']
    df_odds['line_movement_away'] = (df_odds['closing_odds_away'] - df_odds['early_odds_away']) / df_odds['early_odds_away']
    
    # Net line movement (Home perspective)
    df_odds['net_line_movement'] = df_odds['line_movement_home'] - df_odds['line_movement_away']
    
    # Normalize to [0,1] for ML compatibility
    line_movement_min = df_odds['net_line_movement'].min()
    line_movement_max = df_odds['net_line_movement'].max()
    df_odds['line_movement_normalized'] = ((df_odds['net_line_movement'] - line_movement_min) / 
                                         (line_movement_max - line_movement_min))
    
    logger.info(f"    Line movement range: [{line_movement_min:.3f}, {line_movement_max:.3f}]")
    logger.info(f"    Mean line movement: {df_odds['net_line_movement'].mean():.3f}")
    
    # =========================
    # 2. SHARP VS RECREATIONAL MONEY
    # =========================
    logger.info("\n💰 2. SHARP VS RECREATIONAL MONEY DIVERGENCE")
    
    # Pinnacle = sharp money (professional bettors)
    # Bet365 = recreational money (general public)
    # Divergence indicates where smart money disagrees with public
    
    # Convert odds to implied probabilities
    df_odds['pinnacle_prob_home'] = 1 / df_odds['PSH']
    df_odds['bet365_prob_home'] = 1 / df_odds['B365H']
    df_odds['pinnacle_prob_away'] = 1 / df_odds['PSA']
    df_odds['bet365_prob_away'] = 1 / df_odds['B365A']
    
    # Sharp vs public divergence (Home perspective)
    df_odds['sharp_public_divergence'] = ((df_odds['pinnacle_prob_home'] - df_odds['bet365_prob_home']) -
                                        (df_odds['pinnacle_prob_away'] - df_odds['bet365_prob_away']))
    
    # Positive = sharp money favors home more than public
    # Negative = sharp money favors away more than public
    
    # Normalize to [0,1]
    divergence_min = df_odds['sharp_public_divergence'].min()
    divergence_max = df_odds['sharp_public_divergence'].max()
    df_odds['sharp_public_divergence_norm'] = ((df_odds['sharp_public_divergence'] - divergence_min) / 
                                             (divergence_max - divergence_min))
    
    logger.info(f"    Sharp vs public range: [{divergence_min:.3f}, {divergence_max:.3f}]")
    logger.info(f"    Mean divergence: {df_odds['sharp_public_divergence'].mean():.3f}")
    
    # =========================
    # 3. MARKET INEFFICIENCY DETECTION
    # =========================
    logger.info("\n⚡ 3. MARKET INEFFICIENCY DETECTION")
    
    # Standard deviation of odds across ALL bookmakers (market disagreement)
    bookmaker_cols_home = ['B365H', 'BWH', 'PSH', 'WHH']  # Available bookmakers
    bookmaker_cols_away = ['B365A', 'BWA', 'PSA', 'WHA']
    
    # Calculate market disagreement for Home odds
    home_odds_matrix = df_odds[bookmaker_cols_home].fillna(method='ffill', axis=1)
    df_odds['market_disagreement_home'] = home_odds_matrix.std(axis=1, skipna=True)
    
    # Calculate market disagreement for Away odds  
    away_odds_matrix = df_odds[bookmaker_cols_away].fillna(method='ffill', axis=1)
    df_odds['market_disagreement_away'] = away_odds_matrix.std(axis=1, skipna=True)
    
    # Net market inefficiency (higher = more disagreement = potential value)
    df_odds['market_inefficiency'] = df_odds['market_disagreement_home'] + df_odds['market_disagreement_away']
    
    # Normalize to [0,1]
    inefficiency_max = df_odds['market_inefficiency'].max()
    df_odds['market_inefficiency_norm'] = df_odds['market_inefficiency'] / inefficiency_max if inefficiency_max > 0 else 0
    
    logger.info(f"    Market inefficiency mean: {df_odds['market_inefficiency'].mean():.3f}")
    logger.info(f"    Max disagreement: {inefficiency_max:.3f}")
    
    # =========================
    # 4. ODDS VELOCITY TRACKING
    # =========================
    logger.info("\n🏃 4. ODDS VELOCITY TRACKING")
    
    # Calculate rate of change between different bookmakers (proxy for time-based movement)
    # Using bookmaker ordering as time proxy: B365 → BW → PS → WH (market evolution)
    
    df_odds['odds_velocity_home'] = np.abs(df_odds['PSH'] - df_odds['B365H']) / df_odds['B365H']
    df_odds['odds_velocity_away'] = np.abs(df_odds['PSA'] - df_odds['B365A']) / df_odds['B365A']
    
    # Net velocity (market movement magnitude)
    df_odds['market_velocity'] = (df_odds['odds_velocity_home'] + df_odds['odds_velocity_away']) / 2
    
    # Normalize to [0,1]
    velocity_max = df_odds['market_velocity'].max()
    df_odds['market_velocity_norm'] = df_odds['market_velocity'] / velocity_max if velocity_max > 0 else 0
    
    logger.info(f"    Market velocity mean: {df_odds['market_velocity'].mean():.3f}")
    logger.info(f"    Max velocity: {velocity_max:.3f}")
    
    # =========================
    # 5. CORRELATION CONTROL CHECK
    # =========================
    logger.info("\n🔍 5. CORRELATION CONTROL WITH BASELINE FEATURES")
    
    # Advanced features to validate
    advanced_features = [
        'line_movement_normalized',
        'sharp_public_divergence_norm', 
        'market_inefficiency_norm',
        'market_velocity_norm'
    ]
    
    # Create matching keys for correlation analysis
    df_odds['match_key'] = (df_odds['Date'].dt.strftime('%Y-%m-%d') + '_' + 
                           df_odds['HomeTeam'] + '_' + df_odds['AwayTeam'])
    
    # Load baseline features for correlation check (available in corrected_elo dataset)
    baseline_features = ['elo_diff_normalized', 'form_diff_normalized', 'h2h_score', 'home_advantage']
    
    # Merge for correlation analysis
    df_baseline['Date'] = pd.to_datetime(df_baseline['Date'])
    df_baseline['match_key'] = (df_baseline['Date'].dt.strftime('%Y-%m-%d') + '_' + 
                               df_baseline['HomeTeam'] + '_' + df_baseline['AwayTeam'])
    
    # Merge datasets
    df_correlation_check = df_baseline[['match_key'] + baseline_features].merge(
        df_odds[['match_key'] + advanced_features], 
        on='match_key', 
        how='inner'
    )
    
    logger.info(f"    Correlation analysis dataset: {df_correlation_check.shape}")
    
    # Calculate correlations
    logger.info("\n    Advanced features correlation with baseline:")
    correlation_issues = []
    
    for adv_feat in advanced_features:
        logger.info(f"\n    {adv_feat} correlations:")
        for base_feat in baseline_features:
            corr = df_correlation_check[adv_feat].corr(df_correlation_check[base_feat])
            abs_corr = abs(corr)
            
            status = ""
            if abs_corr > 0.8:
                status = "🚨 HIGH CORRELATION - REMOVE"
                correlation_issues.append((adv_feat, base_feat, corr))
            elif abs_corr > 0.6:
                status = "⚠️ MODERATE CORRELATION - MONITOR"
            else:
                status = "✅ OK"
            
            logger.info(f"      vs {base_feat}: {corr:.3f} {status}")
    
    # =========================
    # 6. FEATURE SELECTION & VALIDATION
    # =========================
    logger.info("\n🎯 6. FEATURE SELECTION & VALIDATION")
    
    # Remove highly correlated features
    features_to_keep = []
    features_to_remove = []
    
    for adv_feat in advanced_features:
        max_corr = max([abs(df_correlation_check[adv_feat].corr(df_correlation_check[base_feat])) 
                       for base_feat in baseline_features])
        
        if max_corr > 0.75:
            features_to_remove.append(adv_feat)
            logger.warning(f"    🗑️ REMOVING {adv_feat}: max correlation = {max_corr:.3f}")
        else:
            features_to_keep.append(adv_feat)
            logger.info(f"    ✅ KEEPING {adv_feat}: max correlation = {max_corr:.3f}")
    
    logger.info(f"\n    Features analysis:")
    logger.info(f"      Total engineered: {len(advanced_features)}")
    logger.info(f"      Keeping: {len(features_to_keep)} - {features_to_keep}")
    logger.info(f"      Removing: {len(features_to_remove)} - {features_to_remove}")
    
    # =========================
    # 7. FINAL FEATURE STATISTICS
    # =========================
    logger.info("\n📊 7. FINAL ADVANCED FEATURES STATISTICS")
    
    for feature in features_to_keep:
        values = df_odds[feature].dropna()
        logger.info(f"    {feature}:")
        logger.info(f"      Range: [{values.min():.3f}, {values.max():.3f}]")
        logger.info(f"      Mean: {values.mean():.3f}, Std: {values.std():.3f}")
        logger.info(f"      Missing: {df_odds[feature].isnull().sum()} / {len(df_odds)}")
    
    # =========================
    # 8. SAVE ADVANCED FEATURES
    # =========================
    logger.info("\n💾 8. SAVING ADVANCED ODDS FEATURES")
    
    # Select output columns
    output_columns = [
        'Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult'
    ] + features_to_keep + [
        # Keep raw data for analysis
        'early_odds_home', 'closing_odds_home', 'net_line_movement',
        'pinnacle_prob_home', 'bet365_prob_home', 'sharp_public_divergence',
        'market_disagreement_home', 'market_disagreement_away', 'market_inefficiency',
        'odds_velocity_home', 'odds_velocity_away', 'market_velocity'
    ]
    
    # Filter to existing columns
    existing_columns = [col for col in output_columns if col in df_odds.columns]
    df_output = df_odds[existing_columns].copy()
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = f"data/processed/advanced_odds_features_v2_{timestamp}.csv"
    
    df_output.to_csv(output_file, index=False)
    logger.info(f"    Advanced features saved: {output_file}")
    logger.info(f"    Output shape: {df_output.shape}")
    
    # Save feature metadata
    feature_metadata = {
        "version": "v2.0.1 Advanced Odds Features",
        "timestamp": timestamp,
        "total_features_engineered": len(advanced_features),
        "features_kept": features_to_keep,
        "features_removed": features_to_remove,
        "correlation_threshold": 0.75,
        "correlation_issues": [{"feature": feat, "baseline": base, "correlation": float(corr)} 
                              for feat, base, corr in correlation_issues],
        "feature_descriptions": {
            "line_movement_normalized": "Early vs closing odds movement (market sentiment shift)",
            "sharp_public_divergence_norm": "Pinnacle vs Bet365 divergence (smart vs public money)",
            "market_inefficiency_norm": "Cross-bookmaker disagreement (potential value opportunities)",
            "market_velocity_norm": "Rate of odds change (market momentum)"
        },
        "data_quality": {
            "total_matches": len(df_output),
            "missing_values": int(df_output[features_to_keep].isnull().sum().sum()),
            "correlation_controlled": True
        }
    }
    
    import json
    metadata_file = f"config/advanced_odds_features_v2_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    logger.info(f"    Feature metadata saved: {metadata_file}")
    
    logger.info("\n=== 🚀 PHASE 2.0.1 ADVANCED ODDS FEATURES COMPLETED ===")
    
    return {
        'output_file': output_file,
        'metadata_file': metadata_file,
        'features_engineered': len(advanced_features),
        'features_kept': features_to_keep,
        'features_removed': features_to_remove,
        'correlation_controlled': len(correlation_issues) == len(features_to_remove),
        'total_matches': len(df_output)
    }

if __name__ == "__main__":
    result = engineer_advanced_odds_features()
    print(f"\n🎯 ADVANCED ODDS FEATURES RESULTS:")
    print(f"Features Engineered: {result['features_engineered']}")
    print(f"Features Kept: {len(result['features_kept'])} - {result['features_kept']}")
    print(f"Features Removed: {len(result['features_removed'])} - {result['features_removed']}")
    print(f"Correlation Controlled: {'✅' if result['correlation_controlled'] else '❌'}")
    print(f"Output: {result['output_file']}")
    print(f"Total Matches: {result['total_matches']}")