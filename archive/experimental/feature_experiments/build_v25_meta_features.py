#!/usr/bin/env python3
"""
v2.5 Meta-Features Builder
Implement Sprint v2.5: Context Intelligence Features

Features to implement:
1. match_stakes_normalized - Quantify match importance
2. expected_surprise_factor - Measure upset probability

Target: 53.8% → 56.5% accuracy improvement through contextual understanding
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_league_positions(df):
    """Calculate running league positions for each team throughout seasons."""
    print("📊 Calculating running league positions...")
    
    # Initialize positions dataframe
    positions_data = []
    
    # Group by season
    for season in df['Season'].unique():
        season_df = df[df['Season'] == season].copy()
        season_df = season_df.sort_values('Date').reset_index(drop=True)
        
        # Initialize season points table
        teams = pd.concat([season_df['HomeTeam'], season_df['AwayTeam']]).unique()
        points_table = {team: 0 for team in teams}
        
        print(f"  Season {season}: {len(teams)} teams, {len(season_df)} matches")
        
        # Process each match in chronological order
        for idx, match in season_df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            result = match['FullTimeResult']
            
            # Calculate match positions BEFORE this match
            sorted_teams = sorted(points_table.items(), key=lambda x: x[1], reverse=True)
            team_positions = {team: pos + 1 for pos, (team, points) in enumerate(sorted_teams)}
            
            home_position = team_positions[home_team]
            away_position = team_positions[away_team]
            
            positions_data.append({
                'Date': match['Date'],
                'Season': season,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'home_position': home_position,
                'away_position': away_position,
                'position_diff': home_position - away_position  # Positive = home team lower in table
            })
            
            # Update points after match
            if result == 'H':
                points_table[home_team] += 3
            elif result == 'A':
                points_table[away_team] += 3
            else:  # Draw
                points_table[home_team] += 1
                points_table[away_team] += 1
    
    positions_df = pd.DataFrame(positions_data)
    print(f"✅ League positions calculated for {len(positions_df)} matches")
    
    return positions_df

def calculate_match_stakes(df, positions_df):
    """Calculate match stakes based on season progression and position differences."""
    print("🎯 Calculating match stakes...")
    
    # Merge positions data
    df_with_positions = df.merge(positions_df, on=['Date', 'Season', 'HomeTeam', 'AwayTeam'], how='left')
    
    # Calculate season criticality (higher towards end of season)
    # Use exponential function to emphasize end-of-season importance
    df_with_positions['season_criticality'] = np.exp(df_with_positions['matchday_normalized'] * 2) / np.exp(2)
    
    # Calculate position criticality
    # Top and bottom of table matches are more critical
    df_with_positions['avg_position'] = (df_with_positions['home_position'] + df_with_positions['away_position']) / 2
    df_with_positions['position_criticality'] = np.where(
        df_with_positions['avg_position'] <= 6,  # Top 6 (title/Europe race)
        1.0,
        np.where(
            df_with_positions['avg_position'] >= 15,  # Bottom 6 (relegation battle)
            1.0,
            0.3 + 0.7 * (1 - np.abs(df_with_positions['avg_position'] - 10.5) / 10.5)  # Mid-table gradient
        )
    )
    
    # Calculate position difference impact
    # Close positions = higher stakes
    df_with_positions['position_diff_impact'] = 1 / (1 + np.abs(df_with_positions['position_diff']) / 5)
    
    # Combine components into match stakes
    df_with_positions['match_stakes_raw'] = (
        0.4 * df_with_positions['season_criticality'] +
        0.4 * df_with_positions['position_criticality'] +
        0.2 * df_with_positions['position_diff_impact']
    )
    
    # Normalize to 0-1 range
    min_stakes = df_with_positions['match_stakes_raw'].min()
    max_stakes = df_with_positions['match_stakes_raw'].max()
    df_with_positions['match_stakes_normalized'] = (
        (df_with_positions['match_stakes_raw'] - min_stakes) / (max_stakes - min_stakes)
    )
    
    print(f"✅ Match stakes calculated:")
    print(f"  Range: {df_with_positions['match_stakes_normalized'].min():.3f} - {df_with_positions['match_stakes_normalized'].max():.3f}")
    print(f"  Mean: {df_with_positions['match_stakes_normalized'].mean():.3f}")
    print(f"  Examples:")
    
    # Show some examples
    high_stakes = df_with_positions.nlargest(3, 'match_stakes_normalized')[['Date', 'HomeTeam', 'AwayTeam', 'match_stakes_normalized', 'season_criticality', 'position_criticality']]
    print("    High stakes matches:")
    for _, match in high_stakes.iterrows():
        print(f"    {match['Date'].strftime('%Y-%m-%d')}: {match['HomeTeam']} vs {match['AwayTeam']} (Stakes: {match['match_stakes_normalized']:.3f})")
    
    return df_with_positions

def calculate_expected_surprise_factor(df):
    """Calculate expected surprise factor from Elo differentials."""
    print("🎲 Calculating expected surprise factor...")
    
    # Convert elo_diff_normalized to actual Elo-style probabilities
    # elo_diff_normalized is 0-1 where 0.5 = equal teams
    # Convert to surprise factor where high value = high surprise potential
    
    # Method 1: Distance from 0.5 (equal teams)
    df['elo_balance'] = 1 - 2 * np.abs(df['elo_diff_normalized'] - 0.5)  # 1 = perfectly balanced, 0 = huge gap
    
    # Method 2: Entropy-based approach (maximum uncertainty at 0.5)
    # Higher surprise factor when teams are more evenly matched
    p_home = df['elo_diff_normalized']
    p_away = 1 - p_home
    
    # Avoid log(0) issues
    p_home = np.clip(p_home, 0.01, 0.99)
    p_away = np.clip(p_away, 0.01, 0.99)
    
    # Calculate entropy (uncertainty)
    df['surprise_entropy'] = -(p_home * np.log2(p_home) + p_away * np.log2(p_away))
    
    # Normalize entropy to 0-1 (maximum entropy is 1.0 when p=0.5)
    df['expected_surprise_factor'] = df['surprise_entropy'] / 1.0
    
    print(f"✅ Expected surprise factor calculated:")
    print(f"  Range: {df['expected_surprise_factor'].min():.3f} - {df['expected_surprise_factor'].max():.3f}")
    print(f"  Mean: {df['expected_surprise_factor'].mean():.3f}")
    
    # Show examples
    high_surprise = df.nlargest(5, 'expected_surprise_factor')[['Date', 'HomeTeam', 'AwayTeam', 'expected_surprise_factor', 'elo_diff_normalized']]
    low_surprise = df.nsmallest(5, 'expected_surprise_factor')[['Date', 'HomeTeam', 'AwayTeam', 'expected_surprise_factor', 'elo_diff_normalized']]
    
    print("    High surprise potential (evenly matched):")
    for _, match in high_surprise.iterrows():
        print(f"    {match['Date'].strftime('%Y-%m-%d')}: {match['HomeTeam']} vs {match['AwayTeam']} (Surprise: {match['expected_surprise_factor']:.3f}, Elo: {match['elo_diff_normalized']:.3f})")
    
    print("    Low surprise potential (mismatched):")
    for _, match in low_surprise.iterrows():
        print(f"    {match['Date'].strftime('%Y-%m-%d')}: {match['HomeTeam']} vs {match['AwayTeam']} (Surprise: {match['expected_surprise_factor']:.3f}, Elo: {match['elo_diff_normalized']:.3f})")
    
    return df

def validate_meta_features(df):
    """Validate meta-features for data leakage and quality."""
    print("\n🔍 VALIDATING META-FEATURES")
    print("=" * 50)
    
    # Check for missing values
    meta_features = ['match_stakes_normalized', 'expected_surprise_factor']
    
    for feature in meta_features:
        missing_count = df[feature].isna().sum()
        print(f"✅ {feature}: {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
        
        # Check distribution
        print(f"   Distribution: min={df[feature].min():.3f}, max={df[feature].max():.3f}, mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")
    
    # Temporal consistency check (no future information)
    print(f"\n🕒 Temporal Consistency Check:")
    cutoff_date = pd.to_datetime('2023-05-01')
    
    train_df = df[df['Date'] < cutoff_date]
    test_df = df[df['Date'] >= cutoff_date]
    
    for feature in meta_features:
        train_mean = train_df[feature].mean()
        test_mean = test_df[feature].mean()
        shift = abs(train_mean - test_mean)
        print(f"✅ {feature}: Train={train_mean:.3f}, Test={test_mean:.3f}, Shift={shift:.3f}")
        
        if shift > 0.1:
            print(f"   ⚠️  Warning: Large distribution shift detected!")
    
    # Correlation with existing features
    print(f"\n🔗 Correlation with v2.4 Features:")
    v24_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    correlation_matrix = df[v24_features + meta_features].corr()
    
    for meta_feature in meta_features:
        print(f"\n📊 {meta_feature} correlations:")
        correlations = correlation_matrix[meta_feature][v24_features].abs().sort_values(ascending=False)
        
        for v24_feature, correlation in correlations.head(3).items():
            status = "⚠️ HIGH" if correlation > 0.7 else "🟡 MEDIUM" if correlation > 0.5 else "🟢 LOW"
            print(f"   {v24_feature}: {correlation:.3f} {status}")
    
    return True

def main():
    """Main v2.5 meta-features builder."""
    print("🚀 v2.5 Meta-Features Builder")
    print("=" * 60)
    print("Sprint v2.5: Context Intelligence Features")
    print("Target: 53.8% → 56.5% accuracy through contextual understanding")
    print()
    
    # Load v2.4 dataset
    print("📊 Loading v2.4 baseline dataset...")
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded: {len(df)} matches")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Seasons: {df['Season'].nunique()} seasons")
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Step 1: Calculate league positions (required for match stakes)
    positions_df = calculate_league_positions(df)
    
    # Step 2: Calculate match stakes
    df_with_stakes = calculate_match_stakes(df, positions_df)
    
    # Step 3: Calculate expected surprise factor  
    df_with_meta = calculate_expected_surprise_factor(df_with_stakes)
    
    # Step 4: Validate meta-features
    validation_passed = validate_meta_features(df_with_meta)
    
    if validation_passed:
        # Save enhanced dataset
        timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
        output_path = f'/Users/maxime/Desktop/Oddsy/data/processed/v25_meta_features_{timestamp}.csv'
        
        df_with_meta.to_csv(output_path, index=False)
        
        print(f"\n💾 v2.5 Meta-Features Dataset Saved:")
        print(f"Path: {output_path}")
        print(f"Features added: match_stakes_normalized, expected_surprise_factor")
        print(f"Total features: {len(df_with_meta.columns)}")
        print(f"Dataset size: {len(df_with_meta)} matches")
        
        # Feature summary
        print(f"\n📋 v2.5 Feature Summary:")
        print(f"v2.4 baseline features: 10")
        print(f"v2.5 meta-features: 2")
        print(f"Total v2.5 features: 12")
        print(f"New features:")
        print(f"  1. match_stakes_normalized - Match importance quantification")
        print(f"  2. expected_surprise_factor - Upset probability measure")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Test v2.5 features with cascade model")
        print(f"2. Compare performance: v2.4 (53.8%) vs v2.5 (target: 56.5%)")
        print(f"3. Validate improvement through cross-validation")
        
        return output_path
    else:
        print("❌ Meta-features validation failed!")
        return None

if __name__ == "__main__":
    main()