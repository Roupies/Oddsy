#!/usr/bin/env python3
"""
v2.7 H2H Intelligence Features Builder
Implement Sprint v2.7: Psychology of Confrontations (H2H Intelligence)

Features to implement:
1. bogey_team_score - Teams that consistently underperform vs specific opponents
2. h2h_context_score - Weight H2H by match importance when they occurred

Target: 55.1% → 58%+ accuracy improvement through psychological/tactical insights
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_historical_league_positions(df):
    """Calculate league positions for each match (needed for expected results)."""
    print("📊 Calculating historical league positions...")
    
    positions_data = []
    
    for season in df['Season'].unique():
        season_df = df[df['Season'] == season].copy()
        season_df = season_df.sort_values('Date').reset_index(drop=True)
        
        teams = pd.concat([season_df['HomeTeam'], season_df['AwayTeam']]).unique()
        points_table = {team: 0 for team in teams}
        
        for idx, match in season_df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            result = match['FullTimeResult']
            
            # Get positions BEFORE this match
            sorted_teams = sorted(points_table.items(), key=lambda x: x[1], reverse=True)
            positions = {team: pos + 1 for pos, (team, points) in enumerate(sorted_teams)}
            
            positions_data.append({
                'Date': match['Date'],
                'Season': season,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'home_position': positions[home_team],
                'away_position': positions[away_team]
            })
            
            # Update points after match
            if result == 'H':
                points_table[home_team] += 3
            elif result == 'A':
                points_table[away_team] += 3
            else:
                points_table[home_team] += 1
                points_table[away_team] += 1
    
    return pd.DataFrame(positions_data)

def calculate_bogey_team_scores(df, positions_df):
    """Calculate bogey team scores - teams that underperform vs specific opponents."""
    print("👻 Calculating bogey team scores...")
    
    # Merge position data
    df_with_pos = df.merge(positions_df, on=['Date', 'Season', 'HomeTeam', 'AwayTeam'], how='left')
    
    # Calculate expected results based on league positions
    # Better positioned team (lower number) should have advantage
    df_with_pos['position_advantage'] = df_with_pos['away_position'] - df_with_pos['home_position']
    
    # Convert results to numeric (Home perspective)
    df_with_pos['result_numeric'] = df_with_pos['FullTimeResult'].map({'H': 1, 'D': 0, 'A': -1})
    
    # Expected result based on position difference
    # Positive advantage = home team better positioned = expect positive result
    df_with_pos['expected_result'] = np.tanh(df_with_pos['position_advantage'] / 5.0)  # Smooth sigmoid
    
    # Calculate performance vs expectation
    df_with_pos['result_vs_expected'] = df_with_pos['result_numeric'] - df_with_pos['expected_result']
    
    # Calculate bogey scores for each team pairing
    bogey_data = []
    
    # Get all unique team pairs
    all_matchups = []
    for _, match in df_with_pos.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        matchup = tuple(sorted([home_team, away_team]))
        if matchup not in all_matchups:
            all_matchups.append(matchup)
    
    print(f"Analyzing {len(all_matchups)} unique team matchups...")
    
    matchup_stats = {}
    
    for team1, team2 in all_matchups:
        # Get all matches between these teams
        matchup_matches = df_with_pos[
            ((df_with_pos['HomeTeam'] == team1) & (df_with_pos['AwayTeam'] == team2)) |
            ((df_with_pos['HomeTeam'] == team2) & (df_with_pos['AwayTeam'] == team1))
        ].copy()
        
        if len(matchup_matches) >= 3:  # Need at least 3 meetings for meaningful stat
            # Calculate each team's performance in this matchup
            team1_performance = []
            team2_performance = []
            
            for _, match in matchup_matches.iterrows():
                if match['HomeTeam'] == team1:
                    team1_performance.append(match['result_vs_expected'])
                    team2_performance.append(-match['result_vs_expected'])
                else:
                    team1_performance.append(-match['result_vs_expected'])
                    team2_performance.append(match['result_vs_expected'])
            
            team1_avg = np.mean(team1_performance)
            team2_avg = np.mean(team2_performance)
            
            matchup_stats[(team1, team2)] = {
                'team1_performance': team1_avg,
                'team2_performance': team2_avg,
                'meetings': len(matchup_matches)
            }
    
    # Now assign bogey scores to each match
    for idx, match in df_with_pos.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        matchup = tuple(sorted([home_team, away_team]))
        
        if matchup in matchup_stats:
            stats = matchup_stats[matchup]
            
            # Determine which team is which in the sorted matchup
            if matchup[0] == home_team:
                home_performance = stats['team1_performance']
                away_performance = stats['team2_performance']
            else:
                home_performance = stats['team2_performance'] 
                away_performance = stats['team1_performance']
            
            # Bogey score = away team's historical overperformance vs home team
            # Positive = away team is "bogey team" for home team
            bogey_score = away_performance - home_performance
            
        else:
            bogey_score = 0.0  # Neutral for insufficient data
        
        bogey_data.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'bogey_score_raw': bogey_score
        })
    
    bogey_df = pd.DataFrame(bogey_data)
    
    # Normalize bogey scores to 0-1
    min_score = bogey_df['bogey_score_raw'].min()
    max_score = bogey_df['bogey_score_raw'].max()
    bogey_df['bogey_team_score'] = (bogey_df['bogey_score_raw'] - min_score) / (max_score - min_score)
    
    # Show examples
    high_bogey = bogey_df.nlargest(5, 'bogey_team_score')
    print("\n👻 Top Bogey Team Examples (Away team historically dominates Home):")
    for _, match in high_bogey.iterrows():
        print(f"  {match['Date'].strftime('%Y-%m-%d')}: {match['HomeTeam']} vs {match['AwayTeam']} (Bogey: {match['bogey_team_score']:.3f})")
    
    print(f"\n✅ Bogey team scores calculated:")
    print(f"  Range: {bogey_df['bogey_team_score'].min():.3f} - {bogey_df['bogey_team_score'].max():.3f}")
    print(f"  Mean: {bogey_df['bogey_team_score'].mean():.3f}")
    
    return bogey_df

def calculate_context_weighted_h2h(df, positions_df):
    """Calculate context-weighted H2H scores using match importance."""
    print("⚖️ Calculating context-weighted H2H scores...")
    
    # Merge position data
    df_with_pos = df.merge(positions_df, on=['Date', 'Season', 'HomeTeam', 'AwayTeam'], how='left')
    
    # Calculate match importance (similar to v2.5 match stakes)
    # Season criticality
    df_with_pos['season_criticality'] = np.exp(df_with_pos['matchday_normalized'] * 2) / np.exp(2)
    
    # Position importance (top 6 and bottom 6 more important)
    df_with_pos['avg_position'] = (df_with_pos['home_position'] + df_with_pos['away_position']) / 2
    df_with_pos['position_importance'] = np.where(
        df_with_pos['avg_position'] <= 6,  # Top 6
        1.0,
        np.where(
            df_with_pos['avg_position'] >= 15,  # Bottom 6
            1.0,
            0.3 + 0.7 * (1 - np.abs(df_with_pos['avg_position'] - 10.5) / 10.5)
        )
    )
    
    # Combined match importance
    df_with_pos['match_importance'] = (
        0.6 * df_with_pos['season_criticality'] + 
        0.4 * df_with_pos['position_importance']
    )
    
    # Calculate weighted H2H for each match
    h2h_data = []
    
    for idx, match in df_with_pos.iterrows():
        if idx % 500 == 0:
            print(f"  Processing match {idx+1}/{len(df_with_pos)}")
            
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        current_date = match['Date']
        
        # Get all previous matches between these teams
        historical_matches = df_with_pos[
            (df_with_pos['Date'] < current_date) &
            (
                ((df_with_pos['HomeTeam'] == home_team) & (df_with_pos['AwayTeam'] == away_team)) |
                ((df_with_pos['HomeTeam'] == away_team) & (df_with_pos['AwayTeam'] == home_team))
            )
        ].copy()
        
        if len(historical_matches) > 0:
            # Calculate weighted H2H score
            home_weighted_points = 0
            away_weighted_points = 0
            total_weight = 0
            
            for _, hist_match in historical_matches.iterrows():
                weight = hist_match['match_importance']
                total_weight += weight
                
                if hist_match['HomeTeam'] == home_team:
                    # Current home team was home in historical match
                    if hist_match['FullTimeResult'] == 'H':
                        home_weighted_points += 3 * weight
                    elif hist_match['FullTimeResult'] == 'D':
                        home_weighted_points += 1 * weight
                        away_weighted_points += 1 * weight
                    else:  # 'A'
                        away_weighted_points += 3 * weight
                else:
                    # Current home team was away in historical match
                    if hist_match['FullTimeResult'] == 'A':
                        home_weighted_points += 3 * weight
                    elif hist_match['FullTimeResult'] == 'D':
                        home_weighted_points += 1 * weight
                        away_weighted_points += 1 * weight
                    else:  # 'H'
                        away_weighted_points += 3 * weight
            
            if total_weight > 0:
                home_avg = home_weighted_points / total_weight
                away_avg = away_weighted_points / total_weight
                
                # Context weighted H2H score (home perspective)
                context_h2h_score = (home_avg - away_avg) / 3.0  # Normalize to [-1, 1]
            else:
                context_h2h_score = 0.0
        else:
            context_h2h_score = 0.0  # No historical meetings
        
        h2h_data.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'context_h2h_raw': context_h2h_score,
            'historical_meetings': len(historical_matches)
        })
    
    h2h_df = pd.DataFrame(h2h_data)
    
    # Normalize to 0-1 range
    min_h2h = h2h_df['context_h2h_raw'].min()
    max_h2h = h2h_df['context_h2h_raw'].max()
    h2h_df['h2h_context_score'] = (h2h_df['context_h2h_raw'] - min_h2h) / (max_h2h - min_h2h)
    
    # Show examples
    high_context = h2h_df[h2h_df['historical_meetings'] >= 5].nlargest(3, 'h2h_context_score')
    print(f"\n⚖️ High context-weighted H2H examples (Home team dominance):")
    for _, match in high_context.iterrows():
        print(f"  {match['Date'].strftime('%Y-%m-%d')}: {match['HomeTeam']} vs {match['AwayTeam']} "
              f"(Context H2H: {match['h2h_context_score']:.3f}, Meetings: {match['historical_meetings']})")
    
    print(f"\n✅ Context-weighted H2H calculated:")
    print(f"  Range: {h2h_df['h2h_context_score'].min():.3f} - {h2h_df['h2h_context_score'].max():.3f}")
    print(f"  Mean: {h2h_df['h2h_context_score'].mean():.3f}")
    
    return h2h_df

def validate_h2h_features(df):
    """Validate H2H intelligence features."""
    print("\n🔍 VALIDATING H2H INTELLIGENCE FEATURES")
    print("=" * 50)
    
    h2h_features = ['bogey_team_score', 'h2h_context_score']
    
    # Missing values check
    for feature in h2h_features:
        missing_count = df[feature].isna().sum()
        print(f"✅ {feature}: {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
    
    # Temporal consistency
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df[df['Date'] < cutoff_date]
    test_df = df[df['Date'] >= cutoff_date]
    
    print(f"\n🕒 Temporal Consistency Check:")
    for feature in h2h_features:
        train_mean = train_df[feature].mean()
        test_mean = test_df[feature].mean()
        shift = abs(train_mean - test_mean)
        
        print(f"✅ {feature}: Train={train_mean:.3f}, Test={test_mean:.3f}, Shift={shift:.3f}")
        
        if shift > 0.05:
            print(f"   ⚠️  Warning: Distribution shift detected!")
    
    # Correlation with existing features
    print(f"\n🔗 Correlation with v2.6 Features:")
    v26_features = [
        'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'market_entropy_norm', 'shots_diff_normalized'
    ]
    
    correlation_matrix = df[v26_features + h2h_features].corr()
    
    for h2h_feature in h2h_features:
        print(f"\n📊 {h2h_feature} correlations:")
        correlations = correlation_matrix[h2h_feature][v26_features].abs().sort_values(ascending=False)
        
        for v26_feature, correlation in correlations.head(3).items():
            status = "⚠️ HIGH" if correlation > 0.7 else "🟡 MEDIUM" if correlation > 0.5 else "🟢 LOW"
            print(f"   {v26_feature}: {correlation:.3f} {status}")
    
    return True

def main():
    """Main v2.7 H2H intelligence features builder."""
    print("🧠 v2.7 H2H Intelligence Features Builder")
    print("=" * 60)
    print("Sprint v2.7: Psychology of Confrontations (H2H Intelligence)")
    print("Target: 55.1% → 58%+ accuracy through psychological/tactical insights")
    print()
    
    # Load v2.4 baseline dataset
    print("📊 Loading baseline dataset...")
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded: {len(df)} matches")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Step 1: Calculate historical positions
    positions_df = calculate_historical_league_positions(df)
    
    # Step 2: Calculate bogey team scores
    bogey_df = calculate_bogey_team_scores(df, positions_df)
    
    # Step 3: Calculate context-weighted H2H
    context_h2h_df = calculate_context_weighted_h2h(df, positions_df)
    
    # Step 4: Merge all H2H features
    df_with_h2h = df.merge(
        bogey_df[['Date', 'HomeTeam', 'AwayTeam', 'bogey_team_score']], 
        on=['Date', 'HomeTeam', 'AwayTeam'], how='left'
    )
    df_with_h2h = df_with_h2h.merge(
        context_h2h_df[['Date', 'HomeTeam', 'AwayTeam', 'h2h_context_score']], 
        on=['Date', 'HomeTeam', 'AwayTeam'], how='left'
    )
    
    # Step 5: Validate H2H features
    validation_passed = validate_h2h_features(df_with_h2h)
    
    if validation_passed:
        # Save enhanced dataset
        timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
        output_path = f'/Users/maxime/Desktop/Oddsy/data/processed/v27_h2h_intelligence_{timestamp}.csv'
        
        df_with_h2h.to_csv(output_path, index=False)
        
        print(f"\n💾 v2.7 H2H Intelligence Dataset Saved:")
        print(f"Path: {output_path}")
        print(f"Features added: bogey_team_score, h2h_context_score")
        print(f"Total features: {len(df_with_h2h.columns)}")
        print(f"Dataset size: {len(df_with_h2h)} matches")
        
        # Feature summary
        print(f"\n📋 v2.7 Feature Summary:")
        print(f"v2.6 baseline features: 11 (v2.4 + optimized momentum)")
        print(f"v2.7 H2H intelligence features: 2")
        print(f"Total v2.7 features: 13")
        print(f"New features:")
        print(f"  1. bogey_team_score - Teams that underperform vs specific opponents")
        print(f"  2. h2h_context_score - Context-weighted historical performance")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Test v2.7 features with cascade model")
        print(f"2. Compare: v2.6 (55.1%) vs v2.7 (target: 58%+)")
        print(f"3. Final validation of Gemini's roadmap success")
        
        return output_path
    else:
        print("❌ H2H intelligence features validation failed!")
        return None

if __name__ == "__main__":
    main()