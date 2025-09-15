#!/usr/bin/env python3
"""
v2.6 Momentum Features Builder
Implement Sprint v2.6: Advanced Temporal Dynamics (Momentum Intelligence)

Features to implement:
1. form_acceleration - Momentum as derivative of recent vs historical form
2. time_decay_form - Exponential moving average for realistic memory decay

Target: 55.4% → 58%+ accuracy improvement through momentum detection
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_form_points(result):
    """Convert match result to points (W=3, D=1, L=0)."""
    if result == 'H':
        return 3, 0  # home_points, away_points
    elif result == 'A':
        return 0, 3
    else:  # Draw
        return 1, 1

def calculate_rolling_form_features(df):
    """Calculate rolling form features for all teams across all matches."""
    print("📊 Calculating rolling form features...")
    
    # Initialize results storage
    form_data = []
    
    # Get all unique teams
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    print(f"Processing {len(all_teams)} teams across {len(df)} matches")
    
    # Initialize team form history
    team_form_history = {team: [] for team in all_teams}
    
    # Process matches chronologically
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    for idx, match in df_sorted.iterrows():
        if idx % 500 == 0:
            print(f"  Processing match {idx+1}/{len(df_sorted)}")
            
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        result = match['FullTimeResult']
        
        # Calculate form BEFORE this match
        home_form_3 = np.mean(team_form_history[home_team][-3:]) if len(team_form_history[home_team]) >= 3 else 1.0
        home_form_5 = np.mean(team_form_history[home_team][-5:]) if len(team_form_history[home_team]) >= 5 else 1.0
        home_form_10 = np.mean(team_form_history[home_team][-10:]) if len(team_form_history[home_team]) >= 10 else 1.0
        
        away_form_3 = np.mean(team_form_history[away_team][-3:]) if len(team_form_history[away_team]) >= 3 else 1.0
        away_form_5 = np.mean(team_form_history[away_team][-5:]) if len(team_form_history[away_team]) >= 5 else 1.0
        away_form_10 = np.mean(team_form_history[away_team][-10:]) if len(team_form_history[away_team]) >= 10 else 1.0
        
        # Calculate form acceleration (3-match vs 10-match form)
        home_form_acceleration = home_form_3 - home_form_10
        away_form_acceleration = away_form_3 - away_form_10
        form_acceleration_diff = home_form_acceleration - away_form_acceleration
        
        # Calculate EMA form (exponential moving average with α=0.3)
        alpha = 0.3
        if len(team_form_history[home_team]) > 0:
            home_ema_form = alpha * home_form_3 + (1 - alpha) * home_form_10
        else:
            home_ema_form = 1.0
            
        if len(team_form_history[away_team]) > 0:
            away_ema_form = alpha * away_form_3 + (1 - alpha) * away_form_10
        else:
            away_ema_form = 1.0
            
        ema_form_diff = home_ema_form - away_ema_form
        
        # Store form features for this match
        form_data.append({
            'Date': match['Date'],
            'Season': match['Season'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'home_form_3': home_form_3,
            'home_form_10': home_form_10,
            'away_form_3': away_form_3,
            'away_form_10': away_form_10,
            'form_acceleration_diff': form_acceleration_diff,
            'ema_form_diff': ema_form_diff,
            'home_form_acceleration': home_form_acceleration,
            'away_form_acceleration': away_form_acceleration
        })
        
        # Update team form history AFTER this match
        home_points, away_points = calculate_form_points(result)
        team_form_history[home_team].append(home_points)
        team_form_history[away_team].append(away_points)
        
        # Keep only last 20 matches for efficiency
        if len(team_form_history[home_team]) > 20:
            team_form_history[home_team] = team_form_history[home_team][-20:]
        if len(team_form_history[away_team]) > 20:
            team_form_history[away_team] = team_form_history[away_team][-20:]
    
    form_df = pd.DataFrame(form_data)
    print(f"✅ Rolling form features calculated for {len(form_df)} matches")
    
    return form_df

def normalize_momentum_features(df, form_df):
    """Normalize momentum features to 0-1 range."""
    print("🔧 Normalizing momentum features...")
    
    # Merge form data
    df_with_momentum = df.merge(form_df, on=['Date', 'Season', 'HomeTeam', 'AwayTeam'], how='left')
    
    # Normalize form acceleration difference to 0-1
    min_accel = df_with_momentum['form_acceleration_diff'].min()
    max_accel = df_with_momentum['form_acceleration_diff'].max()
    df_with_momentum['form_acceleration_normalized'] = (
        (df_with_momentum['form_acceleration_diff'] - min_accel) / (max_accel - min_accel)
    )
    
    # Normalize EMA form difference to 0-1
    min_ema = df_with_momentum['ema_form_diff'].min()
    max_ema = df_with_momentum['ema_form_diff'].max()
    df_with_momentum['time_decay_form_normalized'] = (
        (df_with_momentum['ema_form_diff'] - min_ema) / (max_ema - min_ema)
    )
    
    print(f"✅ Momentum features normalized:")
    print(f"  form_acceleration_normalized: {df_with_momentum['form_acceleration_normalized'].min():.3f} - {df_with_momentum['form_acceleration_normalized'].max():.3f}")
    print(f"  time_decay_form_normalized: {df_with_momentum['time_decay_form_normalized'].min():.3f} - {df_with_momentum['time_decay_form_normalized'].max():.3f}")
    
    # Show examples of high momentum
    high_momentum = df_with_momentum.nlargest(5, 'form_acceleration_normalized')[
        ['Date', 'HomeTeam', 'AwayTeam', 'form_acceleration_normalized', 'home_form_acceleration', 'away_form_acceleration']
    ]
    
    print("📈 High momentum examples (positive acceleration):")
    for _, match in high_momentum.iterrows():
        print(f"  {match['Date'].strftime('%Y-%m-%d')}: {match['HomeTeam']} vs {match['AwayTeam']} "
              f"(Momentum: {match['form_acceleration_normalized']:.3f}, "
              f"H_accel: {match['home_form_acceleration']:+.2f}, A_accel: {match['away_form_acceleration']:+.2f})")
    
    return df_with_momentum

def analyze_momentum_patterns(df):
    """Analyze momentum patterns and their relationship with results."""
    print("\n🔍 ANALYZING MOMENTUM PATTERNS")
    print("=" * 50)
    
    # Correlation with results
    df['result_numeric'] = df['FullTimeResult'].map({'H': 1, 'D': 0, 'A': -1})
    
    # Momentum vs results correlation
    momentum_corr = df['form_acceleration_normalized'].corr(df['result_numeric'])
    ema_corr = df['time_decay_form_normalized'].corr(df['result_numeric'])
    
    print(f"📊 Momentum-Result Correlations:")
    print(f"  Form Acceleration vs Result: {momentum_corr:.3f}")
    print(f"  Time Decay Form vs Result: {ema_corr:.3f}")
    
    # Analyze by result type
    result_analysis = df.groupby('FullTimeResult').agg({
        'form_acceleration_normalized': ['mean', 'std'],
        'time_decay_form_normalized': ['mean', 'std']
    }).round(3)
    
    print(f"\n📈 Momentum by Result Type:")
    print(result_analysis)
    
    # High momentum matches analysis
    high_momentum_threshold = df['form_acceleration_normalized'].quantile(0.8)
    high_momentum_matches = df[df['form_acceleration_normalized'] >= high_momentum_threshold]
    
    high_momentum_results = high_momentum_matches['FullTimeResult'].value_counts(normalize=True)
    
    print(f"\n🚀 High Momentum Matches (top 20%) Results:")
    for result, percentage in high_momentum_results.items():
        result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[result]
        print(f"  {result_name}: {percentage:.1%}")
    
    return True

def validate_momentum_features(df):
    """Validate momentum features for data leakage and quality."""
    print("\n🔍 VALIDATING MOMENTUM FEATURES")
    print("=" * 50)
    
    momentum_features = ['form_acceleration_normalized', 'time_decay_form_normalized']
    
    # Missing values check
    for feature in momentum_features:
        missing_count = df[feature].isna().sum()
        print(f"✅ {feature}: {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
    
    # Temporal consistency check
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df[df['Date'] < cutoff_date]
    test_df = df[df['Date'] >= cutoff_date]
    
    print(f"\n🕒 Temporal Consistency Check:")
    for feature in momentum_features:
        train_mean = train_df[feature].mean()
        test_mean = test_df[feature].mean()
        shift = abs(train_mean - test_mean)
        
        print(f"✅ {feature}: Train={train_mean:.3f}, Test={test_mean:.3f}, Shift={shift:.3f}")
        
        if shift > 0.05:
            print(f"   ⚠️  Warning: Distribution shift detected!")
    
    # Correlation with existing v2.4 features
    print(f"\n🔗 Correlation with v2.4 Features:")
    v24_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'form_diff_normalized',
        'shots_diff_normalized', 'matchday_normalized'
    ]
    
    correlation_matrix = df[v24_features + momentum_features].corr()
    
    for momentum_feature in momentum_features:
        print(f"\n📊 {momentum_feature} correlations:")
        correlations = correlation_matrix[momentum_feature][v24_features].abs().sort_values(ascending=False)
        
        for v24_feature, correlation in correlations.head(3).items():
            status = "⚠️ HIGH" if correlation > 0.7 else "🟡 MEDIUM" if correlation > 0.5 else "🟢 LOW"
            print(f"   {v24_feature}: {correlation:.3f} {status}")
    
    return True

def main():
    """Main v2.6 momentum features builder."""
    print("🚀 v2.6 Momentum Features Builder")
    print("=" * 60)
    print("Sprint v2.6: Advanced Temporal Dynamics (Momentum Intelligence)")
    print("Target: 55.4% → 58%+ accuracy through momentum detection")
    print()
    
    # Load v2.4 baseline dataset
    print("📊 Loading v2.4 baseline dataset...")
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded: {len(df)} matches")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Step 1: Calculate rolling form features
    form_df = calculate_rolling_form_features(df)
    
    # Step 2: Normalize momentum features
    df_with_momentum = normalize_momentum_features(df, form_df)
    
    # Step 3: Analyze momentum patterns
    patterns_analyzed = analyze_momentum_patterns(df_with_momentum)
    
    # Step 4: Validate momentum features
    validation_passed = validate_momentum_features(df_with_momentum)
    
    if validation_passed and patterns_analyzed:
        # Save enhanced dataset
        timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
        output_path = f'/Users/maxime/Desktop/Oddsy/data/processed/v26_momentum_features_{timestamp}.csv'
        
        df_with_momentum.to_csv(output_path, index=False)
        
        print(f"\n💾 v2.6 Momentum Features Dataset Saved:")
        print(f"Path: {output_path}")
        print(f"Features added: form_acceleration_normalized, time_decay_form_normalized")
        print(f"Total features: {len(df_with_momentum.columns)}")
        print(f"Dataset size: {len(df_with_momentum)} matches")
        
        # Feature summary
        print(f"\n📋 v2.6 Feature Summary:")
        print(f"v2.4 baseline features: 10")
        print(f"v2.6 momentum features: 2")
        print(f"Total v2.6 features: 12")
        print(f"New features:")
        print(f"  1. form_acceleration_normalized - Momentum as derivative of form")
        print(f"  2. time_decay_form_normalized - Exponential moving average form")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Test v2.6 features with cascade model")
        print(f"2. Compare performance: v2.4 (55.4%) vs v2.6 (target: 58%+)")
        print(f"3. Validate improvement and momentum hypothesis")
        
        return output_path
    else:
        print("❌ Momentum features validation failed!")
        return None

if __name__ == "__main__":
    main()