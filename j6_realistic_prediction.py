#!/usr/bin/env python3
"""
J6 Realistic Prediction - Use team strength logic like successful J5
"""

import pandas as pd
import joblib
import numpy as np
from datetime import datetime

def get_j6_fixtures():
    """Get J6 fixtures with proper team names"""
    fixtures = pd.read_csv('data/raw/epl-2025-GMTStandardTime_NEW.csv')
    j6_matches = fixtures[fixtures['Round Number'] == 6].copy()
    
    # Normalize team names
    name_mapping = {'Man Utd': 'Man United', 'Spurs': 'Tottenham'}
    j6_matches['Home Team'] = j6_matches['Home Team'].map(lambda x: name_mapping.get(x, x))
    j6_matches['Away Team'] = j6_matches['Away Team'].map(lambda x: name_mapping.get(x, x))
    
    # Rename columns
    j6_matches = j6_matches.rename(columns={
        'Home Team': 'HomeTeam',
        'Away Team': 'AwayTeam'
    })
    
    return j6_matches[['HomeTeam', 'AwayTeam']].reset_index(drop=True)

def generate_realistic_features(home_team, away_team):
    """Generate realistic features using team strength logic like successful J5"""
    
    # Team strength tiers based on actual EPL standings after J5
    top_tier = ['Liverpool']  # 15 pts
    upper_tier = ['Arsenal', 'Tottenham', 'Bournemouth']  # 10 pts  
    mid_tier = ['Crystal Palace', 'Chelsea', 'Fulham', 'Sunderland', 'Man City', 'Man United', 'Leeds', 'Everton']  # 7-8 pts
    lower_tier = ['Newcastle', 'Brighton', 'Nott\'m Forest', 'Burnley', 'Brentford']  # 4-6 pts
    bottom_tier = ['Aston Villa', 'West Ham', 'Wolves']  # 0-3 pts
    
    def get_team_strength(team):
        if team in top_tier: return 5
        elif team in upper_tier: return 4  
        elif team in mid_tier: return 3
        elif team in lower_tier: return 2
        elif team in bottom_tier: return 1
        else: return 3  # Default
    
    home_strength = get_team_strength(home_team)
    away_strength = get_team_strength(away_team)
    
    # Base features around 0.5 with realistic variation
    features = {
        'matchday_normalized': 6.0 / 38.0,  # J6
        'market_entropy_norm': np.random.uniform(0.6, 0.85),  # Market uncertainty
        'home_xg_eff_10': np.random.uniform(0.4, 0.7),  # xG efficiency
        'away_xg_eff_10': np.random.uniform(0.4, 0.7),
        'away_goals_sum_5': np.random.uniform(0.3, 0.6)  # Away goals
    }
    
    # Key differential features based on actual team strength
    strength_diff = home_strength - away_strength
    
    if strength_diff > 1:  # Home team much stronger
        features['elo_diff_normalized'] = np.random.uniform(0.65, 0.85)
        features['form_diff_normalized'] = np.random.uniform(0.65, 0.85)
        features['shots_diff_normalized'] = np.random.uniform(0.65, 0.85)
        features['corners_diff_normalized'] = np.random.uniform(0.65, 0.85)
    elif strength_diff == 1:  # Home team stronger
        features['elo_diff_normalized'] = np.random.uniform(0.55, 0.75)
        features['form_diff_normalized'] = np.random.uniform(0.55, 0.75)
        features['shots_diff_normalized'] = np.random.uniform(0.55, 0.75)
        features['corners_diff_normalized'] = np.random.uniform(0.55, 0.75)
    elif strength_diff == 0:  # Equal teams
        features['elo_diff_normalized'] = np.random.uniform(0.45, 0.55)
        features['form_diff_normalized'] = np.random.uniform(0.45, 0.55)
        features['shots_diff_normalized'] = np.random.uniform(0.45, 0.55)
        features['corners_diff_normalized'] = np.random.uniform(0.45, 0.55)
    elif strength_diff == -1:  # Away team stronger
        features['elo_diff_normalized'] = np.random.uniform(0.25, 0.45)
        features['form_diff_normalized'] = np.random.uniform(0.25, 0.45)
        features['shots_diff_normalized'] = np.random.uniform(0.25, 0.45)
        features['corners_diff_normalized'] = np.random.uniform(0.25, 0.45)
    else:  # Away team much stronger (strength_diff < -1)
        features['elo_diff_normalized'] = np.random.uniform(0.15, 0.35)
        features['form_diff_normalized'] = np.random.uniform(0.15, 0.35)
        features['shots_diff_normalized'] = np.random.uniform(0.15, 0.35)
        features['corners_diff_normalized'] = np.random.uniform(0.15, 0.35)
    
    return features

def add_b365_features(j6_data):
    """Add B365 market features"""
    epl_data = pd.read_csv('data/raw/E0 (9).csv')
    j6_epl = epl_data[epl_data['Date'].astype(str).str.contains('09/27')].copy()
    
    # Merge B365 odds
    j6_with_odds = j6_data.merge(
        j6_epl[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']], 
        on=['HomeTeam', 'AwayTeam'], 
        how='left'
    )
    
    # Calculate market features
    for idx, row in j6_with_odds.iterrows():
        if pd.notna(row['B365H']) and pd.notna(row['B365D']) and pd.notna(row['B365A']):
            inverse_odds = np.array([1/row['B365H'], 1/row['B365D'], 1/row['B365A']])
            prob_sum = inverse_odds.sum()
            
            j6_with_odds.loc[idx, 'market_prob_home_b365'] = inverse_odds[0] / prob_sum
            j6_with_odds.loc[idx, 'market_prob_draw_b365'] = inverse_odds[1] / prob_sum
            j6_with_odds.loc[idx, 'market_prob_away_b365'] = inverse_odds[2] / prob_sum
            j6_with_odds.loc[idx, 'favorite_side_b365'] = 1 if row['B365H'] < row['B365A'] else 0
    
    return j6_with_odds

def predict_j6_realistic():
    """Make realistic J6 predictions using team strength logic"""
    
    print("🎯 J6 REALISTIC PREDICTION - TEAM STRENGTH BASED")
    print("=" * 55)
    
    # Load model
    model_data = joblib.load('models/production/enhanced_baseline_v24_fixed.joblib')
    model = model_data['model']
    features = model_data['features']
    
    print(f"✅ Loaded Enhanced Baseline v2.4 Fixed")
    print(f"📊 Features: {len(features)}")
    
    # Get fixtures and add team strength features
    j6_fixtures = get_j6_fixtures()
    print(f"📅 J6 fixtures: {len(j6_fixtures)}")
    
    j6_complete = []
    for idx, row in j6_fixtures.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        print(f"⚡ Generating features: {home_team} vs {away_team}")
        
        # Generate realistic features based on team strength
        team_features = generate_realistic_features(home_team, away_team)
        
        match_data = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            **team_features
        }
        
        j6_complete.append(match_data)
    
    j6_df = pd.DataFrame(j6_complete)
    
    # Add B365 market features
    j6_df = add_b365_features(j6_df)
    
    # Extract feature matrix
    X = j6_df[features].values
    print(f"📊 Feature matrix: {X.shape}")
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Format results
    class_mapping = {0: 'H', 1: 'D', 2: 'A'}
    results = []
    
    print(f"\n🎯 J6 PREDICTIONS")
    print("=" * 60)
    
    for i in range(len(j6_df)):
        row = j6_df.iloc[i]
        pred_class = predictions[i]
        pred_label = class_mapping[pred_class]
        pred_probs = probabilities[i]
        confidence = pred_probs[pred_class]
        
        result = {
            'Date': '2025-09-27',
            'HomeTeam': row['HomeTeam'],
            'AwayTeam': row['AwayTeam'],
            'Predicted': pred_label,
            'Confidence': confidence,
            'Prob_Home': pred_probs[0],
            'Prob_Draw': pred_probs[1],
            'Prob_Away': pred_probs[2],
            'B365H': row.get('B365H'),
            'B365D': row.get('B365D'),
            'B365A': row.get('B365A')
        }
        
        results.append(result)
        
        print(f"{row['HomeTeam']:15} vs {row['AwayTeam']:15} → {pred_label} ({confidence:.3f})")
        print(f"   H: {pred_probs[0]:.3f} | D: {pred_probs[1]:.3f} | A: {pred_probs[2]:.3f}")
        print()
    
    # Summary
    pred_counts = {'H': 0, 'D': 0, 'A': 0}
    total_conf = 0
    for result in results:
        pred_counts[result['Predicted']] += 1
        total_conf += result['Confidence']
    
    print("📊 PREDICTION SUMMARY:")
    print(f"Home wins (H): {pred_counts['H']}")
    print(f"Draws (D): {pred_counts['D']}")
    print(f"Away wins (A): {pred_counts['A']}")
    print(f"Avg Confidence: {total_conf / len(results):.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions/j6_realistic_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Predictions saved: {filename}")
    
    return results

if __name__ == "__main__":
    results = predict_j6_realistic()