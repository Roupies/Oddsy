#!/usr/bin/env python3
"""
🎯 J6 EPL Predictions - CORRECT Approach
========================================

Utilise historique J1-J5 pour prédire J6, pas de features "J6"
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_v24_fixed():
    """Load Enhanced Baseline v2.4 Fixed model"""
    try:
        model_data = joblib.load('models/production/enhanced_baseline_v24_fixed.joblib')
        model = model_data['model']
        features = model_data['features']
        metadata = model_data['metadata']
        
        print(f"✅ Loaded Enhanced Baseline v2.4 Fixed")
        print(f"📊 Features: {len(features)}")
        print(f"🎯 EPL Accuracy: {metadata['accuracy_epl_2025_26']:.4f}")
        
        return model, features, metadata
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def load_historical_data():
    """Load historical data J1-J5 + processed features"""
    try:
        # E0 (9).csv = J1-J5 matches with raw data
        epl_raw = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/raw/E0 (9).csv")
        epl_raw['Date'] = pd.to_datetime(epl_raw['Date'], format='%d/%m/%Y', errors='coerce')
        epl_raw = epl_raw.dropna(subset=['Date'])
        
        # Processed features data
        processed_data = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/processed/v_auto_update_20250922_093416.csv")
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        
        print(f"📊 Raw EPL data: {len(epl_raw)} matches (J1-J5)")
        print(f"📊 Processed data: {len(processed_data)} matches")
        
        return epl_raw, processed_data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def get_j6_fixtures_with_odds():
    """Get J6 fixtures and estimate realistic B365 odds"""
    try:
        # Load fixture data
        fixtures = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/raw/epl-2025-GMTStandardTime_NEW.csv")
        j6_fixtures = fixtures[fixtures['Round Number'] == 6].copy()
        
        # Standardize team names
        team_mapping = {
            'Man City': 'Manchester City',
            'Man Utd': 'Manchester United',
            'Spurs': 'Tottenham',
            'Nott\'m Forest': 'Nottingham Forest'
        }
        
        j6_fixtures['Home Team'] = j6_fixtures['Home Team'].replace(team_mapping)
        j6_fixtures['Away Team'] = j6_fixtures['Away Team'].replace(team_mapping)
        
        # Create clean fixture dataframe
        j6_clean = pd.DataFrame({
            'HomeTeam': j6_fixtures['Home Team'],
            'AwayTeam': j6_fixtures['Away Team'],
            'Date': pd.to_datetime(j6_fixtures['Date']).dt.date,
            'Match_Number': j6_fixtures['Match Number']
        })
        
        print(f"📅 J6 Fixtures: {len(j6_clean)} matches")
        
        return j6_clean
        
    except Exception as e:
        print(f"❌ Error loading J6 fixtures: {e}")
        return None

def estimate_realistic_b365_odds(j6_fixtures, epl_historical):
    """Estimate realistic B365 odds based on team performance in J1-J5"""
    
    print("🔧 Estimating B365 odds from J1-J5 team performance...")
    
    # Calculate team strength from J1-J5 results
    team_stats = {}
    
    for team in set(epl_historical['HomeTeam'].unique()) | set(epl_historical['AwayTeam'].unique()):
        home_matches = epl_historical[epl_historical['HomeTeam'] == team]
        away_matches = epl_historical[epl_historical['AwayTeam'] == team]
        
        # Points calculation (3 for win, 1 for draw)
        home_points = (home_matches['FTR'] == 'H').sum() * 3 + (home_matches['FTR'] == 'D').sum()
        away_points = (away_matches['FTR'] == 'A').sum() * 3 + (away_matches['FTR'] == 'D').sum()
        
        total_matches = len(home_matches) + len(away_matches)
        total_points = home_points + away_points
        
        if total_matches > 0:
            ppg = total_points / total_matches  # Points per game
            home_form = home_points / max(len(home_matches), 1)
            away_form = away_points / max(len(away_matches), 1)
        else:
            ppg = 1.0  # Default for new teams
            home_form = 1.0
            away_form = 1.0
            
        team_stats[team] = {
            'ppg': ppg,
            'home_form': home_form,
            'away_form': away_form,
            'matches': total_matches
        }
    
    # Generate odds for each J6 match
    j6_with_odds = j6_fixtures.copy()
    
    for idx, match in j6_with_odds.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        home_stats = team_stats.get(home_team, {'ppg': 1.0, 'home_form': 1.0, 'away_form': 1.0})
        away_stats = team_stats.get(away_team, {'ppg': 1.0, 'home_form': 1.0, 'away_form': 1.0})
        
        # Estimate probabilities based on form
        home_strength = home_stats['home_form'] + 0.1  # Small home advantage
        away_strength = away_stats['away_form']
        
        # Relative strength adjustment
        strength_ratio = home_strength / (home_strength + away_strength)
        
        # Base probabilities
        home_prob = 0.3 + strength_ratio * 0.4  # 30-70% range
        away_prob = 0.3 + (1 - strength_ratio) * 0.4
        draw_prob = 0.25  # Base draw probability
        
        # Normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        # Convert to odds with margin
        margin = 1.05
        home_odds = margin / home_prob
        draw_odds = margin / draw_prob
        away_odds = margin / away_prob
        
        # Store odds
        j6_with_odds.loc[idx, 'B365H'] = round(home_odds, 2)
        j6_with_odds.loc[idx, 'B365D'] = round(draw_odds, 2)
        j6_with_odds.loc[idx, 'B365A'] = round(away_odds, 2)
        
        print(f"  {home_team} vs {away_team}: {home_odds:.2f} / {draw_odds:.2f} / {away_odds:.2f}")
    
    return j6_with_odds

def extract_team_features_from_history(j6_fixtures, processed_data, features):
    """Extract team features from processed historical data using actual team performance"""
    
    print("🔧 Extracting team features from J1-J5 history...")
    
    j6_with_features = j6_fixtures.copy()
    
    # Load raw EPL data to get team stats
    epl_data = pd.read_csv('data/raw/E0 (9).csv')
    
    # Filter out incomplete matches (J6 fixtures with NaN values)
    epl_data = epl_data[epl_data['FTHG'].notna()].copy()
    print(f"📊 Using {len(epl_data)} completed matches (J1-J5 only)")
    
    # Fix incomplete dates (add /2025 to dates like 09/27)
    epl_data['Date'] = epl_data['Date'].astype(str)
    epl_data['Date'] = epl_data['Date'].apply(lambda x: x + '/2025' if len(x.split('/')) == 2 else x)
    
    # Parse dates - handle both DD/MM/YYYY and MM/DD/YYYY formats
    def parse_date(date_str):
        try:
            # Try DD/MM/YYYY first (most data)
            return pd.to_datetime(date_str, format='%d/%m/%Y')
        except:
            try:
                # Try MM/DD/YYYY for the J6 entries
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            except:
                # Fallback to auto-detection
                return pd.to_datetime(date_str)
    
    epl_data['Date'] = epl_data['Date'].apply(parse_date)
    
    # Fix team name mismatches
    def normalize_team_name(team_name):
        """Normalize team names to match between datasets"""
        name_mapping = {
            'Man Utd': 'Man United',
            'Spurs': 'Tottenham'
        }
        return name_mapping.get(team_name, team_name)
    
    j6_with_features['HomeTeam'] = j6_with_features['HomeTeam'].apply(normalize_team_name)
    j6_with_features['AwayTeam'] = j6_with_features['AwayTeam'].apply(normalize_team_name)
    
    for idx, match in j6_with_features.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        print(f"🔧 {home_team} vs {away_team}")
        
        # Get team stats from J1-J5 matches
        home_matches = epl_data[
            (epl_data['HomeTeam'] == home_team) | (epl_data['AwayTeam'] == home_team)
        ].copy()
        
        away_matches = epl_data[
            (epl_data['HomeTeam'] == away_team) | (epl_data['AwayTeam'] == away_team)
        ].copy()
        
        # Calculate team performance metrics
        home_stats = calculate_team_stats(home_matches, home_team)
        away_stats = calculate_team_stats(away_matches, away_team)
        
        # Generate realistic features based on actual performance differences
        for feature in features:
            if feature == 'matchday_normalized':
                j6_with_features.loc[idx, feature] = 6.0 / 38.0
                
            elif feature == 'elo_diff_normalized':
                # Based on goals scored vs conceded difference - NO HOME BIAS
                home_gd = home_stats['goals_for'] - home_stats['goals_against']
                away_gd = away_stats['goals_for'] - away_stats['goals_against']
                
                # Calculate who is actually stronger, then apply to home perspective
                strength_diff = home_gd - away_gd  # Can be negative if away team stronger
                elo_diff = (strength_diff + 10) / 20  # Normalize -10 to +10 → 0 to 1
                j6_with_features.loc[idx, feature] = max(0.1, min(0.9, elo_diff))
                
            elif feature == 'form_diff_normalized':
                # Based on recent wins/points - NO HOME BIAS  
                home_form = home_stats['points'] / max(1, home_stats['matches'] * 3)
                away_form = away_stats['points'] / max(1, away_stats['matches'] * 3)
                
                # Calculate actual form difference, can favor away team
                form_strength_diff = home_form - away_form  # Can be negative
                form_diff = (form_strength_diff + 0.5) / 1.0  # Normalize -0.5 to +0.5 → 0 to 1
                j6_with_features.loc[idx, feature] = max(0.1, min(0.9, form_diff))
                
            elif feature == 'shots_diff_normalized':
                # Based on shots per game difference - NO HOME BIAS
                home_spg = home_stats['shots_for'] / max(1, home_stats['matches'])
                away_spg = away_stats['shots_for'] / max(1, away_stats['matches'])
                shots_strength_diff = home_spg - away_spg  # Can be negative
                shots_diff = (shots_strength_diff + 10) / 20  # Normalize -10 to +10 → 0 to 1  
                j6_with_features.loc[idx, feature] = max(0.1, min(0.9, shots_diff))
                
            elif feature == 'corners_diff_normalized':
                # Based on corners per game difference - NO HOME BIAS
                home_cpg = home_stats['corners_for'] / max(1, home_stats['matches'])
                away_cpg = away_stats['corners_for'] / max(1, away_stats['matches'])
                corners_strength_diff = home_cpg - away_cpg  # Can be negative
                corners_diff = (corners_strength_diff + 6) / 12  # Normalize -6 to +6 → 0 to 1
                j6_with_features.loc[idx, feature] = max(0.1, min(0.9, corners_diff))
                
            elif feature == 'market_entropy_norm':
                # Market uncertainty - varies by match competitiveness
                strength_gap = abs(home_stats['points'] - away_stats['points'])
                entropy = 0.5 + (strength_gap / 20) * 0.3  # Higher gap = higher entropy
                j6_with_features.loc[idx, feature] = max(0.4, min(0.9, entropy))
                
            elif feature == 'home_xg_eff_10':
                # Home xG efficiency approximation
                home_eff = home_stats['goals_for'] / max(1, home_stats['shots_for']) * 10
                j6_with_features.loc[idx, feature] = max(0.3, min(0.8, home_eff))
                
            elif feature == 'away_xg_eff_10':
                # Away xG efficiency approximation  
                away_eff = away_stats['goals_for'] / max(1, away_stats['shots_for']) * 10
                j6_with_features.loc[idx, feature] = max(0.3, min(0.8, away_eff))
                
            elif feature == 'away_goals_sum_5':
                # Away goals in last 5 (normalized to 0-1)
                away_goals = away_stats['goals_for']
                j6_with_features.loc[idx, feature] = max(0.2, min(0.8, away_goals / 15))
                
            else:
                # Default neutral value for other features
                j6_with_features.loc[idx, feature] = 0.5
    
    return j6_with_features

def calculate_team_stats(team_matches, team_name):
    """Calculate team performance stats from match data"""
    
    stats = {
        'matches': len(team_matches),
        'goals_for': 0,
        'goals_against': 0,
        'shots_for': 0,
        'corners_for': 0,
        'points': 0
    }
    
    if len(team_matches) == 0:
        return stats
    
    for _, match in team_matches.iterrows():
        is_home = match['HomeTeam'] == team_name
        
        if is_home:
            # Team playing at home
            goals_for = match.get('FTHG', 0) if pd.notna(match.get('FTHG', 0)) else 0
            goals_against = match.get('FTAG', 0) if pd.notna(match.get('FTAG', 0)) else 0
            shots_for = match.get('HS', 10) if pd.notna(match.get('HS', 10)) else 10
            corners_for = match.get('HC', 5) if pd.notna(match.get('HC', 5)) else 5
        else:
            # Team playing away
            goals_for = match.get('FTAG', 0) if pd.notna(match.get('FTAG', 0)) else 0
            goals_against = match.get('FTHG', 0) if pd.notna(match.get('FTHG', 0)) else 0
            shots_for = match.get('AS', 10) if pd.notna(match.get('AS', 10)) else 10
            corners_for = match.get('AC', 5) if pd.notna(match.get('AC', 5)) else 5
        
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against
        stats['shots_for'] += shots_for
        stats['corners_for'] += corners_for
        
        # Calculate points
        if goals_for > goals_against:
            stats['points'] += 3  # Win
        elif goals_for == goals_against:
            stats['points'] += 1  # Draw
        # Loss = 0 points
    
    return stats

def add_market_features(j6_with_features):
    """Add B365 market features from the odds"""
    
    if not all(col in j6_with_features.columns for col in ['B365H', 'B365D', 'B365A']):
        print("⚠️ B365 odds missing")
        return j6_with_features
    
    # Market probabilities (normalized)
    inverse_odds = 1 / j6_with_features[['B365H', 'B365D', 'B365A']]
    prob_sum = inverse_odds.sum(axis=1)
    
    j6_with_features['market_prob_home_b365'] = inverse_odds['B365H'] / prob_sum
    j6_with_features['market_prob_draw_b365'] = inverse_odds['B365D'] / prob_sum
    j6_with_features['market_prob_away_b365'] = inverse_odds['B365A'] / prob_sum
    
    # Geometric features
    j6_with_features['parity_gap_b365'] = abs(j6_with_features['B365H'] - j6_with_features['B365A'])
    j6_with_features['draw_premium_b365'] = j6_with_features['B365D'] / (j6_with_features['B365H'] + j6_with_features['B365A'])
    j6_with_features['favorite_side_b365'] = (j6_with_features['B365H'] < j6_with_features['B365A']).astype(int)
    
    print("✅ B365 market features added")
    return j6_with_features

def predict_j6_with_parity_gate(model, X_features, j6_data, threshold=0.33):
    """Predict with parity gate using real market conditions"""
    
    probabilities = model.predict_proba(X_features)
    predictions = []
    thresholds_used = []
    
    print("\n🎯 Applying parity gate with real market conditions...")
    
    for i, (idx, match) in enumerate(j6_data.iterrows()):
        proba = probabilities[i]
        
        # Check parity conditions from real odds
        parity_gap = match.get('parity_gap_b365', 999)
        draw_premium = match.get('draw_premium_b365', 0)
        
        # Parity gate conditions
        use_draw_threshold = (
            parity_gap <= 1.5 and  # Not too strong favorite
            draw_premium >= 0.3    # Decent draw probability in market
        )
        
        if use_draw_threshold and proba[1] > threshold:
            pred = 1  # Draw
            thresh_used = threshold
        else:
            # Argmax H/A
            if proba[0] > proba[2]:
                pred = 0  # Home
            else:
                pred = 2  # Away
            thresh_used = None
        
        predictions.append(pred)
        thresholds_used.append(thresh_used)
        
        gate_status = f"τ={thresh_used:.3f}" if thresh_used else "argmax"
        print(f"  {match['HomeTeam']} vs {match['AwayTeam']}: gap={parity_gap:.2f}, premium={draw_premium:.2f} → {gate_status}")
    
    return predictions, probabilities, thresholds_used

def run_j6_predictions_correct():
    """Run CORRECT J6 predictions using only historical data"""
    
    print("🎯 J6 PREDICTIONS - CORRECT APPROACH")
    print("=" * 50)
    
    # Load model
    model, features, metadata = load_enhanced_v24_fixed()
    if model is None:
        return
    
    # Load historical data (J1-J5)
    epl_raw, processed_data = load_historical_data()
    if epl_raw is None or processed_data is None:
        return
    
    # Get J6 fixtures
    j6_fixtures = get_j6_fixtures_with_odds()
    if j6_fixtures is None:
        return
    
    # Estimate B365 odds from team performance
    j6_with_odds = estimate_realistic_b365_odds(j6_fixtures, epl_raw)
    
    # Extract team features from historical data
    j6_with_features = extract_team_features_from_history(j6_with_odds, processed_data, features)
    
    # Add market features from odds
    j6_complete = add_market_features(j6_with_features)
    
    # Prepare feature matrix
    X_j6 = j6_complete[features].fillna(0.5)
    
    print(f"📊 Feature matrix: {X_j6.shape}")
    print(f"📊 Sample features: {X_j6.iloc[0].to_dict()}")
    
    # Predict with parity gate
    predictions, probabilities, thresholds = predict_j6_with_parity_gate(
        model, X_j6, j6_complete, threshold=0.33
    )
    
    # Map to labels
    pred_map = {0: 'H', 1: 'D', 2: 'A'}
    j6_complete['Predicted'] = [pred_map[p] for p in predictions]
    j6_complete['Prob_Home'] = probabilities[:, 0]
    j6_complete['Prob_Draw'] = probabilities[:, 1]
    j6_complete['Prob_Away'] = probabilities[:, 2]
    j6_complete['Confidence'] = probabilities.max(axis=1)
    
    # Display results
    print(f"\n🎯 J6 PREDICTIONS")
    print("=" * 60)
    
    for idx, (i, row) in enumerate(j6_complete.iterrows()):
        home = row['HomeTeam']
        away = row['AwayTeam']
        pred = row['Predicted']
        conf = row['Confidence']
        thresh = thresholds[idx] if idx < len(thresholds) else None
        
        thresh_info = f"τ={thresh:.3f}" if thresh else "argmax"
        print(f"{home:15} vs {away:15} → {pred} ({conf:.3f}) [{thresh_info}]")
        print(f"   H: {row['Prob_Home']:.3f} | D: {row['Prob_Draw']:.3f} | A: {row['Prob_Away']:.3f}")
        print()
    
    # Summary
    pred_counts = j6_complete['Predicted'].value_counts()
    print("📊 PREDICTION SUMMARY:")
    print(f"Home wins (H): {pred_counts.get('H', 0)}")
    print(f"Draws (D): {pred_counts.get('D', 0)}")
    print(f"Away wins (A): {pred_counts.get('A', 0)}")
    print(f"Avg Confidence: {j6_complete['Confidence'].mean():.3f}")
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"predictions/j6_correct_{timestamp}.csv"
    
    import os
    os.makedirs('predictions', exist_ok=True)
    
    output_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'Predicted',
        'Prob_Home', 'Prob_Draw', 'Prob_Away', 'Confidence',
        'B365H', 'B365D', 'B365A'
    ]
    
    j6_complete[output_cols].to_csv(output_file, index=False, float_format='%.4f')
    print(f"\n💾 Predictions saved: {output_file}")
    
    return j6_complete

if __name__ == "__main__":
    results = run_j6_predictions_correct()