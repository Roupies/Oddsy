#!/usr/bin/env python3
"""
Simple J6 Prediction - Add J6 fixtures to training data with only B365 odds
"""

import pandas as pd
import joblib
import numpy as np
from datetime import datetime

def load_j6_fixtures():
    """Load J6 fixtures and add B365 odds"""
    fixtures = pd.read_csv('data/raw/epl-2025-GMTStandardTime_NEW.csv')
    
    # Get J6 matches (Round Number = 6)
    j6_matches = fixtures[fixtures['Round Number'] == 6].copy()
    
    # Normalize team names
    name_mapping = {'Man Utd': 'Man United', 'Spurs': 'Tottenham'}
    j6_matches['Home Team'] = j6_matches['Home Team'].map(lambda x: name_mapping.get(x, x))
    j6_matches['Away Team'] = j6_matches['Away Team'].map(lambda x: name_mapping.get(x, x))
    
    # Rename columns to match EPL format
    j6_matches = j6_matches.rename(columns={
        'Home Team': 'HomeTeam',
        'Away Team': 'AwayTeam',
        'Date': 'Date'
    })
    
    # Add B365 odds from the EPL data (the incomplete J6 entries)
    epl_data = pd.read_csv('data/raw/E0 (9).csv')
    j6_epl = epl_data[epl_data['Date'].astype(str).str.contains('09/27')].copy()
    
    # Merge odds
    j6_with_odds = j6_matches.merge(
        j6_epl[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']], 
        on=['HomeTeam', 'AwayTeam'], 
        how='left'
    )
    
    print(f"📅 Loaded {len(j6_with_odds)} J6 fixtures with B365 odds")
    return j6_with_odds

def prepare_training_data():
    """Prepare training data including J6 fixtures"""
    
    # Load existing processed data
    processed_data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    print(f"📊 Loaded {len(processed_data)} training samples")
    
    # Load J6 fixtures
    j6_fixtures = load_j6_fixtures()
    
    # Create J6 training entries with available features
    j6_training = []
    
    for _, fixture in j6_fixtures.iterrows():
        # Only features we can calculate for J6
        j6_entry = {
            'Date': '2025-09-27',  # J6 date
            'HomeTeam': fixture['HomeTeam'],
            'AwayTeam': fixture['AwayTeam'],
            'matchday_normalized': 6.0 / 38.0,  # J6 matchday
            'B365H': fixture['B365H'],
            'B365D': fixture['B365D'], 
            'B365A': fixture['B365A']
        }
        
        # Calculate B365 market features
        if pd.notna(fixture['B365H']) and pd.notna(fixture['B365D']) and pd.notna(fixture['B365A']):
            inverse_odds = np.array([1/fixture['B365H'], 1/fixture['B365D'], 1/fixture['B365A']])
            prob_sum = inverse_odds.sum()
            
            j6_entry['market_prob_home_b365'] = inverse_odds[0] / prob_sum
            j6_entry['market_prob_draw_b365'] = inverse_odds[1] / prob_sum
            j6_entry['market_prob_away_b365'] = inverse_odds[2] / prob_sum
            j6_entry['favorite_side_b365'] = 1 if fixture['B365H'] < fixture['B365A'] else 0
        
        # Set other features to neutral/median values from training data
        feature_medians = {
            'elo_diff_normalized': 0.5,
            'market_entropy_norm': processed_data['market_entropy_norm'].median(),
            'form_diff_normalized': 0.5, 
            'shots_diff_normalized': 0.5,
            'corners_diff_normalized': 0.5,
            'home_xg_eff_10': processed_data['home_xg_eff_10'].median(),
            'away_goals_sum_5': processed_data['away_goals_sum_5'].median(),
            'away_xg_eff_10': processed_data['away_xg_eff_10'].median()
        }
        
        for feature, median_val in feature_medians.items():
            j6_entry[feature] = median_val if pd.notna(median_val) else 0.5
            
        j6_training.append(j6_entry)
    
    j6_df = pd.DataFrame(j6_training)
    
    # Combine with existing training data
    extended_data = pd.concat([processed_data, j6_df], ignore_index=True)
    print(f"📊 Extended training data: {len(extended_data)} samples (+{len(j6_df)} J6)")
    
    return extended_data, j6_df

def predict_j6():
    """Make J6 predictions using extended training approach"""
    
    print("🎯 J6 SIMPLE PREDICTION - EXTENDED TRAINING")
    print("=" * 50)
    
    # Load model
    model_data = joblib.load('models/production/enhanced_baseline_v24_fixed.joblib')
    model = model_data['model']
    features = model_data['features']
    
    print(f"✅ Loaded Enhanced Baseline v2.4 Fixed")
    print(f"📊 Features: {len(features)}")
    
    # Prepare data
    extended_data, j6_df = prepare_training_data()
    
    # Extract features for J6 prediction
    X_j6 = j6_df[features].values
    print(f"📊 J6 feature matrix: {X_j6.shape}")
    
    # Make predictions
    predictions = model.predict(X_j6)
    probabilities = model.predict_proba(X_j6)
    
    # Format results
    class_mapping = {0: 'H', 1: 'D', 2: 'A'}
    results = []
    
    print(f"\n🎯 J6 PREDICTIONS")
    print("=" * 60)
    
    for i, (_, fixture) in enumerate(j6_df.iterrows()):
        pred_class = predictions[i]
        pred_label = class_mapping[pred_class]
        pred_probs = probabilities[i]
        confidence = pred_probs[pred_class]
        
        result = {
            'Date': '2025-09-27',
            'HomeTeam': fixture['HomeTeam'],
            'AwayTeam': fixture['AwayTeam'],
            'Predicted': pred_label,
            'Confidence': confidence,
            'Prob_Home': pred_probs[0],
            'Prob_Draw': pred_probs[1], 
            'Prob_Away': pred_probs[2],
            'B365H': fixture['B365H'],
            'B365D': fixture['B365D'],
            'B365A': fixture['B365A']
        }
        
        results.append(result)
        
        print(f"{fixture['HomeTeam']:15} vs {fixture['AwayTeam']:15} → {pred_label} ({confidence:.3f})")
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
    filename = f'predictions/j6_simple_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Predictions saved: {filename}")
    
    return results

if __name__ == "__main__":
    results = predict_j6()