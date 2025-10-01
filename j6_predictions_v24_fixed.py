#!/usr/bin/env python3
"""
🎯 J6 EPL Predictions - Enhanced Baseline v2.4 Fixed
===================================================

Script minimal pour prédictions J6 avec le nouveau champion
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def load_enhanced_v24_fixed():
    """Load Enhanced Baseline v2.4 Fixed model"""
    try:
        model_data = joblib.load('models/production/enhanced_baseline_v24_fixed.joblib')
        model = model_data['model']
        features = model_data['features']
        metadata = model_data['metadata']
        
        print(f"✅ Loaded Enhanced Baseline v2.4 Fixed")
        print(f"📊 Features: {len(features)}")
        print(f"🎯 Accuracy EPL 2025-26: {metadata['accuracy_epl_2025_26']:.4f}")
        print(f"🔧 Draw threshold τ: {metadata['draw_threshold']:.3f}")
        
        return model, features, metadata
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def predict_with_threshold_j6(model, X_test, threshold):
    """Predict using dynamic threshold for draw class"""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        predictions = np.where(
            proba[:, 1] > threshold,  # Draw probability > threshold
            1,  # Predict Draw
            np.argmax(np.column_stack([proba[:, 0], proba[:, 2]]), axis=1) * 2  # H=0, A=2
        )
        return predictions, proba
    else:
        predictions = model.predict(X_test)
        return predictions, None

def get_j6_fixtures():
    """Get real J6 fixtures from EPL 2025-26 data"""
    try:
        # Load EPL fixtures data
        epl_data = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/raw/epl-2025-GMTStandardTime_NEW.csv")
        
        # Filter for Round 6
        j6_matches = epl_data[epl_data['Round Number'] == 6].copy()
        
        # Standardize team names to match processed data format
        team_mapping = {
            'Man City': 'Manchester City',
            'Man Utd': 'Manchester United', 
            'Spurs': 'Tottenham',
            'Nott\'m Forest': 'Nottingham Forest'
        }
        
        j6_matches['Home Team'] = j6_matches['Home Team'].replace(team_mapping)
        j6_matches['Away Team'] = j6_matches['Away Team'].replace(team_mapping)
        
        # Extract date only (remove time)
        j6_matches['Match_Date'] = pd.to_datetime(j6_matches['Date']).dt.date
        
        # Create standard format
        j6_fixtures = pd.DataFrame({
            'HomeTeam': j6_matches['Home Team'],
            'AwayTeam': j6_matches['Away Team'], 
            'Date': j6_matches['Match_Date'],
            'Match_Number': j6_matches['Match Number'],
            'Venue': j6_matches['Location']
        })
        
        print(f"📅 Loaded {len(j6_fixtures)} real J6 fixtures from EPL data")
        return j6_fixtures
        
    except Exception as e:
        print(f"❌ Error loading J6 fixtures: {e}")
        return pd.DataFrame()

def load_latest_processed_data():
    """Load latest processed data for feature engineering"""
    try:
        df = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/processed/v_auto_update_20250922_093416.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"📊 Loaded processed data: {len(df)} matches")
        return df
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return None

def create_j6_features_real(j6_fixtures, processed_data, required_features):
    """Create REAL feature set for J6 predictions using processed data"""
    
    print("🔧 Creating REAL features for J6 from processed data...")
    j6_enhanced = j6_fixtures.copy()
    j6_enhanced['Date'] = pd.to_datetime(j6_enhanced['Date'])
    
    # Load raw EPL 2025-26 data to get the actual match data
    try:
        epl_raw = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/raw/E0 (9).csv")
        epl_raw['Date'] = pd.to_datetime(epl_raw['Date'], format='%d/%m/%Y', errors='coerce')
        print(f"📊 Loaded EPL raw data: {len(epl_raw)} matches")
    except Exception as e:
        print(f"❌ Error loading E0 (9).csv: {e}")
        return j6_fixtures
    
    # For each J6 fixture, try to match with processed data or estimate features
    for idx, fixture in j6_enhanced.iterrows():
        home_team = fixture['HomeTeam']
        away_team = fixture['AwayTeam']
        match_date = fixture['Date']
        
        print(f"Processing {home_team} vs {away_team}...")
        
        # Try to find exact match in processed data first
        match_in_processed = processed_data[
            (processed_data['HomeTeam'] == home_team) & 
            (processed_data['AwayTeam'] == away_team) &
            (pd.to_datetime(processed_data['Date']).dt.date == match_date)
        ]
        
        if len(match_in_processed) > 0:
            # Perfect match found in processed data
            match_data = match_in_processed.iloc[0]
            print(f"  ✅ Found in processed data")
            
            for feature in required_features:
                if feature in match_data:
                    j6_enhanced.loc[idx, feature] = match_data[feature]
                else:
                    # Feature missing, use median from processed data
                    j6_enhanced.loc[idx, feature] = processed_data[feature].median()
        
        else:
            # No exact match, need to estimate features based on teams' recent form
            print(f"  🔧 Estimating features from team history...")
            
            # Get recent matches for both teams
            home_recent = processed_data[
                (processed_data['HomeTeam'] == home_team) | (processed_data['AwayTeam'] == home_team)
            ].tail(5)
            
            away_recent = processed_data[
                (processed_data['HomeTeam'] == away_team) | (processed_data['AwayTeam'] == away_team)
            ].tail(5)
            
            for feature in required_features:
                if feature == 'matchday_normalized':
                    # J6 = 6/38
                    j6_enhanced.loc[idx, feature] = 6.0 / 38.0
                
                elif feature in processed_data.columns:
                    # Use recent team performance average
                    if len(home_recent) > 0 and len(away_recent) > 0:
                        home_avg = home_recent[feature].mean()
                        away_avg = away_recent[feature].mean()
                        
                        # For differential features, estimate based on team strength
                        if '_diff_' in feature:
                            j6_enhanced.loc[idx, feature] = (home_avg - away_avg) / 2
                        else:
                            j6_enhanced.loc[idx, feature] = (home_avg + away_avg) / 2
                    else:
                        # Fallback to overall median
                        j6_enhanced.loc[idx, feature] = processed_data[feature].median()
                else:
                    # Feature not found, use median
                    j6_enhanced.loc[idx, feature] = 0.5
    
    print(f"✅ Created REAL features for {len(j6_fixtures)} J6 matches")
    return j6_enhanced

def run_j6_predictions():
    """Run J6 predictions with Enhanced Baseline v2.4 Fixed"""
    
    print("🎯 EPL J6 PREDICTIONS - Enhanced Baseline v2.4 Fixed")
    print("=" * 60)
    
    # Load model
    model, features, metadata = load_enhanced_v24_fixed()
    if model is None:
        return
    
    # Get J6 fixtures
    j6_fixtures = get_j6_fixtures()
    print(f"📅 J6 Fixtures: {len(j6_fixtures)} matches")
    
    # Load processed data
    processed_data = load_latest_processed_data()
    
    # Create REAL features for J6
    j6_with_features = create_j6_features_real(j6_fixtures, processed_data, features)
    
    # Prepare feature matrix
    X_j6 = j6_with_features[features].fillna(0.5)
    
    # Get predictions with adjusted threshold (more realistic)
    original_threshold = metadata['draw_threshold']
    # Adjust threshold to be more conservative about draws
    adjusted_threshold = max(0.35, original_threshold * 1.4)  # At least 0.35 or 40% higher
    
    print(f"🎯 Adjusting draw threshold: {original_threshold:.3f} → {adjusted_threshold:.3f}")
    
    predictions, probabilities = predict_with_threshold_j6(model, X_j6, adjusted_threshold)
    
    # Map predictions back to labels
    pred_map = {0: 'H', 1: 'D', 2: 'A'}
    j6_with_features['Predicted'] = [pred_map[p] for p in predictions]
    
    if probabilities is not None:
        j6_with_features['Prob_Home'] = probabilities[:, 0]
        j6_with_features['Prob_Draw'] = probabilities[:, 1] 
        j6_with_features['Prob_Away'] = probabilities[:, 2]
        j6_with_features['Confidence'] = probabilities.max(axis=1)
    
    # Display predictions
    print(f"\n🎯 J6 PREDICTIONS (τ={adjusted_threshold:.3f})")
    print("=" * 60)
    
    for i, row in j6_with_features.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        pred = row['Predicted']
        
        if probabilities is not None:
            prob_h = row['Prob_Home']
            prob_d = row['Prob_Draw'] 
            prob_a = row['Prob_Away']
            conf = row['Confidence']
            
            print(f"{home:15} vs {away:15} → {pred} ({conf:.3f})")
            print(f"   H: {prob_h:.3f} | D: {prob_d:.3f} | A: {prob_a:.3f}")
        else:
            print(f"{home:15} vs {away:15} → {pred}")
        print()
    
    # Summary
    pred_counts = j6_with_features['Predicted'].value_counts()
    print("📊 PREDICTION SUMMARY:")
    print(f"Home wins (H): {pred_counts.get('H', 0)}")
    print(f"Draws (D): {pred_counts.get('D', 0)}")  
    print(f"Away wins (A): {pred_counts.get('A', 0)}")
    
    if probabilities is not None:
        avg_confidence = j6_with_features['Confidence'].mean()
        print(f"Avg Confidence: {avg_confidence:.3f}")
    
    # Save predictions
    output_file = f"predictions/j6_enhanced_v24_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Create predictions directory if it doesn't exist
    import os
    os.makedirs('predictions', exist_ok=True)
    
    # Save
    save_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Predicted']
    if probabilities is not None:
        save_cols.extend(['Prob_Home', 'Prob_Draw', 'Prob_Away', 'Confidence'])
    
    j6_with_features[save_cols].to_csv(output_file, index=False)
    print(f"\n💾 Predictions saved: {output_file}")

if __name__ == "__main__":
    run_j6_predictions()