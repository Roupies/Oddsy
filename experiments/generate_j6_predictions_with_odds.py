#!/usr/bin/env python3
"""
🏆 REAL J6 PREDICTIONS WITH BETTING ODDS
========================================
Generate predictions for EPL J6 using:
- UPDATED DATASET with J5 results integrated
- REAL J6 betting odds for market entropy  
- Baseline Champion v2.3 & Cascade Champion v2.0
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import json
from datetime import datetime

def calculate_market_entropy(h_odds, d_odds, a_odds):
    """Calculate market entropy from betting odds."""
    h_prob = 1 / h_odds
    d_prob = 1 / d_odds  
    a_prob = 1 / a_odds
    total = h_prob + d_prob + a_prob
    
    # Normalize probabilities
    h_prob /= total
    d_prob /= total
    a_prob /= total
    
    # Calculate entropy
    entropy = -(h_prob * np.log(h_prob) + d_prob * np.log(d_prob) + a_prob * np.log(a_prob))
    return entropy / np.log(3)  # Normalize to [0,1]

def get_real_j6_odds():
    """Real J6 betting odds from transcribed data."""
    return {
        'Brentford vs Man United': {'H': 3.20, 'D': 3.80, 'A': 2.05},
        'Chelsea vs Brighton': {'H': 1.75, 'D': 4.10, 'A': 4.10},
        'Crystal Palace vs Liverpool': {'H': 4.20, 'D': 3.60, 'A': 1.85},
        'Leeds vs Bournemouth': {'H': 2.80, 'D': 3.25, 'A': 2.50},
        'Man City vs Burnley': {'H': 1.18, 'D': 7.00, 'A': 13.00},
        'Nott\'m Forest vs Sunderland': {'H': 1.85, 'D': 3.60, 'A': 4.10},
        'Tottenham vs Wolves': {'H': 1.48, 'D': 4.50, 'A': 6.25},
        'Aston Villa vs Fulham': {'H': 2.35, 'D': 3.10, 'A': 3.20},
        'Newcastle vs Arsenal': {'H': 3.20, 'D': 3.50, 'A': 2.15},
        'Everton vs West Ham': {'H': 1.80, 'D': 3.50, 'A': 4.50}
    }

def calculate_real_j6_features_with_odds(data, home_team, away_team, match_date):
    """
    Calculate all 10 features for J6 prediction using ESTABLISHED PROJECT METHODS.
    Now includes REAL J6 betting odds for accurate market entropy.
    Uses UPDATED data with J5 results integrated.
    """
    
    # Anti-leakage strict: only data BEFORE match
    match_datetime = pd.to_datetime(match_date)
    # Convert data['Date'] to datetime if it's string
    data['Date'] = pd.to_datetime(data['Date'])
    historical_cutoff = data[data['Date'] < match_datetime]
    
    # Promoted teams (neutral initialization)
    promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
    
    # Real form calculation (méthode établie - 5 matchs, système 3-1-0)
    def calculate_form_diff():
        def get_team_points(team):
            team_matches = historical_cutoff[
                (historical_cutoff['HomeTeam'] == team) | (historical_cutoff['AwayTeam'] == team)
            ].tail(5)
            
            if team in promoted_teams and len(team_matches) < 3:
                return 6  # Neutral (2 wins equivalent)
            
            points = 0
            for _, match in team_matches.iterrows():
                if match['HomeTeam'] == team:
                    if match['FullTimeResult'] == 'H': points += 3
                    elif match['FullTimeResult'] == 'D': points += 1
                else:
                    if match['FullTimeResult'] == 'A': points += 3
                    elif match['FullTimeResult'] == 'D': points += 1
            return points
        
        home_points = get_team_points(home_team)
        away_points = get_team_points(away_team)
        max_points = 15
        return np.clip((home_points - away_points) / max_points + 0.5, 0, 1)
    
    # Elo difference calculation using latest data
    def calculate_elo_diff():
        home_elo = 1500  # Default
        away_elo = 1500  # Default
        
        # Find latest match for each team
        home_matches = historical_cutoff[
            (historical_cutoff['HomeTeam'] == home_team) | (historical_cutoff['AwayTeam'] == home_team)
        ]
        away_matches = historical_cutoff[
            (historical_cutoff['HomeTeam'] == away_team) | (historical_cutoff['AwayTeam'] == away_team)
        ]
        
        if len(home_matches) > 0:
            latest_home = home_matches.iloc[-1]
            if 'elo_diff_normalized' in latest_home and not pd.isna(latest_home['elo_diff_normalized']):
                # Approximate Elo from normalized diff
                if latest_home['HomeTeam'] == home_team:
                    home_elo = 1500 + (latest_home['elo_diff_normalized'] - 0.5) * 400
                else:
                    home_elo = 1500 - (latest_home['elo_diff_normalized'] - 0.5) * 400
        
        if len(away_matches) > 0:
            latest_away = away_matches.iloc[-1]
            if 'elo_diff_normalized' in latest_away and not pd.isna(latest_away['elo_diff_normalized']):
                if latest_away['HomeTeam'] == away_team:
                    away_elo = 1500 + (latest_away['elo_diff_normalized'] - 0.5) * 400
                else:
                    away_elo = 1500 - (latest_away['elo_diff_normalized'] - 0.5) * 400
        
        elo_diff = (home_elo - away_elo) / 400
        return np.clip(0.5 + elo_diff/2, 0, 1)
    
    # H2H historical (established method)
    h2h_matches = historical_cutoff[
        ((historical_cutoff['HomeTeam'] == home_team) & (historical_cutoff['AwayTeam'] == away_team)) |
        ((historical_cutoff['HomeTeam'] == away_team) & (historical_cutoff['AwayTeam'] == home_team))
    ].tail(10)
    
    h2h_score = 0.5  # Neutral default
    if len(h2h_matches) > 0:
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'H')) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'A'))
        ])
        h2h_score = home_wins / len(h2h_matches)
    
    # REAL MARKET ENTROPY from J6 betting odds
    j6_odds = get_real_j6_odds()
    match_key = f"{home_team} vs {away_team}"
    
    market_entropy = 0.6  # Fallback
    if match_key in j6_odds:
        odds = j6_odds[match_key]
        market_entropy = calculate_market_entropy(odds['H'], odds['D'], odds['A'])
        print(f"✅ Real odds for {match_key}: {odds} → entropy {market_entropy:.3f}")
    else:
        print(f"⚠️ No real odds found for {match_key}, using fallback")
    
    # Away goals sum (established method) - now with J5 data
    away_goals_sum = 5.0  # Neutral for promoted teams
    if away_team not in promoted_teams:
        away_matches = historical_cutoff[historical_cutoff['AwayTeam'] == away_team].tail(5)
        if len(away_matches) > 0 and 'FTAG' in away_matches.columns:
            away_goals_sum = away_matches['FTAG'].sum()
        elif len(away_matches) > 0:
            away_goals_sum = len(away_matches) * 1.3  # EPL average
    
    # Matchday normalized (J6 = 6/38)
    matchday_normalized = 6/38
    
    # xG efficiency (neutral early season, can be enhanced with real data)
    home_xg_eff = 1.0
    away_xg_eff = 1.0
    
    # Check if xG efficiency data available from updated dataset
    if 'home_xg_eff_10' in historical_cutoff.columns:
        home_team_matches = historical_cutoff[historical_cutoff['HomeTeam'] == home_team].tail(1)
        if len(home_team_matches) > 0 and not pd.isna(home_team_matches.iloc[-1]['home_xg_eff_10']):
            home_xg_eff = home_team_matches.iloc[-1]['home_xg_eff_10']
    
    if 'away_xg_eff_10' in historical_cutoff.columns:
        away_team_matches = historical_cutoff[historical_cutoff['AwayTeam'] == away_team].tail(1)
        if len(away_team_matches) > 0 and not pd.isna(away_team_matches.iloc[-1]['away_xg_eff_10']):
            away_xg_eff = away_team_matches.iloc[-1]['away_xg_eff_10']
    
    # Final features using established methods
    features = {
        'elo_diff_normalized': calculate_elo_diff(),
        'market_entropy_norm': market_entropy,
        'shots_diff_normalized': 0.5,  # Neutral early season
        'corners_diff_normalized': 0.5,  # Neutral early season
        'form_diff_normalized': calculate_form_diff(),
        'h2h_score': h2h_score,
        'matchday_normalized': matchday_normalized,
        'home_xg_eff_10': home_xg_eff,
        'away_xg_eff_10': away_xg_eff,
        'away_goals_sum_5': away_goals_sum
    }
    
    return features

def load_models():
    """Load Baseline Champion and reconstruct Cascade Champion."""
    
    # Baseline Champion
    try:
        baseline_model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Baseline Champion v2.3 loaded")
    except:
        try:
            baseline_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
            print("✅ Fallback Baseline model loaded")
        except:
            print("❌ Could not load Baseline model")
            return None, None
    
    # Cascade Champion reconstruction
    print("🔧 Reconstructing Cascade Champion...")
    
    # Load training data (updated with J5 results)
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    X = data[features].fillna(0)
    y = data['target']
    
    # Same split as original (but now with more data)
    train_size = min(2280, len(data) - 50)  # Leave some recent data for validation
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    # Stage 1: Draw detection
    y_binary = (y_train == 1).astype(int)
    stage1_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight={0: 1, 1: 2.5}, random_state=42
    )
    stage1_model.fit(X_train, y_binary)
    
    # Stage 2: Home/Away
    non_draw_mask = y_train != 1
    X_non_draw = X_train[non_draw_mask]
    y_non_draw = y_train[non_draw_mask]
    y_binary_ha = (y_non_draw == 2).astype(int)
    
    stage2_model = RandomForestClassifier(
        n_estimators=150, class_weight='balanced', random_state=42
    )
    stage2_model.fit(X_non_draw, y_binary_ha)
    
    # Cascade wrapper
    class CascadeModel:
        def __init__(self, stage1, stage2):
            self.stage1 = stage1
            self.stage2 = stage2
            
        def predict_proba(self, X):
            stage1_proba = self.stage1.predict_proba(X)
            draw_probs = stage1_proba[:, 1] if stage1_proba.shape[1] > 1 else np.zeros(len(X))
            
            stage2_proba = self.stage2.predict_proba(X)
            if stage2_proba.shape[1] == 1:
                # Use features to calculate dynamic H/A probabilities
                ha_probs = []
                for j in range(len(X)):
                    elo_diff = X[j][0]  # elo_diff_normalized
                    form_diff = X[j][4]  # form_diff_normalized  
                    h2h = X[j][5]       # h2h_score
                    
                    # Calculate dynamic home advantage based on features
                    home_prob = 0.5 + (elo_diff - 0.5) * 0.3 + (form_diff - 0.5) * 0.2 + (h2h - 0.5) * 0.1
                    home_prob = np.clip(home_prob, 0.3, 0.8)
                    away_prob = 1 - home_prob
                    
                    ha_probs.append([home_prob, away_prob])
                ha_probs = np.array(ha_probs)
            else:
                ha_probs = stage2_proba
            
            results = []
            for i in range(len(X)):
                p_draw = max(draw_probs[i], 0.15) + X[i][1] * 0.15  # Market entropy boost
                p_draw = min(p_draw, 0.45)
                
                if p_draw >= 0.4:
                    prob_h = ha_probs[i][0] * (1 - p_draw) * 0.7
                    prob_d = p_draw
                    prob_a = ha_probs[i][1] * (1 - p_draw) * 0.7
                else:
                    prob_h = ha_probs[i][0] * (1 - p_draw)
                    prob_d = p_draw
                    prob_a = ha_probs[i][1] * (1 - p_draw)
                
                # Normalize
                total = prob_h + prob_d + prob_a
                results.append([prob_h/total, prob_d/total, prob_a/total])
            
            return np.array(results)
    
    cascade_model = CascadeModel(stage1_model, stage2_model)
    print("✅ Cascade Champion reconstructed")
    
    return baseline_model, cascade_model

def generate_j6_predictions_with_odds():
    """Generate J6 predictions using real betting odds and updated data."""
    
    print("🏆 GENERATING J6 PREDICTIONS WITH REAL BETTING ODDS")
    print("=" * 60)
    
    # Load historical data (updated with J5 results)
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    print(f"✅ Updated data loaded: {len(data)} matches (includes J5 results)")
    
    # Load models
    baseline_model, cascade_model = load_models()
    if baseline_model is None:
        print("❌ Could not load models")
        return None, None
    
    # J6 matches (27-29 September 2025)
    j6_matches = [
        ('Brentford', 'Man United', '2025-09-27'),
        ('Chelsea', 'Brighton', '2025-09-27'),
        ('Crystal Palace', 'Liverpool', '2025-09-27'),
        ('Leeds', 'Bournemouth', '2025-09-27'),
        ('Man City', 'Burnley', '2025-09-27'),
        ('Nott\'m Forest', 'Sunderland', '2025-09-27'),
        ('Tottenham', 'Wolves', '2025-09-27'),
        ('Aston Villa', 'Fulham', '2025-09-28'),
        ('Newcastle', 'Arsenal', '2025-09-28'),
        ('Everton', 'West Ham', '2025-09-29')
    ]
    
    feature_names = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    baseline_predictions = []
    cascade_predictions = []
    
    print("\n📊 GENERATING PREDICTIONS...")
    
    for home_team, away_team, match_date in j6_matches:
        print(f"\n🎯 {home_team} vs {away_team} ({match_date})")
        
        # Calculate features with real odds and updated data
        features = calculate_real_j6_features_with_odds(data, home_team, away_team, match_date)
        
        # Prepare feature vector
        X = np.array([[features[f] for f in feature_names]])
        
        # Baseline prediction
        baseline_proba = baseline_model.predict_proba(X)[0]
        baseline_pred = ['H', 'D', 'A'][np.argmax(baseline_proba)]
        baseline_conf = max(baseline_proba)
        
        baseline_predictions.append({
            'Match': f'{home_team} vs {away_team}',
            'Date': match_date,
            'Final_Pred': baseline_pred,
            'Final_Conf': round(baseline_conf, 3),
            'Prob_H': round(baseline_proba[0], 3),
            'Prob_D': round(baseline_proba[1], 3),
            'Prob_A': round(baseline_proba[2], 3),
            'Model': 'Baseline'
        })
        
        # Cascade prediction
        cascade_proba = cascade_model.predict_proba(X)[0]
        cascade_pred = ['H', 'D', 'A'][np.argmax(cascade_proba)]
        cascade_conf = max(cascade_proba)
        
        cascade_predictions.append({
            'Match': f'{home_team} vs {away_team}',
            'Date': match_date,
            'Final_Pred': cascade_pred,
            'Final_Conf': round(cascade_conf, 3),
            'Prob_H': round(cascade_proba[0], 3),
            'Prob_D': round(cascade_proba[1], 3),
            'Prob_A': round(cascade_proba[2], 3),
            'Model': 'Cascade'
        })
        
        print(f"  Baseline: {baseline_pred} ({baseline_conf:.1%}) | H:{baseline_proba[0]:.2f} D:{baseline_proba[1]:.2f} A:{baseline_proba[2]:.2f}")
        print(f"  Cascade:  {cascade_pred} ({cascade_conf:.1%}) | H:{cascade_proba[0]:.2f} D:{cascade_proba[1]:.2f} A:{cascade_proba[2]:.2f}")
        print(f"  Market Entropy: {features['market_entropy_norm']:.3f}")
        print(f"  Form Diff: {features['form_diff_normalized']:.3f}, Elo Diff: {features['elo_diff_normalized']:.3f}")
    
    # Save predictions
    with open('j6_baseline_predictions_with_odds.json', 'w') as f:
        json.dump(baseline_predictions, f, indent=2)
    
    with open('j6_cascade_predictions_with_odds.json', 'w') as f:
        json.dump(cascade_predictions, f, indent=2)
    
    print(f"\n✅ J6 Predictions saved:")
    print(f"   📁 j6_baseline_predictions_with_odds.json")
    print(f"   📁 j6_cascade_predictions_with_odds.json")
    
    # Display summary
    print(f"\n📋 J6 PREDICTIONS SUMMARY:")
    print(f"{'Match':<25} {'Baseline':<10} {'Cascade':<10} {'Agreement'}")
    print("-" * 55)
    
    for i, (baseline, cascade) in enumerate(zip(baseline_predictions, cascade_predictions)):
        match = baseline['Match'][:24]
        baseline_pred = baseline['Final_Pred']
        cascade_pred = cascade['Final_Pred']
        agreement = "✅" if baseline_pred == cascade_pred else "❌"
        
        print(f"{match:<25} {baseline_pred:<10} {cascade_pred:<10} {agreement}")
    
    return baseline_predictions, cascade_predictions

if __name__ == "__main__":
    baseline_preds, cascade_preds = generate_j6_predictions_with_odds()
    
    print("\n🏆 J6 PREDICTIONS WITH REAL BETTING ODDS COMPLETE")
    print("=" * 60)
    print("Using updated dataset with J5 results integrated")
    print("2 Champions Architecture: Baseline v2.3 + Cascade v2.0")