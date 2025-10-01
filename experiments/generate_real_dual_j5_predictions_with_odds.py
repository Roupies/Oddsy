#!/usr/bin/env python3
"""
🏆 REAL J5 PREDICTIONS WITH BETTING ODDS
========================================
Generate predictions for EPL J5 using:
- ESTABLISHED PROJECT METHODS for all features
- REAL J5 betting odds for market entropy
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

def get_real_j5_odds():
    """Real J5 betting odds from Bet365."""
    return {
        'Liverpool vs Everton': {'H': 1.45, 'D': 4.75, 'A': 6.50},
        'Brighton vs Tottenham': {'H': 2.25, 'D': 3.70, 'A': 3.00},
        'Burnley vs Nott\'m Forest': {'H': 3.50, 'D': 3.30, 'A': 2.15},
        'West Ham vs Crystal Palace': {'H': 2.87, 'D': 3.50, 'A': 2.37},
        'Wolves vs Leeds': {'H': 2.70, 'D': 3.20, 'A': 2.75},
        'Man United vs Chelsea': {'H': 2.60, 'D': 3.80, 'A': 2.45},
        'Fulham vs Brentford': {'H': 2.00, 'D': 3.40, 'A': 3.80},
        'Bournemouth vs Newcastle': {'H': 2.35, 'D': 3.50, 'A': 2.87},
        'Sunderland vs Aston Villa': {'H': 3.40, 'D': 3.50, 'A': 2.10},
        'Arsenal vs Man City': {'H': 1.85, 'D': 3.90, 'A': 3.80}
    }

def calculate_real_j5_features_with_odds(data, home_team, away_team, match_date):
    """
    Calculate all 10 features for J5 prediction using ESTABLISHED PROJECT METHODS.
    Now includes REAL J5 betting odds for accurate market entropy.
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
            
            if team in promoted_teams:
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
    
    # Elo difference calculation
    def calculate_elo_diff():
        # Get latest Elo for each team from historical data
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
    
    # REAL MARKET ENTROPY from J5 betting odds
    j5_odds = get_real_j5_odds()
    match_key = f"{home_team} vs {away_team}"
    
    market_entropy = 0.6  # Fallback
    if match_key in j5_odds:
        odds = j5_odds[match_key]
        market_entropy = calculate_market_entropy(odds['H'], odds['D'], odds['A'])
        print(f"✅ Real odds for {match_key}: {odds} → entropy {market_entropy:.3f}")
    else:
        print(f"⚠️ No real odds found for {match_key}, using fallback")
    
    # Away goals sum (established method)
    away_goals_sum = 5.0  # Neutral for promoted teams
    if away_team not in promoted_teams:
        away_matches = historical_cutoff[historical_cutoff['AwayTeam'] == away_team].tail(5)
        if len(away_matches) > 0 and 'FTAG' in away_matches.columns:
            away_goals_sum = away_matches['FTAG'].sum()
        elif len(away_matches) > 0:
            away_goals_sum = len(away_matches) * 1.3  # EPL average
    
    # Matchday normalized (J5 = 5/38)
    matchday_normalized = 5/38
    
    # xG efficiency (neutral early season)
    home_xg_eff = 1.0
    away_xg_eff = 1.0
    
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
        baseline_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
        print("✅ Fallback Baseline model loaded")
    
    # Cascade Champion reconstruction
    print("🔧 Reconstructing Cascade Champion...")
    
    # Load training data 
    data = pd.read_csv('data/processed/v_auto_update_20250916_110247.csv')
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    X = data[features].fillna(0)
    y = data['target']
    
    # Same split as original
    train_size = 2280
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

def generate_j5_predictions_with_odds():
    """Generate J5 predictions using real betting odds."""
    
    print("🏆 GENERATING J5 PREDICTIONS WITH REAL BETTING ODDS")
    print("=" * 60)
    
    # Load historical data
    data = pd.read_csv('data/processed/v_auto_update_20250916_110247.csv')
    print(f"✅ Historical data loaded: {len(data)} matches")
    
    # Load models
    baseline_model, cascade_model = load_models()
    
    # J5 matches
    j5_matches = [
        ('Liverpool', 'Everton', '2025-09-20'),
        ('Brighton', 'Tottenham', '2025-09-20'),
        ('Burnley', 'Nott\'m Forest', '2025-09-20'),
        ('West Ham', 'Crystal Palace', '2025-09-20'),
        ('Wolves', 'Leeds', '2025-09-20'),
        ('Man United', 'Chelsea', '2025-09-20'),
        ('Fulham', 'Brentford', '2025-09-20'),
        ('Bournemouth', 'Newcastle', '2025-09-21'),
        ('Sunderland', 'Aston Villa', '2025-09-21'),
        ('Arsenal', 'Man City', '2025-09-21')
    ]
    
    feature_names = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    baseline_predictions = []
    cascade_predictions = []
    
    print("\n📊 GENERATING PREDICTIONS...")
    
    for home_team, away_team, match_date in j5_matches:
        print(f"\n🎯 {home_team} vs {away_team} ({match_date})")
        
        # Calculate features with real odds
        features = calculate_real_j5_features_with_odds(data, home_team, away_team, match_date)
        
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
        
        print(f"  Baseline: {baseline_pred} ({baseline_conf:.1%})")
        print(f"  Cascade:  {cascade_pred} ({cascade_conf:.1%})")
        print(f"  Market Entropy: {features['market_entropy_norm']:.3f}")
    
    # Save predictions
    with open('real_baseline_j5_predictions_with_odds.json', 'w') as f:
        json.dump(baseline_predictions, f, indent=2)
    
    with open('real_cascade_j5_predictions_with_odds.json', 'w') as f:
        json.dump(cascade_predictions, f, indent=2)
    
    print(f"\n✅ Predictions saved:")
    print(f"   📁 real_baseline_j5_predictions_with_odds.json")
    print(f"   📁 real_cascade_j5_predictions_with_odds.json")
    
    return baseline_predictions, cascade_predictions

if __name__ == "__main__":
    baseline_preds, cascade_preds = generate_j5_predictions_with_odds()
    
    print("\n🏆 J5 PREDICTIONS WITH REAL BETTING ODDS COMPLETE")
    print("=" * 60)