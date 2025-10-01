#!/usr/bin/env python3
"""
🏆 J6 FINAL PREDICTIONS - SMART CASCADE WITH DRAW LOGIC
======================================================
Version finale avec logique intelligente pour détecter draws:
- Si prob_draw > 0.35 ET (prob_h < 0.40 OU prob_a < 0.40) → DRAW
- Objectif: 3-4 draws réalistes sur matchs très équilibrés
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
    """Calculate features using the same method as before."""
    
    # Anti-leakage strict
    match_datetime = pd.to_datetime(match_date)
    data['Date'] = pd.to_datetime(data['Date'])
    historical_cutoff = data[data['Date'] < match_datetime]
    
    promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
    
    # Form calculation
    def calculate_form_diff():
        def get_team_points(team):
            team_matches = historical_cutoff[
                (historical_cutoff['HomeTeam'] == team) | (historical_cutoff['AwayTeam'] == team)
            ].tail(5)
            
            if team in promoted_teams and len(team_matches) < 3:
                return 6
            
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
    
    # Elo difference
    def calculate_elo_diff():
        home_elo = 1500
        away_elo = 1500
        
        home_matches = historical_cutoff[
            (historical_cutoff['HomeTeam'] == home_team) | (historical_cutoff['AwayTeam'] == home_team)
        ]
        away_matches = historical_cutoff[
            (historical_cutoff['HomeTeam'] == away_team) | (historical_cutoff['AwayTeam'] == away_team)
        ]
        
        if len(home_matches) > 0:
            latest_home = home_matches.iloc[-1]
            if 'elo_diff_normalized' in latest_home and not pd.isna(latest_home['elo_diff_normalized']):
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
    
    # H2H
    h2h_matches = historical_cutoff[
        ((historical_cutoff['HomeTeam'] == home_team) & (historical_cutoff['AwayTeam'] == away_team)) |
        ((historical_cutoff['HomeTeam'] == away_team) & (historical_cutoff['AwayTeam'] == home_team))
    ].tail(10)
    
    h2h_score = 0.5
    if len(h2h_matches) > 0:
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'H')) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'A'))
        ])
        h2h_score = home_wins / len(h2h_matches)
    
    # Market entropy from real odds
    j6_odds = get_real_j6_odds()
    match_key = f"{home_team} vs {away_team}"
    
    market_entropy = 0.6
    if match_key in j6_odds:
        odds = j6_odds[match_key]
        market_entropy = calculate_market_entropy(odds['H'], odds['D'], odds['A'])
    
    # Away goals
    away_goals_sum = 5.0
    if away_team not in promoted_teams:
        away_matches = historical_cutoff[historical_cutoff['AwayTeam'] == away_team].tail(5)
        if len(away_matches) > 0 and 'FTAG' in away_matches.columns:
            away_goals_sum = away_matches['FTAG'].sum()
        elif len(away_matches) > 0:
            away_goals_sum = len(away_matches) * 1.3
    
    # Features
    features = {
        'elo_diff_normalized': calculate_elo_diff(),
        'market_entropy_norm': market_entropy,
        'shots_diff_normalized': 0.5,
        'corners_diff_normalized': 0.5,
        'form_diff_normalized': calculate_form_diff(),
        'h2h_score': h2h_score,
        'matchday_normalized': 6/38,
        'home_xg_eff_10': 1.0,
        'away_xg_eff_10': 1.0,
        'away_goals_sum_5': away_goals_sum
    }
    
    return features

def load_smart_cascade_model():
    """Load Smart Cascade Model with intelligent draw logic."""
    
    # Load baseline for comparison
    try:
        baseline_model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Baseline Champion v2.3 loaded")
    except:
        baseline_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
        print("✅ Fallback Baseline model loaded")
    
    print("🔧 Reconstructing SMART Cascade Champion (intelligent draws)...")
    
    # Load training data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    X = data[features].fillna(0)
    y = data['target']
    
    train_size = min(2280, len(data) - 50)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    # Stage 1: Enhanced Draw detection
    y_binary = (y_train == 1).astype(int)
    stage1_model = RandomForestClassifier(
        n_estimators=250,  # Plus d'arbres
        max_depth=12,      # Plus profond
        min_samples_leaf=4,  # Plus de granularité
        class_weight={0: 1, 1: 3.0},  # Boost draw
        random_state=42
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
    
    # Smart Cascade with intelligent draw logic
    class SmartCascadeModel:
        def __init__(self, stage1, stage2):
            self.stage1 = stage1
            self.stage2 = stage2
            
        def predict_proba(self, X):
            stage1_proba = self.stage1.predict_proba(X)
            draw_probs = stage1_proba[:, 1] if stage1_proba.shape[1] > 1 else np.zeros(len(X))
            
            stage2_proba = self.stage2.predict_proba(X)
            if stage2_proba.shape[1] == 1:
                ha_probs = np.full((len(X), 2), [0.5, 0.5])
            else:
                ha_probs = stage2_proba
            
            results = []
            for i in range(len(X)):
                # Enhanced draw calculation
                base_draw_prob = max(draw_probs[i], 0.18)  # Plancher plus élevé
                
                # Boosts modérés mais ciblés
                entropy_boost = X[i][1] * 0.18  # Market entropy
                early_season_boost = max(0, (0.15 - X[i][6]) * 0.4)  # Early season
                
                # Balance boost si vraiment équilibré
                balance_boost = 0
                elo_diff = abs(X[i][0] - 0.5)
                form_diff = abs(X[i][4] - 0.5)
                if elo_diff < 0.15 and form_diff < 0.15:  # Très équilibré
                    balance_boost = 0.12
                
                # Probabilité draw
                p_draw = base_draw_prob + entropy_boost + early_season_boost + balance_boost
                p_draw = min(p_draw, 0.48)  # Cap raisonnable
                
                # Calcul H/A
                remaining_prob = 1 - p_draw
                prob_h = ha_probs[i][0] * remaining_prob
                prob_a = ha_probs[i][1] * remaining_prob
                
                # Normalisation
                total = prob_h + p_draw + prob_a
                results.append([prob_h/total, p_draw/total, prob_a/total])
                
            return np.array(results)
        
        def predict(self, X):
            """Custom predict with smart draw logic."""
            proba = self.predict_proba(X)
            predictions = []
            
            for i in range(len(proba)):
                prob_h, prob_d, prob_a = proba[i]
                
                # LOGIQUE INTELLIGENTE POUR DRAWS
                # Si draw prob > 35% ET aucune option H/A dominante (< 40%), alors Draw
                if prob_d > 0.35 and max(prob_h, prob_a) < 0.40:
                    predictions.append(1)  # Draw
                # Si draw prob > 38%, force Draw même si autre option plus forte
                elif prob_d > 0.38:
                    predictions.append(1)  # Draw
                # Sinon, argmax classique
                else:
                    predictions.append(np.argmax([prob_h, prob_d, prob_a]))
            
            return np.array(predictions)
    
    smart_cascade = SmartCascadeModel(stage1_model, stage2_model)
    print("✅ SMART Cascade Champion created (intelligent draws)")
    
    return baseline_model, smart_cascade

def generate_j6_smart_predictions():
    """Generate J6 predictions with smart cascade."""
    
    print("🏆 GENERATING J6 PREDICTIONS - SMART CASCADE (INTELLIGENT DRAWS)")
    print("=" * 72)
    
    # Load data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    print(f"✅ Updated data loaded: {len(data)} matches")
    
    # Load models
    baseline_model, smart_cascade = load_smart_cascade_model()
    
    # J6 matches
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
    smart_cascade_predictions = []
    
    print("\n📊 GENERATING SMART PREDICTIONS...")
    
    for home_team, away_team, match_date in j6_matches:
        print(f"\n🎯 {home_team} vs {away_team} ({match_date})")
        
        # Calculate features
        features = calculate_real_j6_features_with_odds(data, home_team, away_team, match_date)
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
        
        # Smart Cascade prediction
        smart_proba = smart_cascade.predict_proba(X)[0]
        smart_pred_num = smart_cascade.predict(X)[0]
        smart_pred = ['H', 'D', 'A'][smart_pred_num]
        smart_conf = max(smart_proba)
        
        smart_cascade_predictions.append({
            'Match': f'{home_team} vs {away_team}',
            'Date': match_date,
            'Final_Pred': smart_pred,
            'Final_Conf': round(smart_conf, 3),
            'Prob_H': round(smart_proba[0], 3),
            'Prob_D': round(smart_proba[1], 3),
            'Prob_A': round(smart_proba[2], 3),
            'Model': 'Smart_Cascade'
        })
        
        print(f"  Baseline:  {baseline_pred} ({baseline_conf:.1%}) | H:{baseline_proba[0]:.2f} D:{baseline_proba[1]:.2f} A:{baseline_proba[2]:.2f}")
        print(f"  Smart:     {smart_pred} ({smart_conf:.1%}) | H:{smart_proba[0]:.2f} D:{smart_proba[1]:.2f} A:{smart_proba[2]:.2f}")
        
        # Show draw logic
        draw_logic = ""
        if smart_proba[1] > 0.35 and max(smart_proba[0], smart_proba[2]) < 0.40:
            draw_logic = "Draw: prob>35% + no dominance"
        elif smart_proba[1] > 0.38:
            draw_logic = "Draw: prob>38% forced"
        else:
            draw_logic = "Regular argmax"
            
        print(f"  Logic: {draw_logic} | Entropy: {features['market_entropy_norm']:.3f}")
    
    # Save predictions
    with open('j6_baseline_predictions_smart.json', 'w') as f:
        json.dump(baseline_predictions, f, indent=2)
    
    with open('j6_smart_cascade_predictions.json', 'w') as f:
        json.dump(smart_cascade_predictions, f, indent=2)
    
    print(f"\n✅ Smart predictions saved:")
    print(f"   📁 j6_baseline_predictions_smart.json")
    print(f"   📁 j6_smart_cascade_predictions.json")
    
    # Summary with intelligent draw count
    print(f"\n📋 SMART PREDICTIONS SUMMARY:")
    print(f"{'Match':<25} {'Baseline':<10} {'Smart':<10} {'Draw Logic'}")
    print("-" * 70)
    
    draw_count_baseline = 0
    draw_count_smart = 0
    
    for i, (baseline, smart) in enumerate(zip(baseline_predictions, smart_cascade_predictions)):
        match = baseline['Match'][:24]
        baseline_pred = baseline['Final_Pred']
        smart_pred = smart['Final_Pred']
        
        if baseline_pred == 'D': draw_count_baseline += 1
        if smart_pred == 'D': draw_count_smart += 1
        
        logic = ""
        if smart_pred == 'D':
            if smart['Prob_D'] > 0.38:
                logic = "Force>38%"
            elif smart['Prob_D'] > 0.35:
                logic = "Smart>35%"
            else:
                logic = "Other"
        else:
            logic = "Regular"
        
        print(f"{match:<25} {baseline_pred:<10} {smart_pred:<10} {logic}")
    
    print(f"\n🎯 INTELLIGENT DRAW RESULTS:")
    print(f"   • Baseline draws: {draw_count_baseline}/10")
    print(f"   • Smart draws: {draw_count_smart}/10")
    print(f"   • Target: 3-4 draws (23-30% EPL rate)")
    print(f"   • Status: {'✅ Perfect!' if 3 <= draw_count_smart <= 4 else '✅ Good' if 2 <= draw_count_smart <= 5 else '❌ Adjust'}")
    
    return baseline_predictions, smart_cascade_predictions

if __name__ == "__main__":
    baseline_preds, smart_preds = generate_j6_smart_predictions()
    
    print("\n🏆 J6 SMART PREDICTIONS COMPLETE")
    print("=" * 45)
    print("Smart Cascade with intelligent draw logic:")
    print("- Draw if prob>35% AND no H/A dominance (<40%)")
    print("- Draw if prob>38% (forced)")
    print("- Otherwise: regular argmax")