#!/usr/bin/env python3
"""
🎯 J6 EPL Predictions - Production Ready
=======================================

Plan J6 avec vraies features pré-match, gate parité, et seuil nul raisonnable
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
        print(f"📊 Features: {features}")
        print(f"🎯 EPL Accuracy: {metadata['accuracy_epl_2025_26']:.4f}")
        print(f"🔧 Original threshold τ: {metadata['draw_threshold']:.3f}")
        
        return model, features, metadata
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def load_enhanced_dataset():
    """Load enhanced features strict temporal dataset + E0 for enrichment"""
    try:
        # Charger dataset enhanced strict
        df_enhanced = pd.read_csv("data/processed/enhanced_features_strict_temporal.csv")
        df_enhanced['Date'] = pd.to_datetime(df_enhanced['Date'])
        
        # Charger E0(14) pour enrichissement features
        df_e0 = pd.read_csv("data/raw/E0 (14).csv", encoding='utf-8-sig')
        df_e0['Date'] = pd.to_datetime(df_e0['Date'], format='%d/%m/%Y', errors='coerce')
        df_e0 = df_e0.dropna(subset=['Date'])
        
        print(f"📊 Loaded enhanced dataset: {len(df_enhanced)} matches")
        print(f"📊 Loaded E0(14) for enrichment: {len(df_e0)} matches")
        
        return df_enhanced, df_e0
    except Exception as e:
        print(f"❌ Error loading enhanced dataset: {e}")
        return None, None

def load_epl_data():
    """Legacy function - kept for compatibility"""
    df_enhanced, df_e0 = load_enhanced_dataset()
    if df_enhanced is not None:
        return df_enhanced
    return None

def enrich_with_e0_features(df_enhanced, df_e0):
    """Enrichir dataset enhanced avec features manquantes depuis E0(14)"""
    print("🔧 Enrichissement avec features E0...")
    
    # Jointure enhanced avec E0 sur [Date, HomeTeam, AwayTeam]
    df_merged = df_enhanced.merge(
        df_e0[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTHG', 'FTAG', 'FTR']],
        on=['Date', 'HomeTeam', 'AwayTeam'],
        how='left'
    )
    
    print(f"📊 Jointure enhanced-E0: {len(df_merged)} matchs")
    
    # Calculer features marché depuis B365
    df_merged['market_entropy_norm'] = df_merged.apply(lambda row: 
        calculate_market_entropy(row['B365H'], row['B365D'], row['B365A']) if pd.notna(row['B365H']) else 0.6, axis=1)
    
    df_merged['favorite_side_b365'] = df_merged.apply(lambda row:
        1.0 if pd.notna(row['B365H']) and pd.notna(row['B365A']) and row['B365H'] < row['B365A'] else 0.0, axis=1)
    
    df_merged['market_prob_away_b365'] = df_merged.apply(lambda row:
        min(max((1/row['B365A']) / ((1/row['B365H']) + (1/row['B365D']) + (1/row['B365A'])), 0.1), 0.8) if pd.notna(row['B365A']) else 0.33, axis=1)
    
    # Calculer features forme/ELO proxy pour chaque équipe
    df_merged = calculate_team_form_features(df_merged, df_e0)
    
    # Calculer away_goals_sum_5 pour chaque équipe
    df_merged = calculate_away_goals_features(df_merged, df_e0)
    
    # Ajouter matchday_normalized
    df_merged['matchday_normalized'] = df_merged['Round'] / 38.0
    
    print(f"✅ Enrichissement terminé: {len(df_merged)} matchs avec features complètes")
    return df_merged

def calculate_market_entropy(b365h, b365d, b365a):
    """Calculer entropy normalisée depuis odds B365"""
    try:
        if pd.isna(b365h) or pd.isna(b365d) or pd.isna(b365a):
            return 0.6  # Valeur neutre
        
        probs = [1/b365h, 1/b365d, 1/b365a]
        total = sum(probs)
        probs = [p/total for p in probs]  # Normaliser
        
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        return min(max(entropy / np.log(3), 0.3), 1.0)  # Normaliser [0.3, 1.0]
    except:
        return 0.6

def calculate_team_form_features(df_merged, df_e0):
    """Calculer form_diff_normalized et elo_diff_normalized"""
    print("🔧 Calcul features forme et ELO proxy...")
    
    df_merged['form_diff_normalized'] = 0.5  # Valeur neutre par défaut
    df_merged['elo_diff_normalized'] = 0.5   # Valeur neutre par défaut
    
    for idx, row in df_merged.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        
        # Récupérer historique équipes (5 derniers matchs avant cette date)
        home_matches = df_e0[
            ((df_e0['HomeTeam'] == home_team) | (df_e0['AwayTeam'] == home_team)) &
            (df_e0['Date'] < match_date)
        ].tail(5)
        
        away_matches = df_e0[
            ((df_e0['HomeTeam'] == away_team) | (df_e0['AwayTeam'] == away_team)) &
            (df_e0['Date'] < match_date)
        ].tail(5)
        
        # Calculer forme (points récents)
        home_points = calculate_recent_points(home_team, home_matches)
        away_points = calculate_recent_points(away_team, away_matches)
        form_diff = (home_points - away_points + 15) / 30.0  # Normaliser [-15,+15] → [0,1]
        df_merged.loc[idx, 'form_diff_normalized'] = min(max(form_diff, 0.1), 0.9)
        
        # Calculer ELO proxy (goal difference récent)
        home_gd = calculate_recent_goal_diff(home_team, home_matches)
        away_gd = calculate_recent_goal_diff(away_team, away_matches)
        elo_diff = (home_gd - away_gd + 10) / 20.0  # Normaliser [-10,+10] → [0,1]
        df_merged.loc[idx, 'elo_diff_normalized'] = min(max(elo_diff, 0.1), 0.9)
    
    return df_merged

def calculate_recent_points(team, matches):
    """Calculer points récents d'une équipe"""
    points = 0
    for _, match in matches.iterrows():
        if match['HomeTeam'] == team:
            if match['FTR'] == 'H':
                points += 3
            elif match['FTR'] == 'D':
                points += 1
        else:  # Away team
            if match['FTR'] == 'A':
                points += 3
            elif match['FTR'] == 'D':
                points += 1
    return points

def calculate_recent_goal_diff(team, matches):
    """Calculer goal difference récente d'une équipe"""
    gd = 0
    for _, match in matches.iterrows():
        if match['HomeTeam'] == team:
            gd += match['FTHG'] - match['FTAG']
        else:  # Away team
            gd += match['FTAG'] - match['FTHG']
    return gd

def calculate_away_goals_features(df_merged, df_e0):
    """Calculer away_goals_sum_5 pour chaque équipe"""
    print("🔧 Calcul away_goals_sum_5...")
    
    df_merged['away_goals_sum_5'] = 0  # Valeur par défaut
    
    for idx, row in df_merged.iterrows():
        away_team = row['AwayTeam']
        match_date = row['Date']
        
        # Récupérer matchs away de l'équipe avant cette date
        away_matches = df_e0[
            (df_e0['AwayTeam'] == away_team) &
            (df_e0['Date'] < match_date)
        ].tail(5)
        
        goals_sum = away_matches['FTAG'].sum() if len(away_matches) > 0 else 0
        df_merged.loc[idx, 'away_goals_sum_5'] = goals_sum
    
    return df_merged

def get_j6_fixtures():
    """Get J6 fixture list (future matches to predict)"""
    try:
        # Load fixture data to get J6 matches
        fixtures = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/raw/epl-2025-GMTStandardTime_NEW.csv")
        j6_fixtures = fixtures[fixtures['Round Number'] == 6].copy()
        
        # Standardize team names to match E0 format
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
            'Match_Number': j6_fixtures['Match Number'],
            'Venue': j6_fixtures['Location']
        })
        
        print(f"📅 J6 Fixtures loaded: {len(j6_clean)} matches")
        for _, match in j6_clean.iterrows():
            print(f"  {match['HomeTeam']} vs {match['AwayTeam']}")
        
        return j6_clean
        
    except Exception as e:
        print(f"❌ Error loading J6 fixtures: {e}")
        return None

def estimate_b365_odds_for_j6(j6_fixtures, epl_historical):
    """Estimate realistic B365 odds for J6 matches based on team strength"""
    
    print("🔧 Estimating realistic B365 odds for J6 matches...")
    
    j6_with_odds = j6_fixtures.copy()
    
    # Calculate team strength metrics from historical data
    team_strength = {}
    
    for team in set(epl_historical['HomeTeam'].unique()) | set(epl_historical['AwayTeam'].unique()):
        # Home performance
        home_matches = epl_historical[epl_historical['HomeTeam'] == team]
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        home_draws = len(home_matches[home_matches['FTR'] == 'D'])
        home_total = len(home_matches)
        
        # Away performance  
        away_matches = epl_historical[epl_historical['AwayTeam'] == team]
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        away_draws = len(away_matches[away_matches['FTR'] == 'D'])
        away_total = len(away_matches)
        
        # Overall strength (win rate + 0.5 * draw rate)
        if home_total + away_total > 0:
            win_rate = (home_wins + away_wins) / (home_total + away_total)
            draw_rate = (home_draws + away_draws) / (home_total + away_total)
            strength = win_rate + 0.5 * draw_rate
            
            # Home advantage factor
            home_advantage = (home_wins + 0.5 * home_draws) / max(home_total, 1)
            away_factor = (away_wins + 0.5 * away_draws) / max(away_total, 1)
            
            team_strength[team] = {
                'overall': strength,
                'home_factor': home_advantage, 
                'away_factor': away_factor
            }
    
    # Generate odds for each J6 fixture
    for idx, fixture in j6_with_odds.iterrows():
        home_team = fixture['HomeTeam']
        away_team = fixture['AwayTeam']
        
        home_str = team_strength.get(home_team, {'overall': 0.5, 'home_factor': 0.5, 'away_factor': 0.3})
        away_str = team_strength.get(away_team, {'overall': 0.5, 'home_factor': 0.3, 'away_factor': 0.5})
        
        # Estimate win probabilities
        home_prob = home_str['home_factor'] * 0.7 + home_str['overall'] * 0.3
        away_prob = away_str['away_factor'] * 0.7 + away_str['overall'] * 0.3
        
        # Adjust for relative strength
        strength_diff = home_str['overall'] - away_str['overall']
        home_prob += strength_diff * 0.2
        away_prob -= strength_diff * 0.2
        
        # Add home advantage (EPL ~0.4-0.45 home win rate)
        home_prob += 0.05  # Small home advantage
        
        # Normalize and add draw probability
        total_ha = home_prob + away_prob
        if total_ha > 0.85:  # Scale down if too high
            home_prob *= 0.85 / total_ha
            away_prob *= 0.85 / total_ha
        
        draw_prob = 1.0 - home_prob - away_prob
        draw_prob = max(0.15, min(0.35, draw_prob))  # Realistic draw range
        
        # Re-normalize
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total  
        away_prob /= total
        
        # Convert to odds (with bookmaker margin ~5%)
        margin = 1.05
        home_odds = margin / home_prob
        draw_odds = margin / draw_prob
        away_odds = margin / away_prob
        
        # Store estimated odds
        j6_with_odds.loc[idx, 'B365H'] = round(home_odds, 2)
        j6_with_odds.loc[idx, 'B365D'] = round(draw_odds, 2)
        j6_with_odds.loc[idx, 'B365A'] = round(away_odds, 2)
        
        print(f"  {home_team} vs {away_team}: {home_odds:.2f} / {draw_odds:.2f} / {away_odds:.2f}")
    
    return j6_with_odds

def create_j6_features_from_history(j6_fixtures, epl_historical, features):
    """Create J6 features based on team history from E0 data"""
    
    print("🔧 Building J6 features from historical data...")
    
    # First, estimate realistic B365 odds for J6 matches
    j6_with_odds = estimate_b365_odds_for_j6(j6_fixtures, epl_historical)
    
    j6_enhanced = j6_with_odds.copy()
    j6_enhanced['Date'] = pd.to_datetime(j6_enhanced['Date'])
    
    # Extract B365 features from estimated odds
    j6_with_market = extract_bet365_features(j6_enhanced)
    
    # Add B365 features extraction for historical matches
    epl_with_market = extract_bet365_features(epl_historical)
    
    # Add rolling features for historical matches with strict temporal filtering
    epl_with_rolling = calculate_rolling_features_preMatch_strict(epl_with_market)
    
    # For each J6 fixture, estimate features based on team recent performance
    for idx, fixture in j6_enhanced.iterrows():
        home_team = fixture['HomeTeam']
        away_team = fixture['AwayTeam']
        match_date = fixture['Date']
        
        print(f"🔧 {home_team} vs {away_team}")
        
        # Get recent performance for home team
        home_matches = epl_with_rolling[
            (epl_with_rolling['HomeTeam'] == home_team) | (epl_with_rolling['AwayTeam'] == home_team)
        ].tail(5)  # Last 5 matches
        
        # Get recent performance for away team
        away_matches = epl_with_rolling[
            (epl_with_rolling['HomeTeam'] == away_team) | (epl_with_rolling['AwayTeam'] == away_team)
        ].tail(5)
        
        # Calculate features for this match
        for feature in features:
            if feature == 'matchday_normalized':
                # J6 = 6/38
                j6_enhanced.loc[idx, feature] = 6.0 / 38.0
                
            elif feature in epl_with_rolling.columns:
                # Estimate feature based on teams' recent average
                home_feature_vals = []
                away_feature_vals = []
                
                # Home team values (adjust for home/away context)
                for _, match in home_matches.iterrows():
                    if not pd.isna(match[feature]):
                        if match['HomeTeam'] == home_team:
                            home_feature_vals.append(match[feature])
                        else:  # Away match, invert differential features
                            if '_diff_' in feature:
                                home_feature_vals.append(-match[feature])
                            else:
                                home_feature_vals.append(match[feature])
                
                # Away team values 
                for _, match in away_matches.iterrows():
                    if not pd.isna(match[feature]):
                        if match['AwayTeam'] == away_team:
                            away_feature_vals.append(match[feature])
                        else:  # Home match, invert differential features  
                            if '_diff_' in feature:
                                away_feature_vals.append(-match[feature])
                            else:
                                away_feature_vals.append(match[feature])
                
                # Calculate feature value
                if home_feature_vals and away_feature_vals:
                    home_avg = np.mean(home_feature_vals)
                    away_avg = np.mean(away_feature_vals)
                    
                    if '_diff_' in feature:
                        # For differential features, use home - away
                        j6_enhanced.loc[idx, feature] = home_avg - away_avg
                    else:
                        # For other features, use average
                        j6_enhanced.loc[idx, feature] = (home_avg + away_avg) / 2
                else:
                    # Fallback to historical median
                    j6_enhanced.loc[idx, feature] = epl_with_rolling[feature].median()
            else:
                # Feature not found, use neutral value
                if '_diff_' in feature or 'normalized' in feature:
                    j6_enhanced.loc[idx, feature] = 0.0  # Neutral for differentials
                else:
                    j6_enhanced.loc[idx, feature] = 0.5  # Conservative default
    
    print(f"✅ J6 features estimated from {len(epl_historical)} historical matches")
    return j6_enhanced

def extract_bet365_features(df):
    """Extract B365 market features (pré-match only)"""
    df_enhanced = df.copy()
    
    # Check if B365 odds are available
    has_b365 = all(col in df.columns for col in ['B365H', 'B365D', 'B365A'])
    
    if not has_b365:
        print("⚠️ B365 odds not available, skipping market features")
        return df_enhanced
    
    # Market probabilities (normalized)
    b365_odds = df[['B365H', 'B365D', 'B365A']].copy()
    
    # Skip matches with missing odds
    valid_odds = b365_odds.notna().all(axis=1)
    
    if valid_odds.sum() == 0:
        print("⚠️ No valid B365 odds found")
        return df_enhanced
    
    # Calculate implied probabilities for matches with valid odds
    inverse_odds = 1 / b365_odds[valid_odds]
    prob_sum = inverse_odds.sum(axis=1)
    
    # Initialize features
    df_enhanced['market_prob_home_b365'] = np.nan
    df_enhanced['market_prob_draw_b365'] = np.nan
    df_enhanced['market_prob_away_b365'] = np.nan
    df_enhanced['parity_gap_b365'] = np.nan
    df_enhanced['draw_premium_b365'] = np.nan
    df_enhanced['favorite_side_b365'] = np.nan
    
    # Set values for valid odds only
    df_enhanced.loc[valid_odds, 'market_prob_home_b365'] = inverse_odds['B365H'] / prob_sum
    df_enhanced.loc[valid_odds, 'market_prob_draw_b365'] = inverse_odds['B365D'] / prob_sum
    df_enhanced.loc[valid_odds, 'market_prob_away_b365'] = inverse_odds['B365A'] / prob_sum
    
    # Geometric features
    df_enhanced.loc[valid_odds, 'parity_gap_b365'] = abs(b365_odds.loc[valid_odds, 'B365H'] - b365_odds.loc[valid_odds, 'B365A'])
    df_enhanced.loc[valid_odds, 'draw_premium_b365'] = (
        b365_odds.loc[valid_odds, 'B365D'] / (b365_odds.loc[valid_odds, 'B365H'] + b365_odds.loc[valid_odds, 'B365A'])
    )
    df_enhanced.loc[valid_odds, 'favorite_side_b365'] = (b365_odds.loc[valid_odds, 'B365H'] < b365_odds.loc[valid_odds, 'B365A']).astype(float)
    
    print(f"✅ B365 features added for {valid_odds.sum()}/{len(df)} matches")
    return df_enhanced

def calculate_rolling_features_preMatch_strict(df):
    """Calculate rolling features with STRICT temporal validation for J6"""
    print("🔧 Calculating STRICT pre-match rolling features...")
    
    df_enhanced = df.copy()
    df_enhanced = df_enhanced.sort_values('Date').reset_index(drop=True)
    
    # Check available booking columns
    has_hbp = 'HBP' in df_enhanced.columns and not df_enhanced['HBP'].isna().all()
    has_abp = 'ABP' in df_enhanced.columns and not df_enhanced['ABP'].isna().all()
    
    if has_hbp and has_abp:
        print("✅ Using HBP/ABP columns (booking points)")
        home_booking_col, away_booking_col = 'HBP', 'ABP'
    else:
        print("⚠️ Fallback to HY/AY columns (yellow cards)")
        home_booking_col, away_booking_col = 'HY', 'AY'
    
    # Initialize rolling features
    df_enhanced['shot_accuracy_diff_roll'] = np.nan
    df_enhanced['booking_points_diff_roll'] = np.nan
    df_enhanced['corners_avg_roll'] = np.nan
    
    # Get all teams
    all_teams = set(df_enhanced['HomeTeam'].unique()) | set(df_enhanced['AwayTeam'].unique())
    
    for team in all_teams:
        # Build rolling stats by team with STRICT temporal filtering
        team_all_matches = df_enhanced[
            (df_enhanced['HomeTeam'] == team) | (df_enhanced['AwayTeam'] == team)
        ].copy()
        
        # Sort by date strictly
        team_all_matches = team_all_matches.sort_values('Date')
        
        # For each match this team played, calculate rolling features
        for match_idx in team_all_matches.index:
            match_date = df_enhanced.loc[match_idx, 'Date']
            
            # Get STRICTLY previous matches (Date < current match date)
            previous_matches = team_all_matches[
                team_all_matches['Date'] < match_date
            ].tail(5)  # Last 5 before this match
            
            if len(previous_matches) >= 3:  # k ≥ 3 minimum
                
                # Calculate team stats from previous matches
                team_shot_acc = []
                team_booking = []
                team_corners = []
                
                for _, prev_match in previous_matches.iterrows():
                    if prev_match['HomeTeam'] == team:
                        # Team was home
                        if prev_match['HS'] > 0:
                            shot_acc = prev_match['HST'] / prev_match['HS']
                        else:
                            shot_acc = 0
                        team_shot_acc.append(np.clip(shot_acc, 0, 1))
                        team_booking.append(prev_match[home_booking_col])
                        team_corners.append(prev_match['HC'])
                    else:
                        # Team was away  
                        if prev_match['AS'] > 0:
                            shot_acc = prev_match['AST'] / prev_match['AS']
                        else:
                            shot_acc = 0
                        team_shot_acc.append(np.clip(shot_acc, 0, 1))
                        team_booking.append(prev_match[away_booking_col])
                        team_corners.append(prev_match['AC'])
                
                # Store rolling averages for this team
                current_match = df_enhanced.loc[match_idx]
                
                if current_match['HomeTeam'] == team:
                    # Team is playing at home in this match
                    home_shot_avg = np.mean(team_shot_acc)
                    home_booking_avg = np.mean(team_booking)
                    home_corners_avg = np.mean(team_corners)
                    
                    # Find opponent's stats
                    opponent = current_match['AwayTeam']
                    opponent_matches = team_all_matches[
                        ((df_enhanced['HomeTeam'] == opponent) | (df_enhanced['AwayTeam'] == opponent)) &
                        (df_enhanced['Date'] < match_date)
                    ].tail(5)
                    
                    if len(opponent_matches) >= 3:
                        opp_shot_acc = []
                        opp_booking = []
                        opp_corners = []
                        
                        for _, opp_match in opponent_matches.iterrows():
                            if opp_match['HomeTeam'] == opponent:
                                if opp_match['HS'] > 0:
                                    shot_acc = opp_match['HST'] / opp_match['HS']
                                else:
                                    shot_acc = 0
                                opp_shot_acc.append(np.clip(shot_acc, 0, 1))
                                opp_booking.append(opp_match[home_booking_col])
                                opp_corners.append(opp_match['HC'])
                            else:
                                if opp_match['AS'] > 0:
                                    shot_acc = opp_match['AST'] / opp_match['AS']
                                else:
                                    shot_acc = 0
                                opp_shot_acc.append(np.clip(shot_acc, 0, 1))
                                opp_booking.append(opp_match[away_booking_col])
                                opp_corners.append(opp_match['AC'])
                        
                        if opp_shot_acc:  # If opponent has history
                            away_shot_avg = np.mean(opp_shot_acc)
                            away_booking_avg = np.mean(opp_booking)
                            away_corners_avg = np.mean(opp_corners)
                            
                            # Set differential features (home - away)
                            df_enhanced.loc[match_idx, 'shot_accuracy_diff_roll'] = home_shot_avg - away_shot_avg
                            df_enhanced.loc[match_idx, 'booking_points_diff_roll'] = home_booking_avg - away_booking_avg
                            df_enhanced.loc[match_idx, 'corners_avg_roll'] = home_corners_avg - away_corners_avg
    
    feature_coverage = df_enhanced[['shot_accuracy_diff_roll', 'booking_points_diff_roll', 'corners_avg_roll']].notna().sum()
    print(f"📊 STRICT rolling features coverage: {feature_coverage.to_dict()}")
    
    return df_enhanced

def calculate_rolling_features_preMatch(df):
    """Calculate rolling features with strict temporal validation (shift +1)"""
    print("🔧 Calculating pre-match rolling features...")
    
    df_enhanced = df.copy()
    df_enhanced = df_enhanced.sort_values('Date').reset_index(drop=True)
    
    # Check available booking columns
    has_hbp = 'HBP' in df_enhanced.columns and not df_enhanced['HBP'].isna().all()
    has_abp = 'ABP' in df_enhanced.columns and not df_enhanced['ABP'].isna().all()
    
    if has_hbp and has_abp:
        print("✅ Using HBP/ABP columns (booking points)")
        home_booking_col, away_booking_col = 'HBP', 'ABP'
    else:
        print("⚠️ Fallback to HY/AY columns (yellow cards)")
        home_booking_col, away_booking_col = 'HY', 'AY'
    
    # Initialize rolling features
    df_enhanced['shot_accuracy_diff_roll'] = np.nan
    df_enhanced['booking_points_diff_roll'] = np.nan
    df_enhanced['corners_avg_roll'] = np.nan
    
    # Get all teams
    all_teams = set(df_enhanced['HomeTeam'].unique()) | set(df_enhanced['AwayTeam'].unique())
    
    for team in all_teams:
        # Build rolling stats by team (home and away separately)
        
        # Home matches for this team
        team_home = df_enhanced[df_enhanced['HomeTeam'] == team].copy()
        
        for idx in team_home.index:
            match_date = df_enhanced.loc[idx, 'Date']
            # Strict temporal: only matches before this date
            prev_home = team_home[(team_home['Date'] < match_date) & (team_home.index < idx)].tail(5)
            
            if len(prev_home) >= 3:  # k ≥ 3 minimum
                # Shot accuracy (handle division by zero)
                home_shot_acc = (prev_home['HST'] / prev_home['HS'].replace(0, np.nan)).fillna(0).mean()
                home_shot_acc = np.clip(home_shot_acc, 0, 1)
                
                # Booking points
                home_booking = prev_home[home_booking_col].mean()
                
                # Corners
                home_corners = prev_home['HC'].mean()
                
                # Find corresponding away team stats
                away_team = df_enhanced.loc[idx, 'AwayTeam']
                away_matches = df_enhanced[df_enhanced['AwayTeam'] == away_team]
                prev_away = away_matches[(away_matches['Date'] < match_date) & (away_matches.index < idx)].tail(5)
                
                if len(prev_away) >= 3:
                    away_shot_acc = (prev_away['AST'] / prev_away['AS'].replace(0, np.nan)).fillna(0).mean()
                    away_shot_acc = np.clip(away_shot_acc, 0, 1)
                    away_booking = prev_away[away_booking_col].mean()
                    away_corners = prev_away['AC'].mean()
                    
                    # Set differential features (home - away)
                    df_enhanced.loc[idx, 'shot_accuracy_diff_roll'] = home_shot_acc - away_shot_acc
                    df_enhanced.loc[idx, 'booking_points_diff_roll'] = home_booking - away_booking
                    df_enhanced.loc[idx, 'corners_avg_roll'] = home_corners - away_corners
    
    feature_coverage = df_enhanced[['shot_accuracy_diff_roll', 'booking_points_diff_roll', 'corners_avg_roll']].notna().sum()
    print(f"📊 Rolling features coverage: {feature_coverage.to_dict()}")
    
    return df_enhanced

def apply_parity_gate_threshold_strict(model, X_features, features, j6_with_odds, original_threshold=0.25):
    """Apply parity gate ONLY with real B365 odds + reasonable draw threshold"""
    
    # Get model probabilities
    probabilities = model.predict_proba(X_features)
    
    # Reasonable threshold range (avoid 0.25 over-triggering)  
    reasonable_threshold = np.clip(original_threshold * 1.3, 0.33, 0.36)
    
    predictions = []
    adjusted_thresholds = []
    
    for i in range(len(probabilities)):
        proba = probabilities[i]
        
        # Check if this match has real B365 odds
        has_real_odds = False
        parity_condition_met = False
        
        if i < len(j6_with_odds) and all(col in j6_with_odds.columns for col in ['B365H', 'B365D', 'B365A']):
            match_odds = j6_with_odds.iloc[i]
            
            if not pd.isna(match_odds['B365H']) and not pd.isna(match_odds['B365D']) and not pd.isna(match_odds['B365A']):
                has_real_odds = True
                
                # Calculate real parity indicators from actual odds
                parity_gap = abs(match_odds['B365H'] - match_odds['B365A'])
                draw_premium = match_odds['B365D'] / (match_odds['B365H'] + match_odds['B365A'])
                
                # Apply strict parity gate conditions
                parity_condition_met = (
                    parity_gap <= 1.2 and  # Not too strong a favorite
                    draw_premium >= 0.35   # Market expects reasonable draw probability
                )
                
                print(f"  Match {i+1}: odds {match_odds['B365H']:.2f}/{match_odds['B365D']:.2f}/{match_odds['B365A']:.2f} "
                      f"→ gap={parity_gap:.2f}, premium={draw_premium:.2f}, parity={parity_condition_met}")
        
        # Decision logic
        if has_real_odds and parity_condition_met and proba[1] > reasonable_threshold:
            # Use draw threshold with real market signals
            pred = 1  # Draw
            threshold_used = reasonable_threshold
        else:
            # Use argmax H/A (no draw threshold)
            if proba[0] > proba[2]:
                pred = 0  # Home
            else:
                pred = 2  # Away
            threshold_used = None
        
        predictions.append(pred)
        adjusted_thresholds.append(threshold_used)
    
    return np.array(predictions), probabilities, adjusted_thresholds

def run_j6_predictions_production():
    """Production J6 predictions with enhanced strict dataset"""
    
    print("🎯 J6 PRODUCTION PREDICTIONS - ENHANCED STRICT")
    print("=" * 60)
    
    # Load model
    model, features, metadata = load_enhanced_v24_fixed()
    if model is None:
        return
    
    print(f"🔧 Model features required: {features}")
    
    # Load enhanced dataset + E0 for enrichment
    df_enhanced, df_e0 = load_enhanced_dataset()
    if df_enhanced is None or df_e0 is None:
        return
    
    # Enrichir avec features manquantes depuis E0
    df_enriched = enrich_with_e0_features(df_enhanced, df_e0)
    
    # Pour cette démo, utiliser les matchs disponibles comme "J6 fixtures"
    # En production réelle, on chargerait les vrais fixtures J6 depuis le calendrier
    print("\n🏆 SIMULATION J6 - Utilisation matchs disponibles comme démo")
    j6_fixtures = df_enriched.tail(5).copy()  # Prendre les 5 derniers comme simulation J6
    
    print(f"📋 Fixtures J6 simulés: {len(j6_fixtures)} matchs")
    for _, match in j6_fixtures.iterrows():
        print(f"   {match['Date'].strftime('%Y-%m-%d')} - {match['HomeTeam']} vs {match['AwayTeam']}")
    
    j6_with_features = j6_fixtures.copy()
    
    # Prepare feature matrix - only use features the model was trained on
    available_features = [f for f in features if f in j6_with_features.columns]
    missing_features = [f for f in features if f not in j6_with_features.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        print("Filling with conservative defaults...")
        
        for feature in missing_features:
            if 'normalized' in feature or '_diff_' in feature:
                j6_with_features[feature] = 0.0  # Neutral for differentials
            elif 'matchday' in feature:
                j6_with_features[feature] = 6.0 / 38.0  # J6 normalized
            else:
                j6_with_features[feature] = 0.5  # Conservative default
    
    X_j6 = j6_with_features[features]
    
    # Fill remaining NaN conservatively
    X_j6 = X_j6.fillna(0.5)
    
    print(f"📊 Feature matrix: {X_j6.shape}")
    
    # Apply STRICT parity gate + threshold (with real B365 odds)  
    print("\n🔧 Applying STRICT parity gate with real B365 odds...")
    predictions, probabilities, thresholds = apply_parity_gate_threshold_strict(
        model, X_j6, features, j6_with_features, metadata['draw_threshold']
    )
    
    # Map predictions to labels
    pred_map = {0: 'H', 1: 'D', 2: 'A'}
    j6_with_features['Predicted'] = [pred_map[p] for p in predictions]
    j6_with_features['Prob_Home'] = probabilities[:, 0]
    j6_with_features['Prob_Draw'] = probabilities[:, 1]
    j6_with_features['Prob_Away'] = probabilities[:, 2]
    j6_with_features['Confidence'] = probabilities.max(axis=1)
    j6_with_features['Draw_Threshold'] = thresholds
    
    # Display predictions
    print(f"\n🎯 J6 PREDICTIONS (Parity Gate + Threshold)")
    print("=" * 60)
    
    for i, row in j6_with_features.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        pred = row['Predicted']
        prob_h = row['Prob_Home']
        prob_d = row['Prob_Draw']
        prob_a = row['Prob_Away']
        conf = row['Confidence']
        threshold = row['Draw_Threshold']
        
        thresh_info = f"τ={threshold:.3f}" if threshold else "argmax"
        print(f"{home:15} vs {away:15} → {pred} ({conf:.3f}) [{thresh_info}]")
        print(f"   H: {prob_h:.3f} | D: {prob_d:.3f} | A: {prob_a:.3f}")
        print()
    
    # Summary
    pred_counts = j6_with_features['Predicted'].value_counts()
    print("📊 PREDICTION SUMMARY:")
    print(f"Home wins (H): {pred_counts.get('H', 0)}")
    print(f"Draws (D): {pred_counts.get('D', 0)}")
    print(f"Away wins (A): {pred_counts.get('A', 0)}")
    print(f"Avg Confidence: {j6_with_features['Confidence'].mean():.3f}")
    
    # Save clean CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"predictions/j6_production_{timestamp}.csv"
    
    # Create directory if needed
    import os
    os.makedirs('predictions', exist_ok=True)
    
    # Clean output columns
    output_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'Predicted',
        'Prob_Home', 'Prob_Draw', 'Prob_Away', 'Confidence'
    ]
    
    j6_with_features[output_cols].to_csv(output_file, index=False, float_format='%.4f')
    print(f"\n💾 Predictions saved: {output_file}")
    
    return j6_with_features

if __name__ == "__main__":
    results = run_j6_predictions_production()