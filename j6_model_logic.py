#!/usr/bin/env python3
"""
J6 Prediction avec vraie logique du modèle
Utilise les données J1-J5 pour calculer les features exactement comme le modèle a été entraîné
"""

import pandas as pd
import joblib
import numpy as np
from datetime import datetime

def load_training_data():
    """Charge les données d'entraînement pour comprendre comment calculer les features"""
    processed_data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    print(f"📊 Données d'entraînement: {len(processed_data)} matches")
    return processed_data

def load_j1_j5_matches():
    """Charge les matches J1-J5 complétés"""
    epl_data = pd.read_csv('data/raw/E0 (9).csv')
    # Garde seulement les matches complétés (avec résultats)
    completed_matches = epl_data[epl_data['FTHG'].notna()].copy()
    print(f"📊 Matches J1-J5 complétés: {len(completed_matches)}")
    return completed_matches

def get_j6_fixtures():
    """Récupère les fixtures J6"""
    fixtures = pd.read_csv('data/raw/epl-2025-GMTStandardTime_NEW.csv')
    j6_matches = fixtures[fixtures['Round Number'] == 6].copy()
    
    # Normalise les noms d'équipes
    name_mapping = {'Man Utd': 'Man United', 'Spurs': 'Tottenham'}
    j6_matches['Home Team'] = j6_matches['Home Team'].map(lambda x: name_mapping.get(x, x))
    j6_matches['Away Team'] = j6_matches['Away Team'].map(lambda x: name_mapping.get(x, x))
    
    j6_matches = j6_matches.rename(columns={
        'Home Team': 'HomeTeam',
        'Away Team': 'AwayTeam'
    })
    
    return j6_matches[['HomeTeam', 'AwayTeam']].reset_index(drop=True)

def calculate_features_like_model(j6_fixtures, j1_j5_data, training_data):
    """Calcule les features pour J6 exactement comme le modèle a été entraîné"""
    
    print("🔧 Calcul des features avec la vraie logique du modèle...")
    
    j6_with_features = j6_fixtures.copy()
    
    # Pour chaque match J6
    for idx, match in j6_fixtures.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        print(f"📊 {home_team} vs {away_team}")
        
        # === MATCHDAY ===
        j6_with_features.loc[idx, 'matchday_normalized'] = 6.0 / 38.0
        
        # === TEAM PERFORMANCE FEATURES ===
        # Trouve les matches de chaque équipe dans J1-J5
        home_matches = j1_j5_data[
            (j1_j5_data['HomeTeam'] == home_team) | (j1_j5_data['AwayTeam'] == home_team)
        ].copy()
        
        away_matches = j1_j5_data[
            (j1_j5_data['HomeTeam'] == away_team) | (j1_j5_data['AwayTeam'] == away_team)
        ].copy()
        
        # Calcule les stats d'équipe
        home_stats = calculate_team_performance(home_matches, home_team)
        away_stats = calculate_team_performance(away_matches, away_team)
        
        # === ELO DIFF ===
        home_gd = home_stats['goals_for'] - home_stats['goals_against'] 
        away_gd = away_stats['goals_for'] - away_stats['goals_against']
        elo_raw = home_gd - away_gd
        elo_normalized = normalize_feature(elo_raw, -10, 10)  # GD range
        j6_with_features.loc[idx, 'elo_diff_normalized'] = elo_normalized
        
        # === FORM DIFF ===
        home_ppg = home_stats['points'] / max(1, home_stats['matches'])  # Points per game
        away_ppg = away_stats['points'] / max(1, away_stats['matches'])
        form_raw = home_ppg - away_ppg
        form_normalized = normalize_feature(form_raw, -3, 3)  # PPG range
        j6_with_features.loc[idx, 'form_diff_normalized'] = form_normalized
        
        # === SHOTS DIFF ===
        home_spg = home_stats['shots'] / max(1, home_stats['matches'])
        away_spg = away_stats['shots'] / max(1, away_stats['matches'])
        shots_raw = home_spg - away_spg
        shots_normalized = normalize_feature(shots_raw, -15, 15)  # Shots range
        j6_with_features.loc[idx, 'shots_diff_normalized'] = shots_normalized
        
        # === CORNERS DIFF ===
        home_cpg = home_stats['corners'] / max(1, home_stats['matches'])
        away_cpg = away_stats['corners'] / max(1, away_stats['matches'])
        corners_raw = home_cpg - away_cpg
        corners_normalized = normalize_feature(corners_raw, -8, 8)  # Corners range
        j6_with_features.loc[idx, 'corners_diff_normalized'] = corners_normalized
        
        # === XG EFFICIENCY ===
        j6_with_features.loc[idx, 'home_xg_eff_10'] = home_stats['xg_eff']
        j6_with_features.loc[idx, 'away_xg_eff_10'] = away_stats['xg_eff']
        j6_with_features.loc[idx, 'away_goals_sum_5'] = normalize_feature(away_stats['goals_for'], 0, 15)
        
        # === PLACEHOLDER POUR FEATURES DE MARCHÉ ===
        j6_with_features.loc[idx, 'market_entropy_norm'] = 0.75
        j6_with_features.loc[idx, 'favorite_side_b365'] = 0.5
        j6_with_features.loc[idx, 'market_prob_away_b365'] = 0.33
        
        print(f"  → ELO: {j6_with_features.loc[idx, 'elo_diff_normalized']:.3f}, FORM: {j6_with_features.loc[idx, 'form_diff_normalized']:.3f}")
    
    return j6_with_features

def calculate_team_performance(team_matches, team_name):
    """Calcule les performances d'une équipe à partir de ses matches"""
    
    stats = {
        'matches': len(team_matches),
        'goals_for': 0,
        'goals_against': 0, 
        'shots': 0,
        'corners': 0,
        'points': 0
    }
    
    if len(team_matches) == 0:
        return {**stats, 'xg_eff': 0.5}
    
    for _, match in team_matches.iterrows():
        is_home = match['HomeTeam'] == team_name
        
        if is_home:
            goals_for = int(match['FTHG']) if pd.notna(match['FTHG']) else 0
            goals_against = int(match['FTAG']) if pd.notna(match['FTAG']) else 0
            shots = int(match['HS']) if pd.notna(match['HS']) else 10
            corners = int(match['HC']) if pd.notna(match['HC']) else 5
        else:
            goals_for = int(match['FTAG']) if pd.notna(match['FTAG']) else 0
            goals_against = int(match['FTHG']) if pd.notna(match['FTHG']) else 0
            shots = int(match['AS']) if pd.notna(match['AS']) else 10
            corners = int(match['AC']) if pd.notna(match['AC']) else 5
        
        stats['goals_for'] += goals_for
        stats['goals_against'] += goals_against
        stats['shots'] += shots
        stats['corners'] += corners
        
        # Points
        if goals_for > goals_against:
            stats['points'] += 3
        elif goals_for == goals_against:
            stats['points'] += 1
    
    # XG Efficiency approximation
    shots_total = max(1, stats['shots'])
    xg_eff = stats['goals_for'] / shots_total
    xg_eff_normalized = normalize_feature(xg_eff, 0, 0.3)  # Typical range 0-30%
    
    return {**stats, 'xg_eff': xg_eff_normalized}

def normalize_feature(value, min_val, max_val):
    """Normalise une valeur entre 0.1 et 0.9"""
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.1, min(0.9, normalized))

def add_market_features(j6_data):
    """Ajoute les features de marché B365"""
    
    # Charge les cotes B365 pour J6
    epl_data = pd.read_csv('data/raw/E0 (9).csv')
    j6_odds = epl_data[epl_data['Date'].astype(str).str.contains('09/27')].copy()
    
    # Merge avec les cotes
    j6_complete = j6_data.merge(
        j6_odds[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']], 
        on=['HomeTeam', 'AwayTeam'], 
        how='left'
    )
    
    # Calcule les features de marché
    for idx, row in j6_complete.iterrows():
        if pd.notna(row['B365H']) and pd.notna(row['B365D']) and pd.notna(row['B365A']):
            
            # Probabilités du marché
            inverse_odds = np.array([1/row['B365H'], 1/row['B365D'], 1/row['B365A']])
            prob_sum = inverse_odds.sum()
            
            market_prob_home = inverse_odds[0] / prob_sum
            market_prob_draw = inverse_odds[1] / prob_sum  
            market_prob_away = inverse_odds[2] / prob_sum
            
            j6_complete.loc[idx, 'market_prob_home_b365'] = market_prob_home
            j6_complete.loc[idx, 'market_prob_draw_b365'] = market_prob_draw
            j6_complete.loc[idx, 'market_prob_away_b365'] = market_prob_away
            j6_complete.loc[idx, 'favorite_side_b365'] = 1 if row['B365H'] < row['B365A'] else 0
            
            # Market entropy corrigé
            entropy = -(market_prob_home * np.log(market_prob_home) + 
                       market_prob_draw * np.log(market_prob_draw) + 
                       market_prob_away * np.log(market_prob_away))
            entropy_normalized = entropy / np.log(3)  # Normalise par max entropy
            j6_complete.loc[idx, 'market_entropy_norm'] = entropy_normalized
    
    print("✅ Features de marché B365 ajoutées")
    return j6_complete

def predict_j6_with_model_logic():
    """Prédiction J6 avec la vraie logique du modèle"""
    
    print("🎯 J6 PREDICTION - VRAIE LOGIQUE DU MODÈLE")  
    print("=" * 50)
    
    # Charge le modèle
    model_data = joblib.load('models/production/enhanced_baseline_v24_fixed.joblib')
    model = model_data['model']
    features = model_data['features']
    
    print(f"✅ Enhanced Baseline v2.4 Fixed chargé")
    print(f"📊 Features: {len(features)}")
    
    # Données
    training_data = load_training_data()
    j1_j5_data = load_j1_j5_matches()
    j6_fixtures = get_j6_fixtures()
    
    print(f"📅 Fixtures J6: {len(j6_fixtures)}")
    
    # Calcule les features comme le modèle
    j6_with_features = calculate_features_like_model(j6_fixtures, j1_j5_data, training_data)
    
    # Ajoute les features de marché
    j6_complete = add_market_features(j6_with_features)
    
    # Matrice des features
    X = j6_complete[features].fillna(0.5).values  # Remplit les NaN par des valeurs neutres
    print(f"📊 Matrice des features: {X.shape}")
    
    # Prédictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Résultats
    class_mapping = {0: 'H', 1: 'D', 2: 'A'}
    results = []
    
    print(f"\n🎯 PRÉDICTIONS J6")
    print("=" * 60)
    
    for i in range(len(j6_complete)):
        row = j6_complete.iloc[i]
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
        
        # Affiche TOUTES les features pour debug complet
        print(f"{row['HomeTeam']:15} vs {row['AwayTeam']:15} → {pred_label} ({confidence:.3f})")
        print(f"   H: {pred_probs[0]:.3f} | D: {pred_probs[1]:.3f} | A: {pred_probs[2]:.3f}")
        
        # Affiche toutes les 11 features du modèle
        print(f"   Features calculées:")
        print(f"     ELO: {row.get('elo_diff_normalized', 0.5):.3f} | FORM: {row.get('form_diff_normalized', 0.5):.3f} | SHOTS: {row.get('shots_diff_normalized', 0.5):.3f}")
        print(f"     CORNERS: {row.get('corners_diff_normalized', 0.5):.3f} | HOME_XG: {row.get('home_xg_eff_10', 0.5):.3f} | AWAY_XG: {row.get('away_xg_eff_10', 0.5):.3f}")  
        print(f"     AWAY_GOALS: {row.get('away_goals_sum_5', 0.5):.3f} | ENTROPY: {row.get('market_entropy_norm', 0.75):.3f}")
        print(f"     FAVORITE: {row.get('favorite_side_b365', 0.5):.3f} | PROB_AWAY: {row.get('market_prob_away_b365', 0.33):.3f}")
        print(f"     MATCHDAY: {row.get('matchday_normalized', 0.158):.3f}")
        print()
    
    # Résumé
    pred_counts = {'H': 0, 'D': 0, 'A': 0}
    for result in results:
        pred_counts[result['Predicted']] += 1
    
    print("📊 RÉSUMÉ:")
    print(f"Victoires domicile (H): {pred_counts['H']}")
    print(f"Nuls (D): {pred_counts['D']}")
    print(f"Victoires extérieur (A): {pred_counts['A']}")
    
    # Sauvegarde
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions/j6_model_logic_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Sauvé: {filename}")
    
    return results

if __name__ == "__main__":
    results = predict_j6_with_model_logic()