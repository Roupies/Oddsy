#!/usr/bin/env python3
"""
🏆 Génération prédictions J5 avec VRAIES features historiques
Utilise les données réelles J1-J4 pour calculer les features prédictives J5
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import json
from datetime import datetime

def load_real_historical_data():
    """Charge les vraies données historiques jusqu'à J4."""
    data = pd.read_csv('data/processed/v_auto_update_20250916_110247.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Dernières données disponibles (J4)
    print(f"📈 Dataset chargé: {len(data)} matchs historiques")
    print(f"📅 Dernière date: {data['Date'].iloc[-1]}")
    
    return data

def load_epl_2025_odds():
    """Charge les cotes EPL 2025-26 pour extraire market entropy."""
    try:
        odds_data = pd.read_csv('data/raw/EPL 2025 2026.csv')
        odds_data['Date'] = pd.to_datetime(odds_data['Date'], format='%d/%m/%Y')
        print(f"📊 Cotes EPL 2025-26 chargées: {len(odds_data)} matchs")
        return odds_data
    except Exception as e:
        print(f"⚠️  Erreur chargement cotes: {e}")
        return None

def calculate_market_entropy(h_odds, d_odds, a_odds):
    """Calcule entropie market normalisée selon méthode établie."""
    try:
        h_prob = 1 / h_odds
        d_prob = 1 / d_odds
        a_prob = 1 / a_odds
        
        total = h_prob + d_prob + a_prob
        h_prob /= total
        d_prob /= total
        a_prob /= total
        
        entropy = -(h_prob * np.log(h_prob) + d_prob * np.log(d_prob) + a_prob * np.log(a_prob))
        max_entropy = np.log(3)
        
        return entropy / max_entropy
    except:
        return 0.6  # Fallback neutre

def calculate_real_j5_features(data, home_team, away_team, match_date, odds_data=None):
    """Calcule les vraies features J5 basées sur l'historique selon méthodes établies."""
    
    # Équipes promues 2025-26
    promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
    
    # Filter données historiques AVANT match (anti-leakage strict)
    match_datetime = pd.to_datetime(match_date)
    historical_cutoff = data[data['Date'] < match_datetime]
    current_season = historical_cutoff[historical_cutoff['Season'] == '2025-2026'].copy()
    
    # Calcul form différence (méthode établie - 5 derniers matchs)
    def calculate_form_diff():
        # Home team récents matchs
        home_matches = historical_cutoff[
            (historical_cutoff['HomeTeam'] == home_team) | 
            (historical_cutoff['AwayTeam'] == home_team)
        ].tail(5)
        
        home_points = 0
        for _, match in home_matches.iterrows():
            if match['HomeTeam'] == home_team:
                if match['FullTimeResult'] == 'H': home_points += 3
                elif match['FullTimeResult'] == 'D': home_points += 1
            else:  # Away
                if match['FullTimeResult'] == 'A': home_points += 3
                elif match['FullTimeResult'] == 'D': home_points += 1
        
        # Away team récents matchs
        away_matches = historical_cutoff[
            (historical_cutoff['HomeTeam'] == away_team) | 
            (historical_cutoff['AwayTeam'] == away_team)
        ].tail(5)
        
        away_points = 0
        for _, match in away_matches.iterrows():
            if match['HomeTeam'] == away_team:
                if match['FullTimeResult'] == 'H': away_points += 3
                elif match['FullTimeResult'] == 'D': away_points += 1
            else:  # Away
                if match['FullTimeResult'] == 'A': away_points += 3
                elif match['FullTimeResult'] == 'D': away_points += 1
        
        # Normalisation établie
        max_points = 15  # 5 matchs * 3 points
        if max_points == 0: return 0.5
        form_diff = (home_points - away_points) / max_points
        return np.clip(0.5 + form_diff/2, 0, 1)
    
    # Calcul Elo différence (méthode simplifiée établie)
    def calculate_elo_diff():
        # Performance récente home (10 matchs)
        home_recent = historical_cutoff[
            (historical_cutoff['HomeTeam'] == home_team) | 
            (historical_cutoff['AwayTeam'] == home_team)
        ].tail(10)
        
        home_avg_points = 1.5  # Neutre par défaut
        if len(home_recent) > 0:
            home_points = []
            for _, match in home_recent.iterrows():
                if match['HomeTeam'] == home_team:
                    if match['FullTimeResult'] == 'H': home_points.append(3)
                    elif match['FullTimeResult'] == 'D': home_points.append(1)
                    else: home_points.append(0)
                else:
                    if match['FullTimeResult'] == 'A': home_points.append(3)
                    elif match['FullTimeResult'] == 'D': home_points.append(1)
                    else: home_points.append(0)
            home_avg_points = np.mean(home_points)
        
        # Performance récente away (10 matchs)
        away_recent = historical_cutoff[
            (historical_cutoff['HomeTeam'] == away_team) | 
            (historical_cutoff['AwayTeam'] == away_team)
        ].tail(10)
        
        away_avg_points = 1.5  # Neutre par défaut
        if len(away_recent) > 0:
            away_points = []
            for _, match in away_recent.iterrows():
                if match['HomeTeam'] == away_team:
                    if match['FullTimeResult'] == 'H': away_points.append(3)
                    elif match['FullTimeResult'] == 'D': away_points.append(1)
                    else: away_points.append(0)
                else:
                    if match['FullTimeResult'] == 'A': away_points.append(3)
                    elif match['FullTimeResult'] == 'D': away_points.append(1)
                    else: away_points.append(0)
            away_avg_points = np.mean(away_points)
        
        # Conversion points -> Elo proxy
        home_elo = 1500 + (home_avg_points - 1.5) * 200
        away_elo = 1500 + (away_avg_points - 1.5) * 200
        elo_diff = (home_elo - away_elo) / 400
        return np.clip(0.5 + elo_diff/2, 0, 1)
    
    # H2H historique (toutes saisons - méthode établie)
    h2h_matches = data[
        ((data['HomeTeam'] == home_team) & (data['AwayTeam'] == away_team)) |
        ((data['HomeTeam'] == away_team) & (data['AwayTeam'] == home_team))
    ].tail(10)  # 10 derniers H2H
    
    h2h_score = 0.5  # Neutre par défaut
    if len(h2h_matches) > 0:
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'H')) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FullTimeResult'] == 'A'))
        ])
        h2h_score = home_wins / len(h2h_matches)
    
    # Market entropy depuis vraies cotes
    market_entropy = 0.6  # Fallback
    if odds_data is not None:
        match_odds = odds_data[
            (odds_data['HomeTeam'] == home_team) & 
            (odds_data['AwayTeam'] == away_team) &
            (odds_data['Date'] == match_datetime)
        ]
        if len(match_odds) > 0:
            row = match_odds.iloc[0]
            if not pd.isna(row.get('B365H')) and not pd.isna(row.get('B365D')) and not pd.isna(row.get('B365A')):
                market_entropy = calculate_market_entropy(
                    row['B365H'], row['B365D'], row['B365A']
                )
    
    # Away goals sum (méthode établie - 5 derniers matchs away)
    away_goals_sum = 5.0  # Neutre par défaut pour promus
    if away_team not in promoted_teams:
        away_matches = historical_cutoff[historical_cutoff['AwayTeam'] == away_team].tail(5)
        if len(away_matches) > 0 and 'FTAG' in away_matches.columns:
            away_goals_sum = away_matches['FTAG'].sum()
        elif len(away_matches) > 0:
            # Estimation depuis autres colonnes si FTAG pas dispo
            away_goals_sum = len(away_matches) * 1.3  # Moyenne EPL ~1.3 goals away
    
    # Matchday normalisé (J5 = 5/38)
    matchday_normalized = 5/38
    
    # xG efficiency (valeurs neutres early season)
    home_xg_eff = 1.0
    away_xg_eff = 1.0
    
    # Features finales selon méthodes établies
    features = {
        'elo_diff_normalized': calculate_elo_diff(),
        'market_entropy_norm': market_entropy,
        'shots_diff_normalized': 0.5,  # Neutre early season
        'corners_diff_normalized': 0.5,  # Neutre early season
        'form_diff_normalized': calculate_form_diff(),
        'h2h_score': h2h_score,
        'matchday_normalized': matchday_normalized,
        'home_xg_eff_10': home_xg_eff,
        'away_xg_eff_10': away_xg_eff,
        'away_goals_sum_5': away_goals_sum
    }
    
    return features

def load_models():
    """Charge les modèles Baseline et reconstruit Cascade."""
    
    # Baseline Champion
    try:
        baseline_model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Baseline Champion v2.3 chargé")
    except:
        baseline_model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
        print("✅ Fallback Baseline model chargé")
    
    # Cascade Champion reconstruit
    print("🔧 Reconstruction Cascade Champion...")
    
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
                # Use features to calculate dynamic H/A probabilities instead of fixed values
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
                    prob_h = ha_probs[i][0] * (1 - p_draw * 0.3)
                    prob_d = p_draw * 0.8
                    prob_a = ha_probs[i][1] * (1 - p_draw * 0.3)
                
                total = prob_h + prob_d + prob_a
                if total > 0:
                    prob_h /= total
                    prob_d /= total
                    prob_a /= total
                
                results.append([prob_h, prob_d, prob_a])
            
            return np.array(results)
        
        def predict(self, X):
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
    
    cascade_model = CascadeModel(stage1_model, stage2_model)
    print("✅ Cascade Champion reconstruit")
    
    return baseline_model, cascade_model

def generate_dual_predictions():
    """Génère les prédictions des deux modèles avec vraies features."""
    
    print("🏆 GÉNÉRATION PRÉDICTIONS J5 - DUAL CHAMPIONS")
    print("="*60)
    
    # Load data et models
    data = load_real_historical_data()
    odds_data = load_epl_2025_odds()
    baseline_model, cascade_model = load_models()
    
    # J5 matches officiels
    j5_matches = [
        {'Date': '2025-09-20', 'HomeTeam': 'Liverpool', 'AwayTeam': 'Everton'},
        {'Date': '2025-09-20', 'HomeTeam': 'Brighton', 'AwayTeam': 'Tottenham'},
        {'Date': '2025-09-20', 'HomeTeam': 'Burnley', 'AwayTeam': "Nott'm Forest"},
        {'Date': '2025-09-20', 'HomeTeam': 'West Ham', 'AwayTeam': 'Crystal Palace'},
        {'Date': '2025-09-20', 'HomeTeam': 'Wolves', 'AwayTeam': 'Leeds'},
        {'Date': '2025-09-20', 'HomeTeam': 'Man United', 'AwayTeam': 'Chelsea'},
        {'Date': '2025-09-20', 'HomeTeam': 'Fulham', 'AwayTeam': 'Brentford'},
        {'Date': '2025-09-21', 'HomeTeam': 'Bournemouth', 'AwayTeam': 'Newcastle'},
        {'Date': '2025-09-21', 'HomeTeam': 'Sunderland', 'AwayTeam': 'Aston Villa'},
        {'Date': '2025-09-21', 'HomeTeam': 'Arsenal', 'AwayTeam': 'Man City'}
    ]
    
    feature_names = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    baseline_predictions = []
    cascade_predictions = []
    
    for match in j5_matches:
        home_team = match['HomeTeam'] 
        away_team = match['AwayTeam']
        date = match['Date']
        
        print(f"\n🔮 {home_team} vs {away_team}")
        
        # Calcul vraies features avec méthodes établies
        features = calculate_real_j5_features(data, home_team, away_team, date, odds_data)
        X = np.array([list(features.values())])
        
        print(f"  📊 Features: elo_diff={features['elo_diff_normalized']:.3f}, form_diff={features['form_diff_normalized']:.3f}, market_entropy={features['market_entropy_norm']:.3f}")
        print(f"  🎯 H2H: {features['h2h_score']:.3f}, away_goals_sum: {features['away_goals_sum_5']:.1f}")
        
        # Prédictions Baseline
        baseline_proba = baseline_model.predict_proba(X)[0]
        baseline_pred = baseline_model.predict(X)[0]
        baseline_conf = baseline_proba[baseline_pred]
        baseline_label = ['H', 'D', 'A'][baseline_pred]
        
        print(f"  ⚡ Baseline: {baseline_label} ({baseline_conf:.1%}) | H:{baseline_proba[0]:.0%} D:{baseline_proba[1]:.0%} A:{baseline_proba[2]:.0%}")
        
        # Prédictions Cascade
        cascade_proba = cascade_model.predict_proba(X)[0]
        cascade_pred = cascade_model.predict(X)[0]
        cascade_conf = cascade_proba[cascade_pred]
        cascade_label = ['H', 'D', 'A'][cascade_pred]
        
        print(f"  🎯 Cascade:  {cascade_label} ({cascade_conf:.1%}) | H:{cascade_proba[0]:.0%} D:{cascade_proba[1]:.0%} A:{cascade_proba[2]:.0%}")
        
        # Stockage Baseline
        baseline_predictions.append({
            'Match': f"{home_team} vs {away_team}",
            'Date': date,
            'Final_Pred': baseline_label,
            'Final_Conf': baseline_conf,
            'Prob_H': baseline_proba[0],
            'Prob_D': baseline_proba[1],
            'Prob_A': baseline_proba[2],
            'Model': 'Baseline'
        })
        
        # Stockage Cascade
        cascade_predictions.append({
            'Match': f"{home_team} vs {away_team}",
            'Date': date,
            'Final_Pred': cascade_label,
            'Final_Conf': cascade_conf,
            'Prob_H': cascade_proba[0],
            'Prob_D': cascade_proba[1],
            'Prob_A': cascade_proba[2],
            'Model': 'Cascade'
        })
    
    return baseline_predictions, cascade_predictions

if __name__ == "__main__":
    baseline_preds, cascade_preds = generate_dual_predictions()
    
    print(f"\n🎯 RÉSUMÉ FINAL:")
    print("-" * 60)
    
    baseline_draws = sum(1 for p in baseline_preds if p['Final_Pred'] == 'D')
    cascade_draws = sum(1 for p in cascade_preds if p['Final_Pred'] == 'D')
    
    print(f"⚡ Baseline: {baseline_draws}/10 draws, avg conf: {np.mean([p['Final_Conf'] for p in baseline_preds]):.1%}")
    print(f"🎯 Cascade:  {cascade_draws}/10 draws, avg conf: {np.mean([p['Final_Conf'] for p in cascade_preds]):.1%}")
    
    # Sauvegarde
    with open('real_baseline_j5_predictions.json', 'w') as f:
        json.dump(baseline_preds, f, indent=2)
    with open('real_cascade_j5_predictions.json', 'w') as f:
        json.dump(cascade_preds, f, indent=2)
    
    print(f"\n💾 Prédictions sauvées:")
    print(f"   📁 real_baseline_j5_predictions.json")
    print(f"   📁 real_cascade_j5_predictions.json")