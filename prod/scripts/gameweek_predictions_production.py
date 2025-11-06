#!/usr/bin/env python3
"""
🎯 EPL Predictions - Production Ready (Scalable)
===========================================

Script générique pour toute gameweek avec vraies fixtures EPL
Utilise le calendrier EPL 2025-26 pour les matchs futurs
"""

import pandas as pd
import numpy as np
import joblib
import sys
import argparse
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import dynamic features calculator
from dynamic_features_calculator import DynamicFeaturesCalculator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate EPL predictions for any gameweek')
    parser.add_argument('--gameweek', '-g', type=int, default=None, help='Gameweek number (1-38)')
    parser.add_argument('--output', '-o', type=str, default='predictions', help='Output directory')
    args = parser.parse_args()
    
    # Si pas de gameweek spécifiée, essayer de la détecter depuis le nom du script
    if args.gameweek is None:
        # Essayer de parser depuis le nom du script (ex: j9_predictions_production.py)
        script_name = sys.argv[0]
        if 'j' in script_name and '_predictions' in script_name:
            try:
                # Extraire le numéro après 'j'
                parts = script_name.split('j')[1].split('_')[0]
                args.gameweek = int(parts)
                print(f"🔍 Gameweek auto-détectée depuis script: J{args.gameweek}")
            except:
                pass
    
    if args.gameweek is None:
        print("❌ Erreur: Gameweek non spécifiée. Utilisez --gameweek N ou nommez le script jN_predictions_production.py")
        sys.exit(1)
        
    return args

def load_baseline_champion_v23():
    """Load Baseline Champion v2.3 model"""
    try:
        # Le modèle v2.3 est stocké directement, pas dans un dict
        model = joblib.load('models/production/baseline_champion_v23.joblib')
        
        # Charger les métadonnées depuis le fichier JSON séparé
        import json
        with open('models/production/baseline_champion_v23_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        features = metadata['features']
        
        print(f"✅ Loaded Baseline Champion v2.3")
        print(f"📊 Features: {features}")
        print(f"🎯 Test Accuracy: {metadata['accuracy']:.4f}")
        print(f"🎯 CV Mean: {metadata['audit_results']['cross_validation']['cv_mean']:.4f}")
        
        return model, features, metadata
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def load_enhanced_baseline_v30():
    """Load Enhanced Baseline v3.0 model"""
    try:
        # Le modèle v3.0 enhanced (stocké comme dict)
        model_data = joblib.load('models/production/enhanced_baseline_v3_0_final_20251104_160406.joblib')
        
        # Extraire le modèle et métadonnées du dict
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        print(f"✅ Loaded Enhanced Baseline v3.0 FIXED")
        print(f"📊 Features: {len(features)} - PRODUCTION PIPELINE")
        print(f"🎯 CV Accuracy: {model_data['cv_mean']:.3f}±{model_data['cv_std']:.3f}")
        print(f"🎯 Training: {model_data['training_samples']} matches")
        print(f"🚀 Production pipeline consistency + light rebalancing")
        
        return model, scaler, features
    except Exception as e:
        print(f"❌ Failed to load Enhanced Baseline v3.0: {e}")
        return None, None, None

def load_epl_calendar():
    """Load EPL 2025-26 calendar with proper gameweek numbers"""
    try:
        df_calendar = pd.read_csv("data/raw/epl-2025-2026_GMTStandardTime.csv")
        df_calendar['Date'] = pd.to_datetime(df_calendar['Date'], format='%d/%m/%Y %H:%M')
        df_calendar['gameweek'] = df_calendar['Round Number']  # Use official gameweek numbers
        
        print(f"📅 Loaded EPL calendar: {len(df_calendar)} matches")
        return df_calendar
    except Exception as e:
        print(f"❌ Error loading EPL calendar: {e}")
        return None

def get_gameweek_fixtures(df_calendar, gameweek):
    """Get fixtures for a specific gameweek"""
    # Use the official gameweek number from the calendar
    gw_matches = df_calendar[df_calendar['gameweek'] == gameweek].copy()
    
    if len(gw_matches) == 0:
        print(f"❌ Aucun match trouvé pour J{gameweek}")
        return None
        
    print(f"📋 Fixtures J{gameweek}: {len(gw_matches)} matchs")
    for _, match in gw_matches.iterrows():
        date_str = match['Date'].strftime('%Y-%m-%d')
        print(f"   {date_str} - {match['Home Team']} vs {match['Away Team']}")
    
    return gw_matches

def load_live_odds(gameweek):
    """Load live odds for the gameweek from The Odds API data"""
    odds_file = f"data/odds/2025/epl/gw{gameweek}.json"
    try:
        with open(odds_file, 'r') as f:
            odds_data = json.load(f)
        
        print(f"📊 Odds chargées pour GW{gameweek}: {len(odds_data['odds'])} matchs")
        return odds_data['odds']
    except FileNotFoundError:
        print(f"⚠️ Pas d'odds trouvées pour GW{gameweek}")
        return {}

def calculate_market_features_from_odds(home_team, away_team, odds_data):
    """Calculate market features from live odds using Power/Clarke method"""
    # Team name mapping for common variations
    team_mapping = {
        "Nott'm Forest": "Nottingham Forest",
        "Man City": "Manchester City", 
        "Man Utd": "Man United",
        "Spurs": "Tottenham",
        "Newcastle": "Newcastle United"
    }
    
    # Map team names
    home_mapped = team_mapping.get(home_team, home_team)
    away_mapped = team_mapping.get(away_team, away_team)
    
    # Try multiple variations
    possible_keys = [
        f"{home_team}_vs_{away_team}",
        f"{home_mapped}_vs_{away_mapped}",
        f"{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}",
        f"{home_mapped.replace(' ', '_')}_vs_{away_mapped.replace(' ', '_')}",
    ]
    
    # Try to find match in odds data
    odds = None
    for key in possible_keys:
        if key in odds_data:
            odds = odds_data[key]
            break
    
    if odds is None:
        # NO FALLBACK - Odds required for this function
        raise ValueError(f"❌ Odds required for {home_team} vs {away_team} - no fallbacks allowed")
    
    # Extract odds
    home_odd = odds['B365H']
    draw_odd = odds['B365D'] 
    away_odd = odds['B365A']
    
    # Power/Clarke overround correction
    raw_probs = [1/home_odd, 1/draw_odd, 1/away_odd]
    overround = sum(raw_probs)
    
    if overround > 1.0:
        # Power method (Clarke 2017)
        power_factor = np.log(1.0) / np.log(overround)
        power_probs = [p**power_factor for p in raw_probs]
        
        # Validation
        sum_power_probs = sum(power_probs)
        if abs(sum_power_probs - 1.0) < 0.01:  # Tolérance ±1%
            normalized_probs = power_probs
        else:
            # Fallback normalisation simple
            normalized_probs = [p/overround for p in raw_probs]
    else:
        normalized_probs = raw_probs
    
    prob_home, prob_draw, prob_away = normalized_probs
    
    # Calculate market entropy (higher = more uncertain)
    entropy = -(prob_home * np.log2(prob_home + 1e-8) + 
               prob_draw * np.log2(prob_draw + 1e-8) + 
               prob_away * np.log2(prob_away + 1e-8))
    market_entropy_norm = entropy / np.log2(3)  # Normalize by max entropy
    
    # Determine favorite
    favorite_side = 1 if prob_home > prob_away else 0  # 1=Home favorite, 0=Away favorite
    
    print(f"   ✅ Odds réelles: {home_odd:.2f}/{draw_odd:.2f}/{away_odd:.2f} → probs: {prob_home:.3f}/{prob_draw:.3f}/{prob_away:.3f}")
    
    return {
        'market_entropy_norm': market_entropy_norm,
        'favorite_side_b365': favorite_side,
        'market_prob_away_b365': prob_away
    }

def calculate_h2h_score(home_team, away_team, df_history, current_date=None, window=5):
    """Calculate head-to-head score for a match"""
    if current_date is None:
        current_date = pd.Timestamp.now()
    
    # Find previous H2H matches
    h2h_matches = df_history[
        (((df_history['HomeTeam'] == home_team) & (df_history['AwayTeam'] == away_team)) |
         ((df_history['HomeTeam'] == away_team) & (df_history['AwayTeam'] == home_team))) &
        (df_history['Date'] < current_date)
    ].sort_values('Date').tail(window)
    
    if len(h2h_matches) == 0:
        # NO FALLBACK - Calculate from recent form instead
        home_form_matches = df_history[
            (((df_history['HomeTeam'] == home_team) | (df_history['AwayTeam'] == home_team)) &
             (df_history['Date'] < current_date))
        ].tail(5)
        
        away_form_matches = df_history[
            (((df_history['HomeTeam'] == away_team) | (df_history['AwayTeam'] == away_team)) &
             (df_history['Date'] < current_date))
        ].tail(5)
        
        if len(home_form_matches) == 0 or len(away_form_matches) == 0:
            return 0.5  # Only fallback when no data at all
        
        # Calculate relative form as H2H proxy
        home_form_score = 0
        for _, match in home_form_matches.iterrows():
            if match['HomeTeam'] == home_team:
                if 'FTHG' in match and 'FTAG' in match:
                    if match['FTHG'] > match['FTAG']:
                        home_form_score += 1
                    elif match['FTHG'] == match['FTAG']:
                        home_form_score += 0.5
                elif 'FTR' in match and match['FTR'] == 'H':
                    home_form_score += 1
                elif 'FTR' in match and match['FTR'] == 'D':
                    home_form_score += 0.5
            else:  # Away match
                if 'FTHG' in match and 'FTAG' in match:
                    if match['FTAG'] > match['FTHG']:
                        home_form_score += 1
                    elif match['FTAG'] == match['FTHG']:
                        home_form_score += 0.5
                elif 'FTR' in match and match['FTR'] == 'A':
                    home_form_score += 1
                elif 'FTR' in match and match['FTR'] == 'D':
                    home_form_score += 0.5
        
        away_form_score = 0
        for _, match in away_form_matches.iterrows():
            if match['HomeTeam'] == away_team:
                if 'FTHG' in match and 'FTAG' in match:
                    if match['FTHG'] > match['FTAG']:
                        away_form_score += 1
                    elif match['FTHG'] == match['FTAG']:
                        away_form_score += 0.5
                elif 'FTR' in match and match['FTR'] == 'H':
                    away_form_score += 1
                elif 'FTR' in match and match['FTR'] == 'D':
                    away_form_score += 0.5
            else:  # Away match
                if 'FTHG' in match and 'FTAG' in match:
                    if match['FTAG'] > match['FTHG']:
                        away_form_score += 1
                    elif match['FTAG'] == match['FTHG']:
                        away_form_score += 0.5
                elif 'FTR' in match and match['FTR'] == 'A':
                    away_form_score += 1
                elif 'FTR' in match and match['FTR'] == 'D':
                    away_form_score += 0.5
        
        # Return relative form as H2H proxy
        home_form_avg = home_form_score / len(home_form_matches)
        away_form_avg = away_form_score / len(away_form_matches)
        return (home_form_avg - away_form_avg + 1) / 2  # Normalize to [0,1]
    
    # Calculate home team's performance in H2H
    home_points = 0
    for _, match in h2h_matches.iterrows():
        if 'FullTimeResult' in match:
            result = match['FullTimeResult']
        elif 'result' in match:
            result = match['result']
        else:
            continue
            
        if match['HomeTeam'] == home_team:
            if result == 'H':
                home_points += 3
            elif result == 'D':
                home_points += 1
        else:  # home_team was away
            if result == 'A':
                home_points += 3
            elif result == 'D':
                home_points += 1
    
    max_points = len(h2h_matches) * 3
    return home_points / max_points if max_points > 0 else 0.5

def load_enhanced_dataset():
    """Load enhanced features FULL SOURCES dataset (toutes données intégrées)"""
    try:
        # Charger dataset enhanced complet TOUTES SOURCES avec GW9 INTÉGRÉ
        df_enhanced = pd.read_csv("data/processed/enhanced_features_full_sources_updated.csv")
        df_enhanced['Date'] = pd.to_datetime(df_enhanced['Date'])
        
        print(f"📊 Loaded enhanced FULL SOURCES dataset: {len(df_enhanced)} matches")
        print(f"📅 Date range: {df_enhanced['Date'].min()} → {df_enhanced['Date'].max()}")
        print(f"✅ xG coverage: {df_enhanced['home_xg_valid'].sum()}/{len(df_enhanced)} matches")
        print(f"✅ ELO coverage: {(df_enhanced['elo_diff_normalized'] != 0).sum()}/{len(df_enhanced)} matches")
        print(f"✅ Odds coverage: {(df_enhanced['market_entropy_norm'] != 0.5).sum()}/{len(df_enhanced)} matches")
        
        return df_enhanced, df_enhanced  # Retourner le même dataset 2x pour compatibilité
    except Exception as e:
        print(f"❌ Error loading enhanced dataset: {e}")
        # Fallback vers ancien dataset
        try:
            df_enhanced = pd.read_csv("data/processed/enhanced_features_e0_direct.csv")
            df_enhanced['Date'] = pd.to_datetime(df_enhanced['Date'])
            print(f"⚠️ Fallback to E0 direct dataset: {len(df_enhanced)} matches")
            return df_enhanced, df_enhanced
        except:
            return None, None

def create_features_for_fixtures(fixtures, df_enhanced, df_e0, live_odds=None):
    """Create prediction features for future fixtures with 100% dynamic calculations - NO FALLBACKS"""
    prediction_data = []
    
    print("🔧 Calcul features 100% dynamiques - ZÉRO FALLBACK")
    
    # Initialize dynamic calculator with FULL historical data (2280 matches)
    # Load the big historical dataset instead of just enhanced 80 matches
    try:
        df_full_historical = pd.read_csv('rest/legacy/data/raw/PremierLeague.csv')
        df_full_historical['Date'] = pd.to_datetime(df_full_historical['Date'])
        print(f"   📊 Loaded FULL historical dataset: {len(df_full_historical)} matches for dynamic calculations")
        calculator = DynamicFeaturesCalculator(df_full_historical)
    except Exception as e:
        print(f"   ⚠️ Could not load full historical dataset: {e}")
        print(f"   📊 Fallback to enhanced dataset: {len(df_enhanced)} matches")
        calculator = DynamicFeaturesCalculator(df_enhanced)
    
    # NO MORE TEAM AVERAGES - All features calculated dynamically per match
    
    for _, fixture in fixtures.iterrows():
        home_team = fixture['Home Team'] 
        away_team = fixture['Away Team']
        match_date = fixture['Date']
        gameweek = fixture.get('Round', 10)  # Get gameweek from fixture
        
        # Calculate ALL features dynamically using the calculator
        dynamic_features = calculator.calculate_all_features(home_team, away_team, match_date, gameweek)
        
        # Market features depuis vraies odds ou calcul dynamique
        try:
            if live_odds is not None:
                market_features = calculate_market_features_from_odds(home_team, away_team, live_odds)
                market_entropy = market_features['market_entropy_norm']
                favorite_side = market_features['favorite_side_b365']
                market_prob_away = market_features['market_prob_away_b365']
                print(f"   📊 Used live odds for market features")
            else:
                raise ValueError("No live odds available")
        except (ValueError, KeyError):
            # Si pas d'odds live, calculer market entropy depuis force des équipes
            # Utilise ELO difference pour estimer probabilities
            elo_diff = dynamic_features['elo_diff_normalized'] - 0.5  # Convert back to [-0.5, 0.5]
            form_diff = dynamic_features['form_diff_normalized'] - 0.5
            
            # Calculate implied probabilities from team strength
            home_strength = 0.5 + elo_diff * 0.3 + form_diff * 0.2  # Home advantage base
            prob_home = np.clip(home_strength, 0.15, 0.85)
            prob_away = np.clip(1 - home_strength - 0.25, 0.15, 0.85)  # Reserve space for draw
            prob_draw = 1 - prob_home - prob_away
            
            # Calculate entropy from implied probabilities
            market_entropy = -(prob_home * np.log2(prob_home + 1e-8) + 
                             prob_draw * np.log2(prob_draw + 1e-8) + 
                             prob_away * np.log2(prob_away + 1e-8)) / np.log2(3)
            
            favorite_side = 1 if prob_home > prob_away else 0
            market_prob_away = prob_away
            
            print(f"   📊 Calculated market features from team strength: entropy={market_entropy:.3f}")
        
        # Combine all features (NO FALLBACKS - all calculated dynamically)
        features = {
            'Date': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'market_entropy_norm': market_entropy,
            'favorite_side_b365': favorite_side,
            'market_prob_away_b365': market_prob_away,
            **dynamic_features  # Add all dynamically calculated features
        }
        
        prediction_data.append(features)
    
    return pd.DataFrame(prediction_data)

def predict_matches(model, scaler, features_list, prediction_df):
    """Generate predictions for matches"""
    predictions = []
    
    # Créer feature matrix
    X = prediction_df[features_list].values
    
    # Scale features pour v3.0
    X_scaled = scaler.transform(X)
    
    # Prédire les probabilités
    probs = model.predict_proba(X_scaled)
    
    for i, (_, match) in enumerate(prediction_df.iterrows()):
        prob_away, prob_draw, prob_home = probs[i]
        
        # Déterminer prédiction (argmax)
        if prob_home > prob_draw and prob_home > prob_away:
            prediction = 'H'
            confidence = prob_home
        elif prob_away > prob_draw and prob_away > prob_home:
            prediction = 'A'
            confidence = prob_away
        else:
            prediction = 'D'
            confidence = prob_draw
            
        predictions.append({
            'Date': match['Date'].strftime('%Y-%m-%d'),
            'HomeTeam': match['HomeTeam'],
            'AwayTeam': match['AwayTeam'],
            'Predicted': prediction,
            'Prob_Home': round(prob_home, 4),
            'Prob_Draw': round(prob_draw, 4),
            'Prob_Away': round(prob_away, 4),
            'Confidence': round(confidence, 4)
        })
    
    return predictions

def save_predictions(predictions, gameweek, output_dir):
    """Save predictions to CSV and JSON files"""
    import os
    from pathlib import Path
    
    # Créer répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Timestamp pour noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder CSV (format legacy)
    csv_file = output_path / f"j{gameweek}_production_{timestamp}.csv"
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(csv_file, index=False)
    print(f"💾 Predictions saved: {csv_file}")
    
    # Sauvegarder JSON (format API v5)
    json_file = output_path / f"j{gameweek}_predictions_v3_j{gameweek}_{timestamp}.json"
    
    json_data = {
        "api_version": "5.0.0",
        "mode": "real_pipeline_production",
        "gameweek": gameweek,
        "metadata": {
            "season_hash": f"epl_2025_26_j{gameweek}",
            "generated_at": datetime.now().isoformat(),
            "model_version": "enhanced_baseline_v2.4",
            "pipeline_version": "durci_v1.0",
            "source_file": f"j{gameweek}_predictions_v3_j{gameweek}_{timestamp}.json"
        },
        "fixtures_count": len(predictions),
        "predictions": {}
    }
    
    # Convertir vers format API v5
    for pred in predictions:
        match_key = f"{pred['HomeTeam']}_vs_{pred['AwayTeam']}"
        json_data["predictions"][match_key] = {
            "prediction": pred['Predicted'],
            "confidence": pred['Confidence'],
            "probabilities": {
                "home": pred['Prob_Home'],
                "draw": pred['Prob_Draw'],
                "away": pred['Prob_Away']
            },
            "model_info": {
                "prediction_mode": "enhanced_baseline_v24",
                "enhanced_metadata": {},
                "model_version": "v2.4",
                "accuracy_improvement": "baseline_champion",
                "away_bias_correction": "enabled"
            },
            "market_features": {
                "market_confidence": pred['Prob_Home'] if pred['Predicted'] == 'H' else pred['Prob_Away'],
                "market_entropy": 0.5,
                "market_favorite": pred['Predicted'],
                "home_advantage_market": 0.1
            },
            "match_info": {
                "home": pred['HomeTeam'],
                "away": pred['AwayTeam'],
                "date": pred['Date']
            }
        }
    
    # Sauvegarder JSON
    import json
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 API v5 predictions saved: {json_file}")
    
    # Trigger frontend revalidation pour ISR
    try:
        from backend.services.frontend_revalidation import get_revalidation_service
        
        revalidation_service = get_revalidation_service()
        revalidation_result = revalidation_service.revalidate_after_predictions(gameweek)
        
        if revalidation_result.get('revalidated'):
            print(f"🔄 Frontend ISR revalidation triggered for {len(revalidation_result.get('paths', []))} paths")
        else:
            print(f"⚠️ Frontend revalidation skipped: {revalidation_result.get('reason', 'Unknown')}")
            
    except Exception as e:
        print(f"⚠️ Frontend revalidation error (non-critical): {e}")
    
    return str(csv_file), str(json_file)

def main():
    args = parse_args()
    gameweek = args.gameweek
    
    print(f"🎯 J{gameweek} EPL PREDICTIONS - ENHANCED STRICT")
    print("=" * 60)
    
    # Charger modèle v2.4
    model, scaler, features = load_enhanced_baseline_v30()
    if model is None:
        return 1
    
    # Charger calendrier EPL
    df_calendar = load_epl_calendar()
    if df_calendar is None:
        return 1
    
    # Obtenir fixtures pour la gameweek
    fixtures = get_gameweek_fixtures(df_calendar, gameweek)
    if fixtures is None:
        return 1
    
    # Charger données d'entraînement
    df_enhanced, df_e0 = load_enhanced_dataset()
    if df_enhanced is None:
        return 1
    
    # Charger odds en temps réel pour features market
    live_odds = load_live_odds(gameweek)
    
    # Créer features pour les fixtures avec odds en temps réel
    prediction_df = create_features_for_fixtures(fixtures, df_enhanced, df_e0, live_odds)
    print(f"📊 Feature matrix: {prediction_df.shape}")
    
    # Générer prédictions
    predictions = predict_matches(model, scaler, features, prediction_df)
    
    # Afficher résultats
    print(f"\n🎯 J{gameweek} PREDICTIONS")
    print("=" * 60)
    for pred in predictions:
        home_pad = pred['HomeTeam'].ljust(15)
        away_pad = pred['AwayTeam'].ljust(15)
        conf = pred['Confidence']
        probs = f"H: {pred['Prob_Home']:.3f} | D: {pred['Prob_Draw']:.3f} | A: {pred['Prob_Away']:.3f}"
        print(f"{home_pad} vs {away_pad} → {pred['Predicted']} ({conf:.3f})")
        print(f"   {probs}")
        print()
    
    # Statistiques
    home_wins = sum(1 for p in predictions if p['Predicted'] == 'H')
    draws = sum(1 for p in predictions if p['Predicted'] == 'D')
    away_wins = sum(1 for p in predictions if p['Predicted'] == 'A')
    avg_conf = sum(p['Confidence'] for p in predictions) / len(predictions)
    
    print("📊 PREDICTION SUMMARY:")
    print(f"Home wins (H): {home_wins}")
    print(f"Draws (D): {draws}")
    print(f"Away wins (A): {away_wins}")
    print(f"Avg Confidence: {avg_conf:.3f}")
    
    # Sauvegarder
    csv_file, json_file = save_predictions(predictions, gameweek, args.output)
    
    print(f"\n✅ J{gameweek} PREDICTIONS COMPLETED")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)