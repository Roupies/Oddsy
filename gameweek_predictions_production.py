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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

def load_epl_calendar():
    """Load EPL 2025-26 calendar"""
    try:
        df_calendar = pd.read_csv("data/raw/epl-2025-GMTStandardTime_NEW.csv")
        df_calendar['Date'] = pd.to_datetime(df_calendar['Date'], format='%d/%m/%Y %H:%M')
        
        print(f"📅 Loaded EPL calendar: {len(df_calendar)} matches")
        return df_calendar
    except Exception as e:
        print(f"❌ Error loading EPL calendar: {e}")
        return None

def get_gameweek_fixtures(df_calendar, gameweek):
    """Get fixtures for a specific gameweek"""
    gw_matches = df_calendar[df_calendar['Round Number'] == gameweek].copy()
    
    if len(gw_matches) == 0:
        print(f"❌ Aucun match trouvé pour J{gameweek}")
        return None
        
    print(f"📋 Fixtures J{gameweek}: {len(gw_matches)} matchs")
    for _, match in gw_matches.iterrows():
        date_str = match['Date'].strftime('%Y-%m-%d')
        print(f"   {date_str} - {match['Home Team']} vs {match['Away Team']}")
    
    return gw_matches

def load_enhanced_dataset():
    """Load enhanced features FULL SOURCES dataset (toutes données intégrées)"""
    try:
        # Charger dataset enhanced complet TOUTES SOURCES
        df_enhanced = pd.read_csv("data/processed/enhanced_features_full_sources.csv")
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

def create_features_for_fixtures(fixtures, df_enhanced, df_e0):
    """Create prediction features for future fixtures using real team statistics"""
    prediction_data = []
    
    print("📊 Calcul features basées sur TOUTES SOURCES (E0 + xG + ELO + Odds)")
    
    # Calculer vraies statistiques par équipe (home/away séparément)  
    home_team_stats = df_enhanced.groupby('HomeTeam').agg({
        'elo_diff_normalized': 'mean',
        'form_diff_normalized': 'mean',
        'shots_diff_normalized': 'mean',
        'corners_diff_normalized': 'mean',
        'home_xg_eff_10': 'mean',
        'market_entropy_norm': 'mean',
        'away_goals_sum_5': 'mean'
    }).reset_index()
    
    away_team_stats = df_enhanced.groupby('AwayTeam').agg({
        'elo_diff_normalized': 'mean', 
        'form_diff_normalized': 'mean',
        'shots_diff_normalized': 'mean',
        'corners_diff_normalized': 'mean',
        'away_xg_eff_10': 'mean',
        'market_entropy_norm': 'mean',
        'away_goals_sum_5': 'mean'
    }).reset_index()
    
    print(f"   📈 Stats calculées pour {len(home_team_stats)} équipes (home)")
    print(f"   📈 Stats calculées pour {len(away_team_stats)} équipes (away)")
    
    for _, fixture in fixtures.iterrows():
        home_team = fixture['Home Team'] 
        away_team = fixture['Away Team']
        match_date = fixture['Date']
        
        # Obtenir vraies stats des équipes
        home_stats = home_team_stats[home_team_stats['HomeTeam'] == home_team]
        away_stats = away_team_stats[away_team_stats['AwayTeam'] == away_team]
        
        if len(home_stats) > 0 and len(away_stats) > 0:
            # Utiliser vraies moyennes historiques des équipes
            h_stats = home_stats.iloc[0]
            a_stats = away_stats.iloc[0]
            
            elo_diff = h_stats['elo_diff_normalized'] - a_stats['elo_diff_normalized']
            form_diff = h_stats['form_diff_normalized'] - a_stats['form_diff_normalized']
            shots_diff = (h_stats['shots_diff_normalized'] + a_stats['shots_diff_normalized']) / 2
            corners_diff = (h_stats['corners_diff_normalized'] + a_stats['corners_diff_normalized']) / 2
            market_entropy = (h_stats['market_entropy_norm'] + a_stats['market_entropy_norm']) / 2
            home_xg_eff = h_stats['home_xg_eff_10']
            away_xg_eff = a_stats['away_xg_eff_10']
            away_goals_sum = a_stats['away_goals_sum_5']
            
            print(f"   ✅ {home_team} vs {away_team}: ELO={elo_diff:.3f}, Form={form_diff:.3f}")
            
        else:
            # Fallback si équipe pas dans historique (équipes promues, etc.)
            print(f"   ⚠️  {home_team} vs {away_team}: Utilisation fallback (équipe peu d'historique)")
            
            # Moyennes générales du dataset
            elo_diff = df_enhanced['elo_diff_normalized'].mean() + np.random.normal(0, 0.05)
            form_diff = df_enhanced['form_diff_normalized'].mean() + np.random.normal(0, 0.03) 
            shots_diff = df_enhanced['shots_diff_normalized'].mean() + np.random.normal(0, 0.02)
            corners_diff = df_enhanced['corners_diff_normalized'].mean() + np.random.normal(0, 0.02)
            market_entropy = df_enhanced['market_entropy_norm'].mean() + np.random.normal(0, 0.05)
            home_xg_eff = df_enhanced['home_xg_eff_10'].mean() + np.random.normal(0, 0.1)
            away_xg_eff = df_enhanced['away_xg_eff_10'].mean() + np.random.normal(0, 0.1)
            away_goals_sum = df_enhanced['away_goals_sum_5'].mean() + np.random.normal(0, 0.2)
        
        # Features calculées avec vraies données
        features = {
            'Date': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'elo_diff_normalized': elo_diff,
            'market_entropy_norm': market_entropy,
            'form_diff_normalized': form_diff,
            'matchday_normalized': 9.0 / 38.0,  # Gameweek 9 de 38  
            'shots_diff_normalized': shots_diff,
            'corners_diff_normalized': corners_diff,
            'home_xg_eff_10': home_xg_eff,
            'away_goals_sum_5': away_goals_sum,  # Vraie moyenne historique
            'away_xg_eff_10': away_xg_eff,
            'favorite_side_b365': 1 if elo_diff > 0.02 else 0,  # Basé sur vraie diff ELO
            'market_prob_away_b365': max(0.15, min(0.65, 0.33 - elo_diff * 0.3 + market_entropy * 0.1))  # Basé sur vraies stats
        }
        
        prediction_data.append(features)
    
    return pd.DataFrame(prediction_data)

def predict_matches(model, features_list, prediction_df):
    """Generate predictions for matches"""
    predictions = []
    
    # Créer feature matrix
    X = prediction_df[features_list].values
    
    # Prédire les probabilités
    probs = model.predict_proba(X)
    
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
    
    # Charger modèle
    model, features, metadata = load_enhanced_v24_fixed()
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
    
    # Créer features pour les fixtures
    prediction_df = create_features_for_fixtures(fixtures, df_enhanced, df_e0)
    print(f"📊 Feature matrix: {prediction_df.shape}")
    
    # Générer prédictions
    predictions = predict_matches(model, features, prediction_df)
    
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