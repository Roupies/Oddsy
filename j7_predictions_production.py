"""
Prédictions J7 EPL 2025-26 - Production Models
Utilise les Champion Models validés en production
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from j7_odds_input import get_j7_dataframe
import sys
import os

# Ajout du chemin pour les modules
sys.path.append('dashboards/core')
from data_loader import load_match_data

def load_production_models():
    """Charge les modèles de production validés"""
    models = {}
    
    # Baseline Champion v2.3 (53.5% CV)
    try:
        models['baseline'] = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Baseline Champion v2.3 chargé")
    except:
        print("❌ Erreur chargement Baseline Champion")
        
    # Cascade Champion v2.0 (spécialiste draws)
    try:
        models['cascade'] = joblib.load('models/production/cascade_champion_v20_production.joblib')
        print("✅ Cascade Champion v2.0 chargé")
    except:
        print("❌ Erreur chargement Cascade Champion")
        
    return models

def prepare_features_for_prediction(df_historical, j7_matches):
    """Prépare les features pour les prédictions J7"""
    
    # Récupère les dernières données pour calculer les features
    latest_date = pd.to_datetime(df_historical['Date'].max(), format='%d/%m/%Y')
    print(f"Dernière date dataset: {latest_date}")
    
    predictions_input = []
    
    for _, match in j7_matches.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Features de base depuis les cotes
        home_odds = match['B365H']
        draw_odds = match['B365D'] 
        away_odds = match['B365A']
        
        # Calcul des probabilités implicites
        total_prob = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        home_prob = (1/home_odds) / total_prob
        draw_prob = (1/draw_odds) / total_prob
        away_prob = (1/away_odds) / total_prob
        
        # Entropie du marché (feature importante)
        market_entropy = -(home_prob * np.log(home_prob) + 
                          draw_prob * np.log(draw_prob) + 
                          away_prob * np.log(away_prob))
        
        # Normalisation de l'entropie (0-1)
        market_entropy_norm = market_entropy / np.log(3)
        
        # Features historiques (simplifiées pour cette démo)
        home_recent = df_historical[
            (df_historical['HomeTeam'] == home_team) | 
            (df_historical['AwayTeam'] == home_team)
        ].tail(5)
        
        away_recent = df_historical[
            (df_historical['HomeTeam'] == away_team) | 
            (df_historical['AwayTeam'] == away_team)
        ].tail(5)
        
        # Stats moyennes récentes (placeholder)
        home_form = len(home_recent) / 5.0 if len(home_recent) > 0 else 0.5
        away_form = len(away_recent) / 5.0 if len(away_recent) > 0 else 0.5
        
        # Construction du vecteur de features
        features = {
            'market_entropy_norm': market_entropy_norm,
            'home_odds': home_odds,
            'draw_odds': draw_odds, 
            'away_odds': away_odds,
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
            'home_form': home_form,
            'away_form': away_form,
            'total_goals_expectation': 2.5,  # Moyenne EPL
        }
        
        predictions_input.append({
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': match['Date'],
            'Time': match['Time'],
            'features': features
        })
    
    return predictions_input

def make_predictions(models, predictions_input):
    """Génère les prédictions avec les modèles"""
    
    results = []
    
    for match_data in predictions_input:
        home_team = match_data['HomeTeam']
        away_team = match_data['AwayTeam']
        features = match_data['features']
        
        print(f"\n🏈 {home_team} vs {away_team}")
        print(f"   Market Entropy: {features['market_entropy_norm']:.3f}")
        print(f"   Cotes: {features['home_odds']:.2f} | {features['draw_odds']:.2f} | {features['away_odds']:.2f}")
        
        match_predictions = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': match_data['Date'],
            'Time': match_data['Time'],
            'market_entropy': features['market_entropy_norm'],
            'odds': [features['home_odds'], features['draw_odds'], features['away_odds']]
        }
        
        # Baseline Champion v2.3
        if 'baseline' in models:
            try:
                # Prépare les features dans l'ordre attendu
                feature_vector = np.array([
                    features['market_entropy_norm'],
                    features['home_odds'],
                    features['draw_odds'],
                    features['away_odds'],
                    features['home_prob'],
                    features['draw_prob'],
                    features['away_prob'],
                    features['home_form'],
                    features['away_form'],
                    features['total_goals_expectation']
                ]).reshape(1, -1)
                
                baseline_pred = models['baseline'].predict(feature_vector)[0]
                baseline_proba = models['baseline'].predict_proba(feature_vector)[0]
                
                match_predictions['baseline'] = {
                    'prediction': int(baseline_pred),
                    'probabilities': {
                        'H': float(baseline_proba[0]) if len(baseline_proba) > 0 else 0.33,
                        'D': float(baseline_proba[1]) if len(baseline_proba) > 1 else 0.33,
                        'A': float(baseline_proba[2]) if len(baseline_proba) > 2 else 0.33
                    },
                    'confidence': float(max(baseline_proba))
                }
                
                pred_labels = ['H', 'D', 'A']
                print(f"   🎯 Baseline v2.3: {pred_labels[baseline_pred]} (conf: {max(baseline_proba):.3f})")
                
            except Exception as e:
                print(f"   ❌ Erreur Baseline: {e}")
                match_predictions['baseline'] = {'error': str(e)}
        
        # Cascade Champion v2.0
        if 'cascade' in models:
            try:
                # Utilise les mêmes features
                cascade_pred = models['cascade'].predict(feature_vector)[0]
                cascade_proba = models['cascade'].predict_proba(feature_vector)[0]
                
                match_predictions['cascade'] = {
                    'prediction': int(cascade_pred),
                    'probabilities': {
                        'H': float(cascade_proba[0]) if len(cascade_proba) > 0 else 0.33,
                        'D': float(cascade_proba[1]) if len(cascade_proba) > 1 else 0.33,
                        'A': float(cascade_proba[2]) if len(cascade_proba) > 2 else 0.33
                    },
                    'confidence': float(max(cascade_proba))
                }
                
                print(f"   🎯 Cascade v2.0: {pred_labels[cascade_pred]} (conf: {max(cascade_proba):.3f})")
                
            except Exception as e:
                print(f"   ❌ Erreur Cascade: {e}")
                match_predictions['cascade'] = {'error': str(e)}
        
        # Consensus (si les deux modèles sont disponibles)
        if 'baseline' in match_predictions and 'cascade' in match_predictions:
            if 'error' not in match_predictions['baseline'] and 'error' not in match_predictions['cascade']:
                baseline_conf = match_predictions['baseline']['confidence']
                cascade_conf = match_predictions['cascade']['confidence']
                
                if baseline_conf > cascade_conf:
                    consensus = match_predictions['baseline']['prediction']
                    consensus_source = 'baseline'
                else:
                    consensus = match_predictions['cascade']['prediction']
                    consensus_source = 'cascade'
                
                match_predictions['consensus'] = {
                    'prediction': int(consensus),
                    'source': consensus_source,
                    'baseline_conf': float(baseline_conf),
                    'cascade_conf': float(cascade_conf)
                }
                
                print(f"   🏆 Consensus: {pred_labels[consensus]} (via {consensus_source})")
        
        results.append(match_predictions)
    
    return results

def save_predictions(predictions):
    """Sauvegarde les prédictions"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions/j7_predictions_{timestamp}.json"
    
    # Crée le dossier si nécessaire
    os.makedirs('predictions', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'matchday': 'J7',
            'season': '2025-26',
            'model_versions': {
                'baseline': 'Champion v2.3',
                'cascade': 'Champion v2.0'
            },
            'predictions': predictions
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Prédictions sauvegardées: {filename}")
    return filename

def main():
    print("=== PRÉDICTIONS J7 EPL 2025-26 ===")
    print("Production Models: Baseline v2.3 + Cascade v2.0")
    print()
    
    # 1. Charge les données historiques
    print("📊 Chargement données historiques...")
    df_historical = load_match_data()
    print(f"   Dataset: {len(df_historical)} matchs")
    
    # 2. Charge les modèles
    print("\n🤖 Chargement modèles production...")
    models = load_production_models()
    
    # 3. Données J7
    print("\n📅 Préparation matchs J7...")
    j7_matches = get_j7_dataframe()
    print(f"   {len(j7_matches)} matchs à prédire")
    
    # 4. Prépare les features
    print("\n🔧 Calcul des features...")
    predictions_input = prepare_features_for_prediction(df_historical, j7_matches)
    
    # 5. Génère les prédictions
    print("\n🎯 Génération des prédictions...")
    predictions = make_predictions(models, predictions_input)
    
    # 6. Sauvegarde
    filename = save_predictions(predictions)
    
    # 7. Résumé
    print("\n" + "="*50)
    print("📋 RÉSUMÉ PRÉDICTIONS J7")
    print("="*50)
    
    for pred in predictions:
        home = pred['HomeTeam']
        away = pred['AwayTeam']
        print(f"\n{home} vs {away}")
        
        if 'baseline' in pred and 'error' not in pred['baseline']:
            b_pred = ['H', 'D', 'A'][pred['baseline']['prediction']]
            b_conf = pred['baseline']['confidence']
            print(f"  Baseline v2.3: {b_pred} ({b_conf:.3f})")
        
        if 'cascade' in pred and 'error' not in pred['cascade']:
            c_pred = ['H', 'D', 'A'][pred['cascade']['prediction']]  
            c_conf = pred['cascade']['confidence']
            print(f"  Cascade v2.0:  {c_pred} ({c_conf:.3f})")
        
        if 'consensus' in pred:
            cons_pred = ['H', 'D', 'A'][pred['consensus']['prediction']]
            cons_source = pred['consensus']['source']
            print(f"  🏆 CONSENSUS: {cons_pred} (via {cons_source})")

if __name__ == "__main__":
    main()