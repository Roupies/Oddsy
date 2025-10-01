"""
Prédictions J7 EPL 2025-26 - Pipeline EXACTE sans approximation
Utilise le système auto-update du projet pour calculer les vraies features
AVEC validations anti-fuite et tracking fallback
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from j7_odds_input import get_j7_dataframe
import sys
import os

# Utiliser notre calculateur exact complet
from j7_feature_calculator_complete import J7FeatureCalculator

# Importer validations et tracking
from scripts.analysis.anti_leak_unit_test import AntiLeakValidator
from feature_fallback_tracker import global_fallback_tracker, track_fallback

def load_historical_data():
    """Charge les données historiques jusqu'à J6"""
    # Utilise le dataset le plus récent
    df = pd.read_csv('data/processed/v_auto_update_20250916_110247.csv')
    
    # Convertir Date au format datetime pour tri temporel strict
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
    # Trier par date pour garantir ordre chronologique
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"📊 Dataset chargé: {len(df)} matchs (jusqu'au {df['Date'].max().strftime('%Y-%m-%d')})")
    return df

def calculate_j7_features(df_historical, j7_matches):
    """Calcule les features J7 avec la pipeline exacte du projet + validations anti-fuite"""
    
    # Initialiser validateur anti-fuite et calculateur exact
    validator = AntiLeakValidator(strict_mode=True)
    feature_calc = J7FeatureCalculator()
    
    j7_with_features = []
    
    print(f"\n🔧 Calcul features pour {len(j7_matches)} matchs J7...")
    print("🛡️ Validations anti-fuite activées")
    
    for _, match in j7_matches.iterrows():
        print(f"\n⚽ {match['HomeTeam']} vs {match['AwayTeam']}")
        
        # Date de référence J7 
        match_date = pd.to_datetime('2025-10-05')  # Date J7
        
        # Filtrer données strictement avant J7 (≤ 2025-10-02)
        cutoff_date = pd.to_datetime('2025-10-02')
        historical_before_j7 = df_historical[df_historical['Date'] <= cutoff_date].copy()
        
        print(f"   📈 Données disponibles: {len(historical_before_j7)} matchs avant J7")
        
        # 🛡️ VALIDATION ANTI-FUITE: Vérifier intégrité temporelle complète
        try:
            validation_result = validator.validate_feature_calculation_pipeline(
                match_date, 
                match['HomeTeam'], 
                match['AwayTeam'], 
                historical_before_j7, 
                feature_calc
            )
            print(f"   ✅ Validation anti-fuite: RÉUSSIE ({len(validation_result['validations'])} checks)")
            
        except Exception as e:
            print(f"   ❌ FUITE TEMPORELLE DÉTECTÉE: {str(e)}")
            print(f"   🚫 Match ignoré pour sécurité")
            continue
        
        try:
            # Calculer toutes les features avec notre calculateur exact
            features = feature_calc.calculate_all_features(match, historical_before_j7)
            
            # 📊 TRACKING FALLBACK: Enregistrer qualité des features calculées
            match_id = f"{match['HomeTeam']}_vs_{match['AwayTeam']}"
            for feature_name, feature_value in features.items():
                is_fallback = pd.isna(feature_value)
                fallback_reason = "k<3 threshold" if is_fallback else None
                
                track_fallback(
                    matchday="J7",
                    match_id=match_id,
                    feature_name=feature_name,
                    is_fallback=is_fallback,
                    reason=fallback_reason
                )
            
            # Obtenir vecteur dans l'ordre exact du modèle
            features_vector = feature_calc.get_features_vector(features)
            
            # Vérifier si des features sont NaN (données insuffisantes)
            nan_features = [name for name, value in features.items() if pd.isna(value)]
            if nan_features:
                print(f"   ⚠️ Features NaN (k<3): {', '.join(nan_features)}")
                print(f"   🔄 Continuation avec NaN...")
            
            # Ajouter match avec features
            match_with_features = {
                'Date': match['Date'],
                'Time': match['Time'],
                'HomeTeam': match['HomeTeam'],
                'AwayTeam': match['AwayTeam'],
                'odds': [float(match['B365H']), float(match['B365D']), float(match['B365A'])],
                'features_vector': features_vector,
                'features_dict': features
            }
            
            j7_with_features.append(match_with_features)
            
        except Exception as e:
            print(f"   ❌ Erreur calcul features: {e}")
            continue
    
    return j7_with_features

def generate_exact_predictions(j7_data):
    """Génère les prédictions avec le modèle et les vraies features"""
    
    # Charger le modèle Baseline Champion v2.3
    try:
        model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Baseline Champion v2.3 chargé")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return []
    
    predictions = []
    pred_labels = ['H', 'D', 'A']
    
    print(f"\n🎯 Génération prédictions exactes...")
    
    for match_data in j7_data:
        home_team = match_data['HomeTeam']
        away_team = match_data['AwayTeam']
        
        print(f"\n🏈 {home_team} vs {away_team}")
        
        try:
            # Préparer vecteur features dans l'ordre exact du modèle
            feature_vector = np.array(match_data['features_vector']).reshape(1, -1)
            
            # Prédiction
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            
            match_prediction = {
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'Date': match_data['Date'],
                'Time': match_data['Time'],
                'odds': match_data['odds'],
                'features_used': match_data['features_dict'],
                'prediction': int(prediction),
                'prediction_label': pred_labels[prediction],
                'probabilities': {
                    'H': float(probabilities[0]),
                    'D': float(probabilities[1]),
                    'A': float(probabilities[2])
                },
                'confidence': float(max(probabilities)),
                'method': 'EXACT_PIPELINE'
            }
            
            print(f"   🎯 Prédiction: {pred_labels[prediction]} (conf: {max(probabilities):.3f})")
            print(f"   📊 Probas: H={probabilities[0]:.3f} D={probabilities[1]:.3f} A={probabilities[2]:.3f}")
            
            predictions.append(match_prediction)
            
        except Exception as e:
            print(f"   ❌ Erreur prédiction: {e}")
            
    return predictions

def save_exact_predictions(predictions):
    """Sauvegarde les prédictions exactes"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions/j7_predictions_exact_pipeline_{timestamp}.json"
    
    os.makedirs('predictions', exist_ok=True)
    
    # Ajouter métadonnées de validation
    output_data = {
        'timestamp': timestamp,
        'matchday': 'J7',
        'season': '2025-26',
        'method': 'EXACT_PIPELINE_NO_APPROXIMATION',
        'model_version': 'Baseline Champion v2.3',
        'features_source': 'Auto-update pipeline exacte du projet',
        'anti_leakage_validated': True,
        'temporal_cutoff': '2025-10-02',
        'features_order': [
            'form_diff_normalized',
            'elo_diff_normalized', 
            'h2h_score',
            'matchday_normalized',
            'shots_diff_normalized',
            'corners_diff_normalized',
            'market_entropy_norm',
            'home_xg_eff_10',
            'away_goals_sum_5',
            'away_xg_eff_10'
        ],
        'predictions': predictions
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Prédictions exactes sauvegardées: {filename}")
    return filename

def main():
    print("=" * 70)
    print("🔬 PRÉDICTIONS J7 EXACTES - PIPELINE SANS APPROXIMATION")
    print("=" * 70)
    print("✓ Utilise le système auto-update du projet")
    print("✓ Features calculées avec modules exacts")  
    print("✓ Anti-leakage temporel strict (≤ 2025-10-02)")
    print("✓ Baseline Champion v2.3 (54.5% accuracy)")
    print()
    
    # 1. Charger données historiques
    print("📊 Chargement données historiques...")
    df_historical = load_historical_data()
    
    # 2. Données matchs J7
    print("\n📅 Préparation matchs J7...")
    j7_matches = get_j7_dataframe()
    print(f"   {len(j7_matches)} matchs à traiter")
    
    # 3. Calculer features exactes
    print("\n🔧 Calcul features exactes avec pipeline projet...")
    j7_with_features = calculate_j7_features(df_historical, j7_matches)
    
    if not j7_with_features:
        print("❌ Aucune feature calculée, arrêt.")
        return
    
    print(f"✅ {len(j7_with_features)} matchs avec features calculées")
    
    # 4. Générer prédictions
    print("\n🎯 Génération prédictions...")
    predictions = generate_exact_predictions(j7_with_features)
    
    if not predictions:
        print("❌ Aucune prédiction générée, arrêt.")
        return
    
    # 5. Sauvegarder
    filename = save_exact_predictions(predictions)
    
    # 6. Résumé final
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ FINAL - PRÉDICTIONS J7 EXACTES")
    print("=" * 70)
    
    pred_distribution = {'H': 0, 'D': 0, 'A': 0}
    
    for pred in predictions:
        home = pred['HomeTeam']
        away = pred['AwayTeam']
        prediction = pred['prediction_label']
        confidence = pred['confidence']
        probas = pred['probabilities']
        
        pred_distribution[prediction] += 1
        
        print(f"\n{home} vs {away}")
        print(f"  🏆 PRÉDICTION: {prediction} (conf: {confidence:.3f})")
        print(f"  📊 H={probas['H']:.3f} | D={probas['D']:.3f} | A={probas['A']:.3f}")
        print(f"  💰 Cotes: {pred['odds'][0]:.2f} | {pred['odds'][1]:.2f} | {pred['odds'][2]:.2f}")
    
    print(f"\n📈 Distribution prédictions:")
    print(f"   🏠 Home (H): {pred_distribution['H']} matchs")
    print(f"   🤝 Draw (D): {pred_distribution['D']} matchs")
    print(f"   ✈️  Away (A): {pred_distribution['A']} matchs")
    
    # 📊 RAPPORT FALLBACK J7
    print(f"\n📊 Génération rapport fallback...")
    try:
        j7_stats = global_fallback_tracker.calculate_matchday_fallback_percentage("J7")
        if j7_stats:
            print(f"   📈 Fallback J7: {j7_stats['overall_fallback_percentage']:.1f}%")
            print(f"   📊 Matchs analysés: {j7_stats['matches_analyzed']}")
            
            # Features les plus problématiques
            problematic_features = [
                f for f, stats in j7_stats['by_feature'].items() 
                if stats['fallback_percentage'] > 50
            ]
            if problematic_features:
                print(f"   ⚠️ Features >50% fallback: {', '.join(problematic_features)}")
        
        # Export rapport complet
        fallback_report_path = global_fallback_tracker.export_fallback_report()
        print(f"   📋 Rapport fallback: {fallback_report_path}")
        
    except Exception as e:
        print(f"   ⚠️ Erreur génération rapport fallback: {e}")
    
    print(f"\n✅ Pipeline EXACTE terminée - Fichier: {filename}")
    print(f"🛡️ Validations anti-fuite: ACTIVÉES")
    print(f"📊 Seuil minimal k≥3: APPLIQUÉ")

if __name__ == "__main__":
    main()