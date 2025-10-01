#!/usr/bin/env python3
"""
🎯 EXTENSION TEST J4 - 40 MATCHES VALIDATION

Objectif: Intégrer J4 pour test sur 40 matches avec features temporellement correctes
Méthodologie: Features figées à fin J3, aucun data leakage, simulation production authentique

Modèle: Reste entraîné sur 2019-2025 (inchangé)
Test élargi: 30 → 40 matches (J1-J2-J3-J4)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime

def load_existing_j1_j3_data():
    """Charger données test existantes J1-J3 (30 matches)"""
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    cutoff_date = pd.Timestamp('2025-08-01')
    test_j1_j3 = df[df['Date'] >= cutoff_date].copy()
    
    print(f"📊 J1-J3 existants: {len(test_j1_j3)} matches")
    print(f"Date range: {test_j1_j3['Date'].min().strftime('%Y-%m-%d')} → {test_j1_j3['Date'].max().strftime('%Y-%m-%d')}")
    
    return test_j1_j3

def parse_j4_results():
    """Parser résultats J4 depuis fichier raw"""
    print("\n🔍 PARSING RÉSULTATS J4:")
    print("-" * 40)
    
    # Charger fichier raw
    df_raw = pd.read_csv('/Users/maxime/Desktop/Oddsy/data/raw/epl-2025-GMTStandardTime_NEW.csv')
    j4_raw = df_raw[df_raw['Round Number'] == 4].copy()
    
    print(f"J4 matches trouvés: {len(j4_raw)}")
    
    # Mapping noms équipes pour cohérence avec dataset existant
    team_mapping = {
        'Spurs': 'Tottenham',
        'Man Utd': 'Man United', 
        'Man City': 'Manchester City',
        "Nott'm Forest": 'Nottingham Forest'
    }
    
    j4_processed = []
    
    for _, match in j4_raw.iterrows():
        home_team = match['Home Team']
        away_team = match['Away Team']
        result_raw = match['Result']
        date_raw = match['Date']
        
        # Appliquer mapping noms
        home_team = team_mapping.get(home_team, home_team)
        away_team = team_mapping.get(away_team, away_team)
        
        # Parser résultat (ex: "3 - 0" → "H")
        try:
            home_goals, away_goals = map(int, result_raw.split(' - '))
            if home_goals > away_goals:
                result = 'H'
            elif home_goals < away_goals:
                result = 'A'
            else:
                result = 'D'
        except:
            print(f"⚠️ Erreur parsing résultat: {result_raw}")
            continue
        
        # Parser date (format: "13/09/2025 12:30")
        try:
            match_date = pd.to_datetime(date_raw, format='%d/%m/%Y %H:%M')
        except:
            match_date = pd.Timestamp('2025-09-13')  # Fallback
        
        j4_processed.append({
            'Date': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FullTimeResult': result,
            'home_goals': home_goals,
            'away_goals': away_goals
        })
        
        print(f"{home_team} vs {away_team}: {result} ({home_goals}-{away_goals})")
    
    return j4_processed

def get_team_features_j3(test_j1_j3, team_name):
    """Récupérer dernières features connues pour une équipe"""
    # Features dans l'ordre EXACT du modèle (depuis metadata.json)
    baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Trouver derniers matches de cette équipe (home ou away)
    team_matches = test_j1_j3[
        (test_j1_j3['HomeTeam'] == team_name) | (test_j1_j3['AwayTeam'] == team_name)
    ].sort_values('Date')
    
    if len(team_matches) > 0:
        # Prendre dernière occurrence
        last_match = team_matches.iloc[-1]
        
        # Extraire features (ajuster selon position home/away)
        features = {}
        for feat in baseline_features:
            if feat in last_match:
                features[feat] = last_match[feat]
            else:
                features[feat] = 0.5  # Valeur neutre par défaut
        
        return features
    else:
        # Équipe non trouvée, utiliser valeurs neutres
        return {feat: 0.5 for feat in baseline_features}

def create_j4_with_frozen_features(test_j1_j3, j4_processed):
    """Créer dataset J4 avec features figées à fin J3"""
    print("\n🧊 CRÉATION J4 AVEC FEATURES FIGÉES:")
    print("-" * 40)
    
    # Features dans l'ordre EXACT du modèle (depuis metadata.json)
    baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    j4_matches = []
    
    for match in j4_processed:
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Récupérer features figées pour chaque équipe
        home_features = get_team_features_j3(test_j1_j3, home_team)
        away_features = get_team_features_j3(test_j1_j3, away_team)
        
        # Construire ligne J4 avec features figées
        j4_row = {
            'Date': match['Date'],
            'Season': '2025-2026',
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FullTimeResult': match['FullTimeResult']
        }
        
        # Features figées: utilisation intelligente des dernières valeurs
        # Approximation: moyenner certaines features, garder autres spécifiques
        for feat in baseline_features:
            if feat in ['h2h_score', 'matchday_normalized']:
                # Features contextuelles: valeurs neutres/calculées
                if feat == 'matchday_normalized':
                    j4_row[feat] = 4/38  # J4 sur 38 journées
                else:
                    j4_row[feat] = 0.5
            else:
                # Features équipes: moyenne des dernières valeurs connues
                home_val = home_features.get(feat, 0.5)
                away_val = away_features.get(feat, 0.5)
                
                if feat in ['form_diff_normalized', 'elo_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized']:
                    # Features différentielles: home - away
                    j4_row[feat] = (home_val + (1 - away_val)) / 2
                elif feat in ['home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5']:
                    # Features équipe-spécifiques
                    if feat == 'home_xg_eff_10':
                        j4_row[feat] = home_val
                    elif feat == 'away_xg_eff_10':
                        j4_row[feat] = away_val
                    else:  # away_goals_sum_5
                        j4_row[feat] = away_val
                else:
                    # Autres: moyenne
                    j4_row[feat] = (home_val + away_val) / 2
        
        j4_matches.append(j4_row)
        
        print(f"✅ {home_team} vs {away_team} → {match['FullTimeResult']}")
    
    return pd.DataFrame(j4_matches)

def test_model_40_matches():
    """Test principal sur 40 matches avec comparaison robustesse"""
    print("\n🎯 TEST MODÈLE SUR 40 MATCHES")
    print("=" * 60)
    
    # 1. Charger données existantes J1-J3
    test_j1_j3 = load_existing_j1_j3_data()
    
    # 2. Parser J4
    j4_processed = parse_j4_results()
    
    # 3. Créer J4 avec features figées
    j4_df = create_j4_with_frozen_features(test_j1_j3, j4_processed)
    
    # 4. Fusionner pour dataset 40 matches
    test_40_matches = pd.concat([test_j1_j3, j4_df], ignore_index=True)
    
    print(f"\n📊 DATASET FINAL:")
    print(f"Total matches: {len(test_40_matches)}")
    print(f"J1-J3: {len(test_j1_j3)} matches")
    print(f"J4: {len(j4_df)} matches")
    
    # 5. Préparer données pour test
    # Features dans l'ordre EXACT du modèle (depuis metadata.json)
    baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Nettoyer dataset 40 matches
    test_clean = test_40_matches.dropna(subset=baseline_features + ['FullTimeResult'])
    
    X_test_40 = test_clean[baseline_features]
    y_test_40 = test_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print(f"Matches après nettoyage: {len(test_clean)}")
    
    # 6. Charger modèle final
    try:
        model = joblib.load('models/final_robust_model_20250915_163023.joblib')
        print("✅ Modèle final chargé")
    except FileNotFoundError:
        print("❌ Erreur: Modèle final non trouvé")
        return None
    
    # 7. Test performance
    print("\n🧪 PRÉDICTIONS ET PERFORMANCE:")
    print("-" * 40)
    
    y_pred_40 = model.predict(X_test_40)
    accuracy_40 = accuracy_score(y_test_40, y_pred_40)
    
    # Comparaison avec performance 30 matches (J1-J3)
    test_j1_j3_clean = test_j1_j3.dropna(subset=baseline_features + ['FullTimeResult'])
    X_test_30 = test_j1_j3_clean[baseline_features]
    y_test_30 = test_j1_j3_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    y_pred_30 = model.predict(X_test_30)
    accuracy_30 = accuracy_score(y_test_30, y_pred_30)
    
    print(f"Performance J1-J3 (30 matches): {accuracy_30:.3f} ({np.sum(y_pred_30 == y_test_30)}/{len(y_test_30)})")
    print(f"Performance J1-J4 (40 matches): {accuracy_40:.3f} ({np.sum(y_pred_40 == y_test_40)}/{len(y_test_40)})")
    print(f"Robustesse temporelle: {accuracy_40-accuracy_30:+.3f}")
    
    # 8. Analyse détaillée
    print("\n📋 ANALYSE DÉTAILLÉE 40 MATCHES:")
    print("-" * 40)
    
    # Rapport par classe
    class_report = classification_report(y_test_40, y_pred_40, target_names=['H', 'D', 'A'], output_dict=True)
    
    for class_name in ['H', 'D', 'A']:
        recall = class_report[class_name]['recall']
        precision = class_report[class_name]['precision']
        f1 = class_report[class_name]['f1-score']
        support = int(class_report[class_name]['support'])
        
        print(f"{class_name}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f} (n={support})")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test_40, y_pred_40)
    print(f"\nMatrice de confusion 40 matches:")
    print(f"     H   D   A")
    for i, row_label in enumerate(['H', 'D', 'A']):
        row_str = f"{row_label} [{' '.join(f'{val:3d}' for val in cm[i])}]"
        print(row_str)
    
    # 9. Analyse spécifique J4
    print(f"\n🔍 ANALYSE SPÉCIFIQUE J4 ({len(j4_df)} matches):")
    print("-" * 40)
    
    j4_indices = list(range(len(test_j1_j3_clean), len(test_clean)))
    if j4_indices:
        y_test_j4 = y_test_40.iloc[j4_indices]
        y_pred_j4 = y_pred_40[j4_indices]
        accuracy_j4 = accuracy_score(y_test_j4, y_pred_j4)
        
        print(f"Performance J4 seulement: {accuracy_j4:.3f} ({np.sum(y_pred_j4 == y_test_j4)}/{len(y_test_j4)})")
        
        # Détail matches J4
        j4_clean = test_clean.iloc[j4_indices]
        for i, (_, match) in enumerate(j4_clean.iterrows()):
            pred_label = ['H', 'D', 'A'][y_pred_j4[i]]
            actual_label = match['FullTimeResult']
            status = "✅" if pred_label == actual_label else "❌"
            print(f"{status} {match['HomeTeam']} vs {match['AwayTeam']}: {pred_label} (réel: {actual_label})")
    
    return {
        'accuracy_30': accuracy_30,
        'accuracy_40': accuracy_40,
        'accuracy_j4': accuracy_j4 if j4_indices else 0,
        'robustness': accuracy_40 - accuracy_30,
        'dataset_40': test_clean
    }

if __name__ == "__main__":
    print("🚀 EXTENSION TEST J4 - VALIDATION 40 MATCHES")
    print("=" * 80)
    
    results = test_model_40_matches()
    
    if results:
        print(f"\n🏁 RÉSULTATS FINAUX:")
        print(f"📈 Performance 30 matches: {results['accuracy_30']:.1%}")
        print(f"📈 Performance 40 matches: {results['accuracy_40']:.1%}")
        print(f"🎯 Robustesse temporelle: {results['robustness']:+.1%}")
        print(f"🔍 Performance J4 seul: {results['accuracy_j4']:.1%}")
        print("\n✅ VALIDATION 40 MATCHES TERMINÉE!")