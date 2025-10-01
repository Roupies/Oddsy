#!/usr/bin/env python3
"""
🔧 CORRECTION XG ROLLING - MÉTHODOLOGIE PRODUCTION
=====================================

Problème identifié:
- Features xG début saison mal calculées (rolling sur données inexistantes)
- 5 features contextuelles non validées ajoutées

Solution correcte:
- Équipes EPL existantes: Rolling 10 derniers matchs saison 2024-25  
- Équipes promues (Leeds, Sunderland, Burnley): Valeur neutre 0.5
- Suppression des 5 features contextuelles non validées
- Test sur J1-J4 (40 matches) pour validation

Méthodologie production réaliste.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

def load_and_clean_dataset():
    """Charge dataset et supprime features contextuelles"""
    print("🗂️  Chargement dataset v16...")
    
    # Charger dataset avec features contextuelles
    df = pd.read_csv('data/processed/v16_contextual_features_20250915_171540.csv')
    
    print(f"Dataset original: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")
    
    # Features contextuelles à supprimer
    contextual_features = [
        'rest_days_diff',
        'promoted_team_factor', 
        'early_season_volatility',
        'manager_continuity',
        'transfer_window_impact'
    ]
    
    # Vérifier et supprimer features contextuelles
    features_to_remove = [f for f in contextual_features if f in df.columns]
    if features_to_remove:
        print(f"🗑️  Suppression features contextuelles: {features_to_remove}")
        df = df.drop(columns=features_to_remove)
    else:
        print("⚠️  Features contextuelles déjà supprimées")
    
    print(f"Dataset nettoyé: {df.shape}")
    return df

def identify_promoted_teams():
    """Identifie équipes promues 2025-26"""
    promoted_teams_2025 = ['Leeds', 'Sunderland', 'Burnley']
    print(f"🔼 Équipes promues 2025-26: {promoted_teams_2025}")
    return promoted_teams_2025

def get_last_season_matches():
    """Récupère derniers matchs saison 2024-25 pour rolling xG"""
    print("📅 Récupération matchs fin saison 2024-25...")
    
    # Charger données xG avec colonnes correctes
    try:
        df_xg = pd.read_csv('data/external/understat_xg_data_corrected_names.csv')
        df_xg['Date'] = pd.to_datetime(df_xg['Date'])
        
        # Filtrer saison 2024-25 (derniers matchs)
        season_2024_25 = df_xg[
            (df_xg['Date'] >= '2024-08-01') & 
            (df_xg['Date'] < '2025-08-01')
        ].copy()
        
        print(f"Matchs saison 2024-25: {len(season_2024_25)}")
        return season_2024_25
        
    except FileNotFoundError:
        print("⚠️  Données xG non trouvées, utilisation valeurs neutres")
        return pd.DataFrame()

def calculate_proper_xg_features(df, promoted_teams):
    """Calcule features xG avec méthodologie production correcte"""
    print("🎯 Calcul features xG avec méthodologie production...")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Charger données saison précédente
    last_season_data = get_last_season_matches()
    
    # Features xG à recalculer
    xg_features = [
        'home_xg_eff_10', 'away_xg_eff_10',
        'home_xg_eff_5', 'away_xg_eff_5',
        'home_xg_avg_10', 'away_xg_avg_10'
    ]
    
    # Initialiser nouvelles features
    for feature in xg_features:
        df[f'{feature}_corrected'] = np.nan
    
    # Traiter chaque équipe
    all_teams = list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    
    for team in all_teams:
        print(f"  🏟️  Traitement {team}...")
        
        if team in promoted_teams:
            # Équipes promues: valeur neutre 0.5
            print(f"    🔼 {team} promue → valeur neutre 0.5")
            
            # Home features
            home_mask = df['HomeTeam'] == team
            df.loc[home_mask, 'home_xg_eff_10_corrected'] = 0.5
            df.loc[home_mask, 'home_xg_eff_5_corrected'] = 0.5
            df.loc[home_mask, 'home_xg_avg_10_corrected'] = 1.0
            
            # Away features  
            away_mask = df['AwayTeam'] == team
            df.loc[away_mask, 'away_xg_eff_10_corrected'] = 0.5
            df.loc[away_mask, 'away_xg_eff_5_corrected'] = 0.5
            df.loc[away_mask, 'away_xg_avg_10_corrected'] = 1.0
            
        else:
            # Équipes existantes: rolling 10 derniers matchs 2024-25
            if not last_season_data.empty:
                team_last_season = last_season_data[
                    (last_season_data['HomeTeam'] == team) | 
                    (last_season_data['AwayTeam'] == team)
                ].tail(10)  # 10 derniers matchs
                
                if len(team_last_season) > 0:
                    print(f"    📊 {team} → rolling {len(team_last_season)} matchs 2024-25")
                    
                    # Calculer moyennes xG réelles fin 2024-25
                    home_matches = team_last_season[team_last_season['HomeTeam'] == team]
                    away_matches = team_last_season[team_last_season['AwayTeam'] == team]
                    
                    # xG efficiency = Goals/xG (efficacité réelle)
                    home_xg_eff = np.mean(
                        home_matches['HomeGoals'] / np.maximum(home_matches['HomeXG'], 0.1)
                    ) if len(home_matches) > 0 else 1.0
                    
                    away_xg_eff = np.mean(
                        away_matches['AwayGoals'] / np.maximum(away_matches['AwayXG'], 0.1)  
                    ) if len(away_matches) > 0 else 1.0
                    
                    # Limiter valeurs aberrantes
                    home_xg_eff = np.clip(home_xg_eff, 0.2, 3.0)
                    away_xg_eff = np.clip(away_xg_eff, 0.2, 3.0)
                    
                    # xG moyenne
                    home_xg_avg = np.mean(home_matches['HomeXG']) if len(home_matches) > 0 else 1.0
                    away_xg_avg = np.mean(away_matches['AwayXG']) if len(away_matches) > 0 else 1.0
                    
                    # Appliquer aux matchs 2025-26
                    home_mask = df['HomeTeam'] == team
                    df.loc[home_mask, 'home_xg_eff_10_corrected'] = home_xg_eff
                    df.loc[home_mask, 'home_xg_eff_5_corrected'] = home_xg_eff
                    df.loc[home_mask, 'home_xg_avg_10_corrected'] = home_xg_avg
                    
                    away_mask = df['AwayTeam'] == team
                    df.loc[away_mask, 'away_xg_eff_10_corrected'] = away_xg_eff
                    df.loc[away_mask, 'away_xg_eff_5_corrected'] = away_xg_eff
                    df.loc[away_mask, 'away_xg_avg_10_corrected'] = away_xg_avg
                    
                else:
                    print(f"    ⚠️  {team} sans données 2024-25 → valeur neutre")
                    # Fallback valeur neutre
                    home_mask = df['HomeTeam'] == team
                    df.loc[home_mask, 'home_xg_eff_10_corrected'] = 0.5
                    df.loc[home_mask, 'home_xg_eff_5_corrected'] = 0.5
                    df.loc[home_mask, 'home_xg_avg_10_corrected'] = 1.0
                    
                    away_mask = df['AwayTeam'] == team
                    df.loc[away_mask, 'away_xg_eff_10_corrected'] = 0.5
                    df.loc[away_mask, 'away_xg_eff_5_corrected'] = 0.5
                    df.loc[away_mask, 'away_xg_avg_10_corrected'] = 1.0
    
    # Remplacer anciennes features par nouvelles
    for feature in ['home_xg_eff_10', 'away_xg_eff_10']:
        if f'{feature}_corrected' in df.columns:
            df[feature] = df[f'{feature}_corrected']
            df = df.drop(columns=[f'{feature}_corrected'])
    
    # Ajouter target encoding si manquant
    if 'target' not in df.columns and 'FullTimeResult' in df.columns:
        df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        print("✅ Target encoding ajouté")
    
    print("✅ Features xG corrigées avec méthodologie production")
    return df

def test_corrected_model(df):
    """Test modèle avec features xG corrigées sur J1-J4"""
    print("\n🧪 TEST MODÈLE AVEC FEATURES XG CORRIGÉES")
    print("=" * 50)
    
    # Filtrer J1-J4 (exactement 40 premiers matchs saison 2025-26)
    df['Date'] = pd.to_datetime(df['Date'])
    df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
    df_test = df_season_2025.head(40).copy()
    
    print(f"Matchs test J1-J4: {len(df_test)}")
    print(f"Période: {df_test['Date'].min()} → {df_test['Date'].max()}")
    
    if len(df_test) < 30:
        print("⚠️  Pas assez de données J1-J4 pour test fiable")
        return None
    
    # Features production v2.3 (ordre exact du modèle)
    production_features = [
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
    ]
    
    # Vérifier features disponibles
    available_features = [f for f in production_features if f in df_test.columns]
    missing_features = [f for f in production_features if f not in df_test.columns]
    
    if missing_features:
        print(f"⚠️  Features manquantes: {missing_features}")
        return None
    
    print(f"✅ Features disponibles: {len(available_features)}/10")
    
    # Préparer données test
    X_test = df_test[available_features].fillna(0.5)
    y_test = df_test['target']
    
    print(f"Shape données test: X{X_test.shape}, y{y_test.shape}")
    
    # Charger modèle production v2.3
    try:
        model_path = 'models/v23_retrained_2025_09_11_154613.joblib'
        model = joblib.load(model_path)
        print(f"✅ Modèle chargé: {model_path}")
        
        # Prédictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 RÉSULTATS TEST J1-J4 (40 matchs)")
        print(f"Accuracy avec xG corrigé: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Distribution prédictions vs réalité
        print(f"\nDistribution réelle: H={np.mean(y_test==0):.1%}, D={np.mean(y_test==1):.1%}, A={np.mean(y_test==2):.1%}")
        print(f"Distribution prédite: H={np.mean(y_pred==0):.1%}, D={np.mean(y_pred==1):.1%}, A={np.mean(y_pred==2):.1%}")
        
        # Rapport détaillé
        print(f"\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Home', 'Draw', 'Away']))
        
        return accuracy
        
    except FileNotFoundError:
        print(f"❌ Modèle non trouvé: {model_path}")
        return None

def save_corrected_dataset(df):
    """Sauvegarde dataset avec corrections xG"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder dataset corrigé
    output_path = f'data/processed/v17_xg_corrected_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    
    # Métadonnées
    metadata = {
        'timestamp': timestamp,
        'corrections_applied': [
            'Suppression 5 features contextuelles non validées',
            'Correction rolling xG début saison',
            'Équipes promues: valeur neutre 0.5',
            'Équipes existantes: rolling 10 derniers matchs 2024-25'
        ],
        'promoted_teams_2025': ['Leeds', 'Sunderland', 'Burnley'],
        'methodology': 'Production realistic xG features',
        'features_count': len(df.columns),
        'matches_count': len(df),
        'test_period': 'J1-J4 (40 matches EPL 2025-26)'
    }
    
    metadata_path = f'data/processed/v17_xg_corrected_metadata_{timestamp}.json'
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Dataset corrigé sauvegardé: {output_path}")
    print(f"📋 Métadonnées: {metadata_path}")
    
    return output_path

def main():
    """Pipeline principal correction xG"""
    print("🚀 CORRECTION XG ROLLING - MÉTHODOLOGIE PRODUCTION")
    print("=" * 60)
    
    try:
        # 1. Charger et nettoyer dataset
        df = load_and_clean_dataset()
        
        # 2. Identifier équipes promues
        promoted_teams = identify_promoted_teams()
        
        # 3. Corriger features xG
        df_corrected = calculate_proper_xg_features(df, promoted_teams)
        
        # 4. Tester sur J1-J4
        test_accuracy = test_corrected_model(df_corrected)
        
        # 5. Sauvegarder dataset corrigé
        output_path = save_corrected_dataset(df_corrected)
        
        # Résumé final
        print(f"\n🎯 CORRECTION TERMINÉE")
        print(f"Dataset corrigé: {output_path}")
        if test_accuracy:
            print(f"Performance J1-J4: {test_accuracy:.1%}")
            
            if test_accuracy >= 0.47:
                print("✅ Performance satisfaisante (≥47%)")
            else:
                print("⚠️  Performance sous target (<47%)")
        
        print(f"\n📋 Méthodologie appliquée:")
        print(f"  • Équipes promues → xG neutre (0.5)")
        print(f"  • Équipes existantes → rolling 10 matchs 2024-25")
        print(f"  • Features contextuelles supprimées")
        print(f"  • Validation sur J1-J4 réels")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()