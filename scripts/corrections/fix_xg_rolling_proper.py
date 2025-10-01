#!/usr/bin/env python3
"""
🔧 CORRECTION XG ROLLING - MÉTHODOLOGIE PRODUCTION

Problème: Features xG début saison mal calculées (rolling sur données inexistantes)

Solution correcte:
- Équipes EPL existantes: Rolling 10 derniers matchs saison 2024-25
- Équipes promues (Leeds, Sunderland, Burnley): Valeur neutre 0.5
- Test sur J1-J4 (40 matches) pour validation

Objectif: Comparer neutral (0.5) vs rolling correct pour améliorer 42.5% → 47-50%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime

def identify_promoted_teams():
    """Identifier équipes promues 2025-26"""
    # Équipes promues de Championship vers EPL 2025-26
    promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
    
    print(f"🔄 Équipes promues 2025-26: {promoted_teams}")
    return promoted_teams

def calculate_proper_xg_features():
    """Calculer features xG avec méthodologie correcte"""
    print("🔧 CALCUL FEATURES XG - MÉTHODOLOGIE CORRECTE")
    print("=" * 60)
    
    # Charger dataset
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    
    # Équipes promues
    promoted_teams = identify_promoted_teams()
    
    # Split saisons
    season_2024_end = pd.Timestamp('2025-05-31')  # Fin saison 2024-25
    season_2025_start = pd.Timestamp('2025-08-01')  # Début saison 2025-26
    
    # Données saison 2024-25 pour rolling
    season_2024_data = df[
        (df['Date'] >= pd.Timestamp('2024-08-01')) & 
        (df['Date'] <= season_2024_end)
    ].copy()
    
    # Données saison 2025-26 pour correction
    season_2025_data = df[df['Date'] >= season_2025_start].copy()
    
    print(f"📊 Saison 2024-25: {len(season_2024_data)} matches")
    print(f"📊 Saison 2025-26: {len(season_2025_data)} matches")
    
    # Calculer xG efficiency 10 derniers matchs pour chaque équipe saison 2024-25
    team_xg_end_2024 = {}
    
    print(f"\n🎯 CALCUL ROLLING XG SAISON 2024-25:")
    print("-" * 40)
    
    # Toutes les équipes présentes en 2024-25
    all_teams_2024 = set(
        list(season_2024_data['HomeTeam'].unique()) + 
        list(season_2024_data['AwayTeam'].unique())
    )
    
    for team in all_teams_2024:
        if team in promoted_teams:
            # Équipe promue = pas de données EPL → skip
            continue
            
        # Matches de cette équipe en 2024-25
        team_matches_2024 = season_2024_data[
            (season_2024_data['HomeTeam'] == team) | 
            (season_2024_data['AwayTeam'] == team)
        ].sort_values('Date')
        
        if len(team_matches_2024) >= 10:
            # 10 derniers matchs de la saison
            last_10_matches = team_matches_2024.tail(10)
            
            # Calculer xG efficiency sur ces matchs
            # Note: Utiliser features existantes comme proxy
            home_xg_values = []
            away_xg_values = []
            
            for _, match in last_10_matches.iterrows():
                if match['HomeTeam'] == team:
                    # Équipe jouait à domicile
                    if 'home_xg_eff_10' in match and not pd.isna(match['home_xg_eff_10']):
                        home_xg_values.append(match['home_xg_eff_10'])
                else:
                    # Équipe jouait à l'extérieur  
                    if 'away_xg_eff_10' in match and not pd.isna(match['away_xg_eff_10']):
                        away_xg_values.append(match['away_xg_eff_10'])
            
            # Moyenner xG efficiency (approximation)
            if home_xg_values:
                avg_home_xg = np.mean(home_xg_values)
            else:
                avg_home_xg = 0.5  # Neutre si pas de données
                
            if away_xg_values:
                avg_away_xg = np.mean(away_xg_values)
            else:
                avg_away_xg = 0.5  # Neutre si pas de données
            
            team_xg_end_2024[team] = {
                'home_xg_eff': avg_home_xg,
                'away_xg_eff': avg_away_xg,
                'matches_analyzed': len(last_10_matches)
            }
            
            print(f"  {team:15}: home={avg_home_xg:.3f}, away={avg_away_xg:.3f} ({len(last_10_matches)} matchs)")
        
        else:
            # Pas assez de matchs → neutre
            team_xg_end_2024[team] = {
                'home_xg_eff': 0.5,
                'away_xg_eff': 0.5,
                'matches_analyzed': len(team_matches_2024)
            }
            print(f"  {team:15}: NEUTRE (seulement {len(team_matches_2024)} matchs)")
    
    # Équipes promues = neutre
    print(f"\n🔄 ÉQUIPES PROMUES (NEUTRE):")
    print("-" * 30)
    for team in promoted_teams:
        team_xg_end_2024[team] = {
            'home_xg_eff': 0.5,
            'away_xg_eff': 0.5, 
            'matches_analyzed': 0
        }
        print(f"  {team:15}: home=0.500, away=0.500 (promu)")
    
    print(f"\n✅ Features xG calculées pour {len(team_xg_end_2024)} équipes")
    
    return team_xg_end_2024

def create_corrected_dataset(team_xg_corrections):
    """Créer dataset avec features xG corrigées"""
    print(f"\n📊 CRÉATION DATASET CORRIGÉ")
    print("=" * 40)
    
    # Charger dataset original
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    df_corrected = df.copy()
    
    # Appliquer corrections pour saison 2025-26
    season_2025_start = pd.Timestamp('2025-08-01')
    season_2025_mask = df_corrected['Date'] >= season_2025_start
    
    corrections_applied = 0
    
    for idx in df_corrected[season_2025_mask].index:
        home_team = df_corrected.loc[idx, 'HomeTeam']
        away_team = df_corrected.loc[idx, 'AwayTeam']
        
        # Appliquer corrections basées sur fin saison 2024-25
        if home_team in team_xg_corrections:
            df_corrected.loc[idx, 'home_xg_eff_10'] = team_xg_corrections[home_team]['home_xg_eff']
            corrections_applied += 1
            
        if away_team in team_xg_corrections:
            df_corrected.loc[idx, 'away_xg_eff_10'] = team_xg_corrections[away_team]['away_xg_eff']
            corrections_applied += 1
    
    print(f"✅ Corrections appliquées: {corrections_applied}")
    
    return df_corrected

def test_rolling_vs_neutral_performance():
    """Tester performance rolling correct vs neutral sur J1-J4"""
    print(f"\n🧪 TEST ROLLING CORRECT VS NEUTRAL")
    print("=" * 60)
    
    # 1. Calculer corrections rolling
    team_xg_corrections = calculate_proper_xg_features()
    df_rolling = create_corrected_dataset(team_xg_corrections)
    
    # 2. Créer dataset neutral (référence)
    df_neutral = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    season_2025_mask = df_neutral['Date'] >= pd.Timestamp('2025-08-01')
    
    # Appliquer neutral pour saison 2025-26
    df_neutral.loc[season_2025_mask, 'home_xg_eff_10'] = 0.5
    df_neutral.loc[season_2025_mask, 'away_xg_eff_10'] = 0.5
    
    # 3. Préparer test J1-J4 (40 matches)
    def add_j4_matches(df_base):
        j4_matches = [
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Arsenal', 'AwayTeam': 'Nottingham Forest', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Bournemouth', 'AwayTeam': 'Brighton', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Crystal Palace', 'AwayTeam': 'Sunderland', 'FullTimeResult': 'D'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Everton', 'AwayTeam': 'Aston Villa', 'FullTimeResult': 'D'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Fulham', 'AwayTeam': 'Leeds', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Newcastle', 'AwayTeam': 'Wolves', 'FullTimeResult': 'H'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'West Ham', 'AwayTeam': 'Tottenham', 'FullTimeResult': 'A'},
            {'Date': pd.Timestamp('2025-09-13'), 'HomeTeam': 'Brentford', 'AwayTeam': 'Chelsea', 'FullTimeResult': 'D'},
            {'Date': pd.Timestamp('2025-09-14'), 'HomeTeam': 'Burnley', 'AwayTeam': 'Liverpool', 'FullTimeResult': 'A'},
            {'Date': pd.Timestamp('2025-09-14'), 'HomeTeam': 'Manchester City', 'AwayTeam': 'Man United', 'FullTimeResult': 'H'}
        ]
        
        for match in j4_matches:
            match['Season'] = '2025-2026'
            
            # Features de base (par défaut)
            for feat in ['form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'shots_diff_normalized', 
                        'corners_diff_normalized', 'market_entropy_norm', 'away_goals_sum_5']:
                match[feat] = 0.5
            
            match['matchday_normalized'] = 4/38
            
            # Features xG seront appliquées selon méthode
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            if home_team in team_xg_corrections:
                match['home_xg_eff_10'] = team_xg_corrections[home_team]['home_xg_eff']
            else:
                match['home_xg_eff_10'] = 0.5
                
            if away_team in team_xg_corrections:
                match['away_xg_eff_10'] = team_xg_corrections[away_team]['away_xg_eff']
            else:
                match['away_xg_eff_10'] = 0.5
        
        j4_df = pd.DataFrame(j4_matches)
        return pd.concat([df_base, j4_df], ignore_index=True)
    
    # Créer datasets test 40 matches
    cutoff_date = pd.Timestamp('2025-08-01')
    
    # Rolling dataset
    test_rolling_j1j3 = df_rolling[df_rolling['Date'] >= cutoff_date]
    test_rolling_40 = add_j4_matches(test_rolling_j1j3)
    
    # Neutral dataset  
    test_neutral_j1j3 = df_neutral[df_neutral['Date'] >= cutoff_date]
    test_neutral_40 = add_j4_matches(test_neutral_j1j3)
    
    # Features baseline
    baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Charger modèle
    try:
        model = joblib.load('models/final_robust_model_20250915_163023.joblib')
        print("✅ Modèle chargé")
    except FileNotFoundError:
        print("❌ Modèle non trouvé")
        return None
    
    # 4. Test performance
    print(f"\n📊 COMPARAISON PERFORMANCE:")
    print(f"{'Méthode':20} | {'Accuracy':>8} | {'Matches':>7} | {'Amélioration':>12}")
    print("-" * 65)
    
    results = {}
    
    for method_name, test_data in [('Neutral', test_neutral_40), ('Rolling', test_rolling_40)]:
        test_clean = test_data.dropna(subset=baseline_features + ['FullTimeResult'])
        
        if len(test_clean) > 0:
            X_test = test_clean[baseline_features]
            y_test = test_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[method_name] = {
                'accuracy': accuracy,
                'correct': np.sum(y_pred == y_test),
                'total': len(y_test)
            }
            
            improvement = 0 if method_name == 'Neutral' else (accuracy - results['Neutral']['accuracy'])
            
            print(f"{method_name:20} | {accuracy:8.3f} | {len(test_clean):7d} | {improvement:+12.3f}")
    
    # 5. Analyse détaillée
    if 'Rolling' in results and 'Neutral' in results:
        improvement = results['Rolling']['accuracy'] - results['Neutral']['accuracy']
        
        print(f"\n🎯 RÉSULTATS DÉTAILLÉS:")
        print("-" * 40)
        print(f"Méthode neutral: {results['Neutral']['accuracy']:.1%} ({results['Neutral']['correct']}/{results['Neutral']['total']})")
        print(f"Méthode rolling: {results['Rolling']['accuracy']:.1%} ({results['Rolling']['correct']}/{results['Rolling']['total']})")
        print(f"Amélioration: {improvement:+.1%} ({improvement*100:+.1f}pp)")
        
        # Analyse par équipe
        print(f"\n🔍 ANALYSE PAR TYPE ÉQUIPE:")
        print("-" * 30)
        
        promoted_teams = identify_promoted_teams()
        
        # Compter matches avec équipes promues vs existantes
        test_data_rolling = test_rolling_40.dropna(subset=baseline_features + ['FullTimeResult'])
        
        promoted_matches = 0
        existing_matches = 0
        
        for _, match in test_data_rolling.iterrows():
            home_promoted = match['HomeTeam'] in promoted_teams
            away_promoted = match['AwayTeam'] in promoted_teams
            
            if home_promoted or away_promoted:
                promoted_matches += 1
            else:
                existing_matches += 1
        
        print(f"Matches avec équipes promues: {promoted_matches}")
        print(f"Matches équipes existantes: {existing_matches}")
        
        # Recommandation
        print(f"\n💡 RECOMMANDATION:")
        print("-" * 20)
        if improvement > 0.01:  # +1pp
            print("✅ Rolling methodology RECOMMANDÉE")
            print(f"🚀 Gain significatif: {improvement*100:+.1f}pp")
            print("📊 À déployer pour J5+")
        elif improvement > 0:
            print("🟡 Rolling methodology MARGINALE")
            print("🔧 Gain faible mais méthodologie plus robuste")
            print("💭 À considérer pour cohérence production")
        else:
            print("❌ Pas d'amélioration détectée")
            print("🔄 Conserver approche neutral plus simple")
    
    return results

def save_rolling_corrected_model():
    """Sauvegarder pipeline avec rolling corrections"""
    print(f"\n💾 SAUVEGARDE PIPELINE ROLLING")
    print("-" * 40)
    
    # Calculer corrections
    team_xg_corrections = calculate_proper_xg_features()
    
    # Créer code pipeline
    pipeline_code = f'''#!/usr/bin/env python3
"""
🎯 PIPELINE XG ROLLING CORRECT - PRODUCTION
Auto-généré le {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Méthodologie:
- Équipes EPL existantes: Rolling 10 derniers matchs saison 2024-25  
- Équipes promues: Valeur neutre 0.5
- Application: Saison 2025-26 début saison seulement
"""

import pandas as pd
import numpy as np
import joblib

# Corrections calculées fin saison 2024-25
TEAM_XG_CORRECTIONS = {team_xg_corrections}

PROMOTED_TEAMS_2025 = {identify_promoted_teams()}

def apply_rolling_xg_corrections(df):
    """Appliquer corrections xG rolling pour saison 2025-26"""
    df_corrected = df.copy()
    
    season_2025_start = pd.Timestamp('2025-08-01')
    season_2025_mask = df_corrected['Date'] >= season_2025_start
    
    corrections_applied = 0
    
    for idx in df_corrected[season_2025_mask].index:
        home_team = df_corrected.loc[idx, 'HomeTeam']
        away_team = df_corrected.loc[idx, 'AwayTeam']
        
        # Correction home team
        if home_team in TEAM_XG_CORRECTIONS:
            df_corrected.loc[idx, 'home_xg_eff_10'] = TEAM_XG_CORRECTIONS[home_team]['home_xg_eff']
            corrections_applied += 1
            
        # Correction away team
        if away_team in TEAM_XG_CORRECTIONS:
            df_corrected.loc[idx, 'away_xg_eff_10'] = TEAM_XG_CORRECTIONS[away_team]['away_xg_eff']
            corrections_applied += 1
    
    print(f"✅ Rolling XG corrections appliquées: {{corrections_applied}}")
    return df_corrected

def predict_with_rolling_corrections(df, model_path='models/final_robust_model_20250915_163023.joblib'):
    """Prédiction avec corrections rolling XG"""
    # Charger modèle
    model = joblib.load(model_path)
    
    # Appliquer corrections
    df_corrected = apply_rolling_xg_corrections(df)
    
    # Features baseline
    features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Prédictions
    X = df_corrected[features]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return predictions, probabilities, df_corrected

if __name__ == "__main__":
    print("🔧 Pipeline XG Rolling - Production Ready!")
'''
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_path = f"scripts/corrections/rolling_xg_pipeline_{timestamp}.py"
    
    with open(pipeline_path, 'w') as f:
        f.write(pipeline_code)
    
    print(f"✅ Pipeline sauvé: {pipeline_path}")
    return pipeline_path

def main():
    """Exécution complète test rolling vs neutral"""
    print("🔧 TEST MÉTHODOLOGIE XG ROLLING CORRECTE")
    print("=" * 80)
    
    # Test performance
    results = test_rolling_vs_neutral_performance()
    
    # Sauvegarder pipeline si amélioration
    if results and 'Rolling' in results and 'Neutral' in results:
        improvement = results['Rolling']['accuracy'] - results['Neutral']['accuracy']
        
        if improvement >= 0:  # Même si marginal, méthodologie plus robuste
            pipeline_path = save_rolling_corrected_model()
            print(f"\n✅ Pipeline rolling sauvé: {pipeline_path}")
    
    # Résumé final
    print(f"\n🎯 RÉSUMÉ FINAL:")
    print("=" * 40)
    if results:
        print(f"✅ Test complété sur 40 matches J1-J4")
        print(f"📊 Neutral: {results['Neutral']['accuracy']:.1%}")
        print(f"📊 Rolling: {results['Rolling']['accuracy']:.1%}")  
        improvement = results['Rolling']['accuracy'] - results['Neutral']['accuracy']
        print(f"🚀 Amélioration: {improvement:+.1%}")
        
        if improvement > 0.01:
            print("💪 Rolling methodology validée pour J5+!")
        elif improvement > 0:
            print("🔧 Rolling methodology cohérente, à considérer")
        else:
            print("🔄 Conserver neutral pour simplicité")
    
    print(f"\n✅ ANALYSE TERMINÉE!")

if __name__ == "__main__":
    main()