#!/usr/bin/env python3
"""
🔍 POST-MORTEM ANALYSIS J1-J4 - APPRENTISSAGE DURABLE

Objectif: Analyser performance sur 40 premiers matches pour comprendre les limites
et améliorer le modèle de façon durable pour J5+ (pas boost artificiel)

Approche scientifique: Diagnostic complet → Insights → Ajustements structurels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib
from datetime import datetime

def load_j1_j4_data():
    """Charger et préparer données J1-J4 avec prédictions"""
    print("📊 CHARGEMENT DONNÉES J1-J4")
    print("=" * 50)
    
    # Charger dataset original
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    cutoff_date = pd.Timestamp('2025-08-01')
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    # Features baseline
    baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Étendre avec matches J4 (simulation réaliste)
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
    
    # Créer features approximatives pour J4 (simulation features figées)
    j4_data = []
    for match in j4_matches:
        # Features neutres/moyennes pour simulation
        j4_row = {
            'Date': match['Date'],
            'Season': '2025-2026',
            'HomeTeam': match['HomeTeam'],
            'AwayTeam': match['AwayTeam'], 
            'FullTimeResult': match['FullTimeResult']
        }
        
        # Features approximatives (simplification pour analyse)
        for feat in baseline_features:
            if feat == 'matchday_normalized':
                j4_row[feat] = 4/38  # J4
            elif feat == 'h2h_score':
                j4_row[feat] = 0.5   # Neutre
            else:
                j4_row[feat] = 0.5   # Valeur neutre par défaut
        
        j4_data.append(j4_row)
    
    j4_df = pd.DataFrame(j4_data)
    
    # Fusionner J1-J3 + J4
    test_40 = pd.concat([test_df, j4_df], ignore_index=True)
    test_clean = test_40.dropna(subset=baseline_features + ['FullTimeResult'])
    
    print(f"📊 Dataset J1-J4: {len(test_clean)} matches")
    print(f"📊 J1-J3: {len(test_df)} matches") 
    print(f"📊 J4: {len(j4_df)} matches")
    
    return test_clean, baseline_features

def analyze_prediction_errors(test_data, features):
    """Analyse détaillée des erreurs de prédiction"""
    print("\n🎯 ANALYSE DÉTAILLÉE DES ERREURS")
    print("=" * 50)
    
    # Charger modèle et faire prédictions
    try:
        model = joblib.load('models/final_robust_model_20250915_163023.joblib')
    except FileNotFoundError:
        print("❌ Modèle non trouvé")
        return None
    
    X_test = test_data[features]
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy J1-J4: {accuracy:.3f} ({np.sum(y_pred == y_test)}/{len(y_test)})")
    
    # 1. MATRICE DE CONFUSION DÉTAILLÉE
    print(f"\n📊 MATRICE DE CONFUSION:")
    print("-" * 40)
    
    cm = confusion_matrix(y_test, y_pred)
    labels = ['H', 'D', 'A']
    
    print(f"      Prédit")
    print(f"     H   D   A")
    print(f"Réel")
    for i, label in enumerate(labels):
        row = f"{label}  [{' '.join(f'{cm[i][j]:3d}' for j in range(3))}]"
        print(row)
    
    # 2. ANALYSE PAR CLASSE
    print(f"\n📋 PERFORMANCE PAR CLASSE:")
    print("-" * 40)
    
    class_report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    for class_name in labels:
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall'] 
        f1 = class_report[class_name]['f1-score']
        support = int(class_report[class_name]['support'])
        
        print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (n={support})")
    
    # 3. ANALYSE DES ERREURS PAR ÉQUIPE
    print(f"\n🏆 ANALYSE PAR ÉQUIPE:")
    print("-" * 40)
    
    test_data_pred = test_data.copy()
    test_data_pred['predicted'] = [labels[p] for p in y_pred]
    test_data_pred['correct'] = (y_test == y_pred)
    
    # Erreurs par équipe (Home)
    home_errors = test_data_pred.groupby('HomeTeam').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(3)
    home_errors.columns = ['correct', 'total', 'accuracy']
    home_errors = home_errors.sort_values('accuracy')
    
    print("Équipes les MOINS bien prédites (à domicile):")
    worst_home = home_errors.head(5)
    for team, row in worst_home.iterrows():
        print(f"  {team:15}: {row['accuracy']:.1%} ({int(row['correct'])}/{int(row['total'])})")
    
    # Erreurs par équipe (Away)
    away_errors = test_data_pred.groupby('AwayTeam').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(3)
    away_errors.columns = ['correct', 'total', 'accuracy']
    away_errors = away_errors.sort_values('accuracy')
    
    print("\nÉquipes les MOINS bien prédites (à l'extérieur):")
    worst_away = away_errors.head(5)
    for team, row in worst_away.iterrows():
        print(f"  {team:15}: {row['accuracy']:.1%} ({int(row['correct'])}/{int(row['total'])})")
    
    # 4. ANALYSE DES PROBABILITÉS PRÉDITES
    print(f"\n🎲 ANALYSE PROBABILITÉS:")
    print("-" * 40)
    
    # Confiance moyenne par classe prédite
    for i, class_name in enumerate(labels):
        mask = y_pred == i
        if mask.any():
            avg_confidence = y_proba[mask, i].mean()
            max_proba = y_proba[mask, i].max()
            min_proba = y_proba[mask, i].min()
            print(f"{class_name} prédits: confiance moy={avg_confidence:.3f}, min={min_proba:.3f}, max={max_proba:.3f}")
    
    # Confiance des erreurs vs succès
    correct_mask = (y_test == y_pred)
    if correct_mask.any():
        correct_confidence = np.max(y_proba[correct_mask], axis=1).mean()
        incorrect_confidence = np.max(y_proba[~correct_mask], axis=1).mean()
        print(f"\nConfiance prédictions correctes: {correct_confidence:.3f}")
        print(f"Confiance prédictions incorrectes: {incorrect_confidence:.3f}")
        print(f"Différence: {correct_confidence - incorrect_confidence:+.3f}")
    
    # 5. MATCHES LES PLUS MAL PRÉDITS
    print(f"\n❌ TOP 10 ERREURS LES PLUS CONFIANTES:")
    print("-" * 40)
    
    # Erreurs avec haute confiance = pires erreurs du modèle
    error_data = []
    for i in range(len(test_data_pred)):
        if not test_data_pred.iloc[i]['correct']:
            confidence = np.max(y_proba[i])
            error_data.append({
                'index': i,
                'match': f"{test_data_pred.iloc[i]['HomeTeam']} vs {test_data_pred.iloc[i]['AwayTeam']}",
                'predicted': test_data_pred.iloc[i]['predicted'],
                'actual': test_data_pred.iloc[i]['FullTimeResult'], 
                'confidence': confidence,
                'date': test_data_pred.iloc[i]['Date'].strftime('%Y-%m-%d')
            })
    
    if error_data:
        error_df = pd.DataFrame(error_data).sort_values('confidence', ascending=False)
        for _, error in error_df.head(10).iterrows():
            print(f"  {error['date']} | {error['match']:25} | {error['predicted']}→{error['actual']} ({error['confidence']:.3f})")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_report': class_report,
        'worst_home': worst_home,
        'worst_away': worst_away,
        'error_analysis': error_df if error_data else None
    }

def analyze_feature_drift(test_data, features):
    """Analyser drift des features entre training et J1-J4"""
    print("\n📈 ANALYSE DRIFT DES FEATURES")
    print("=" * 50)
    
    # Charger données training pour comparaison
    df_full = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    cutoff_date = pd.Timestamp('2025-08-01')
    train_data = df_full[df_full['Date'] < cutoff_date]
    train_clean = train_data.dropna(subset=features + ['FullTimeResult'])
    
    print(f"📊 Training data: {len(train_clean)} matches")
    print(f"📊 J1-J4 data: {len(test_data)} matches")
    
    # Comparaison statistiques par feature
    print(f"\n📊 COMPARAISON STATISTIQUES:")
    print(f"{'Feature':25} | {'Train Mean':>10} | {'J1-J4 Mean':>10} | {'Drift':>8} | {'Train Std':>9} | {'J1-J4 Std':>9}")
    print("-" * 95)
    
    drift_analysis = []
    
    for feat in features:
        if feat in train_clean.columns and feat in test_data.columns:
            train_mean = train_clean[feat].mean()
            test_mean = test_data[feat].mean()
            train_std = train_clean[feat].std()
            test_std = test_data[feat].std()
            
            drift = abs(test_mean - train_mean) / train_std if train_std > 0 else 0
            
            print(f"{feat:25} | {train_mean:10.3f} | {test_mean:10.3f} | {drift:8.3f} | {train_std:9.3f} | {test_std:9.3f}")
            
            drift_analysis.append({
                'feature': feat,
                'train_mean': train_mean,
                'test_mean': test_mean, 
                'train_std': train_std,
                'test_std': test_std,
                'drift_score': drift
            })
    
    # Identifier features avec le plus de drift
    drift_df = pd.DataFrame(drift_analysis).sort_values('drift_score', ascending=False)
    
    print(f"\n🚨 TOP 5 FEATURES AVEC PLUS DE DRIFT:")
    print("-" * 40)
    for _, row in drift_df.head(5).iterrows():
        print(f"  {row['feature']:25}: {row['drift_score']:.3f} (drift score)")
    
    return drift_df

def analyze_feature_importance_j1_j4(test_data, features):
    """Analyser importance features spécifiquement sur J1-J4"""
    print("\n🔍 IMPORTANCE FEATURES SUR J1-J4")
    print("=" * 50)
    
    # Charger modèle
    try:
        model = joblib.load('models/final_robust_model_20250915_163023.joblib')
    except FileNotFoundError:
        print("❌ Modèle non trouvé")
        return None
    
    X_test = test_data[features]
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Permutation importance sur J1-J4
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
    )
    
    # Comparaison avec importance training (approximation via feature_importances_)
    train_importance = model.feature_importances_
    
    print(f"📊 COMPARAISON IMPORTANCE TRAINING vs J1-J4:")
    print(f"{'Feature':25} | {'Train Imp':>10} | {'J1-J4 Imp':>10} | {'Différence':>11}")
    print("-" * 70)
    
    importance_analysis = []
    
    for i, feat in enumerate(features):
        train_imp = train_importance[i]
        test_imp = perm_importance.importances_mean[i]
        diff = test_imp - train_imp
        
        print(f"{feat:25} | {train_imp:10.3f} | {test_imp:10.3f} | {diff:+11.3f}")
        
        importance_analysis.append({
            'feature': feat,
            'train_importance': train_imp,
            'test_importance': test_imp,
            'importance_diff': diff,
            'test_std': perm_importance.importances_std[i]
        })
    
    importance_df = pd.DataFrame(importance_analysis)
    
    # Features qui perdent le plus d'importance
    print(f"\n📉 FEATURES QUI PERDENT LE PLUS D'IMPORTANCE:")
    print("-" * 50)
    worst_features = importance_df.sort_values('importance_diff').head(3)
    for _, row in worst_features.iterrows():
        print(f"  {row['feature']:25}: {row['importance_diff']:+.3f}")
    
    # Features qui gagnent en importance  
    print(f"\n📈 FEATURES QUI GAGNENT EN IMPORTANCE:")
    print("-" * 50)
    best_features = importance_df.sort_values('importance_diff', ascending=False).head(3)
    for _, row in best_features.iterrows():
        print(f"  {row['feature']:25}: {row['importance_diff']:+.3f}")
    
    return importance_df

def identify_contextual_features():
    """Identifier features contextuelles simples pour début saison"""
    print("\n💡 FEATURES CONTEXTUELLES RECOMMANDÉES")
    print("=" * 50)
    
    recommendations = [
        {
            'name': 'rest_days_diff',
            'description': 'Différence jours repos entre équipes',
            'calculation': 'days_since_last_match_home - days_since_last_match_away',
            'rationale': 'Début saison: calendriers irréguliers, différences fatigue importantes',
            'implementation': 'Calculer depuis derniers matchs amicaux/officiels'
        },
        {
            'name': 'new_signings_impact', 
            'description': 'Impact transferts récents sur équipe',
            'calculation': 'sum(player_value * minutes_played) / total_minutes pour nouveaux joueurs',
            'rationale': 'Mercato août impacte performance début saison',
            'implementation': 'Base données transferts + temps jeu'
        },
        {
            'name': 'early_season_volatility',
            'description': 'Facteur volatilité historique début saison',
            'calculation': 'std(results_J1_J4_historical) par équipe',
            'rationale': 'Certaines équipes plus imprévisibles en début saison',
            'implementation': 'Calculer sur données 2019-2024'
        },
        {
            'name': 'promoted_team_factor',
            'description': 'Indicateur équipe promue avec ajustement',
            'calculation': '1.0 si promu, ajusté par performance Championship',
            'rationale': 'Promus = patterns différents, ELO initial approximatif',
            'implementation': 'Binary + pondération par niveau Championship'
        },
        {
            'name': 'manager_continuity',
            'description': 'Continuité équipe technique été',
            'calculation': '1.0 si même manager, 0.5 si changement récent',
            'rationale': 'Changement entraîneur été = adaptation nécessaire',
            'implementation': 'Base données changements managers'
        }
    ]
    
    print("🎯 TOP 5 FEATURES CONTEXTUELLES RECOMMANDÉES:")
    print("-" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Description: {rec['description']}")
        print(f"   Calcul: {rec['calculation']}")
        print(f"   Rationale: {rec['rationale']}")
        print(f"   Implémentation: {rec['implementation']}")
        print()
    
    return recommendations

def generate_j5_plus_recommendations(error_analysis, drift_analysis, importance_analysis):
    """Générer recommandations pour J5+ basées sur insights"""
    print("\n🚀 RECOMMANDATIONS POUR J5+")
    print("=" * 50)
    
    recommendations = []
    
    # 1. Basé sur analyse erreurs
    if error_analysis and 'accuracy' in error_analysis:
        acc = error_analysis['accuracy']
        if acc < 0.45:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'Calibration',
                'action': 'Ajuster seuils de décision pour classes sous-représentées',
                'rationale': f'Accuracy J1-J4 = {acc:.1%} < 45% baseline'
            })
    
    # 2. Basé sur drift features
    if drift_analysis is not None:
        high_drift = drift_analysis[drift_analysis['drift_score'] > 0.5]
        if len(high_drift) > 0:
            worst_feature = high_drift.iloc[0]['feature']
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'Feature Engineering',
                'action': f'Recalibrer {worst_feature} pour début saison',
                'rationale': f'Drift score élevé: {high_drift.iloc[0]["drift_score"]:.3f}'
            })
    
    # 3. Basé sur importance features
    if importance_analysis is not None:
        declining_features = importance_analysis[importance_analysis['importance_diff'] < -0.01]
        if len(declining_features) > 0:
            recommendations.append({
                'priority': 'LOW',
                'type': 'Model Architecture',
                'action': 'Réduire poids des features qui perdent importance début saison',
                'rationale': f'{len(declining_features)} features perdent importance significativement'
            })
    
    # 4. Recommandations générales
    recommendations.extend([
        {
            'priority': 'HIGH',
            'type': 'Monitoring',
            'action': 'Surveillance mensuelle performance avec seuil alerte <47%',
            'rationale': 'Détecter drift continu et ajuster rapidement'
        },
        {
            'priority': 'MEDIUM', 
            'type': 'Data Collection',
            'action': 'Intégrer features contextuelles début saison (rest_days, transferts)',
            'rationale': 'Capturer spécificités volatilité août-septembre'
        },
        {
            'priority': 'LOW',
            'type': 'Model Update',
            'action': 'Réentraîner modèle après J10 avec données début saison intégrées',
            'rationale': 'Incorporer patterns 2025-26 dans modèle pour suite saison'
        }
    ])
    
    # Trier par priorité
    priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations.sort(key=lambda x: priority_order[x['priority']])
    
    print("📋 PLAN D'ACTION STRUCTURÉ:")
    print("-" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[rec['priority']]
        print(f"{i}. {priority_emoji} [{rec['priority']}] {rec['type']}")
        print(f"   Action: {rec['action']}")
        print(f"   Rationale: {rec['rationale']}")
        print()
    
    return recommendations

def main():
    """Analyse complète post-mortem J1-J4"""
    print("🔍 POST-MORTEM ANALYSIS J1-J4 - APPRENTISSAGE DURABLE")
    print("=" * 80)
    
    # 1. Charger données
    test_data, features = load_j1_j4_data()
    
    # 2. Analyser erreurs en détail
    error_analysis = analyze_prediction_errors(test_data, features)
    
    # 3. Analyser drift des features
    drift_analysis = analyze_feature_drift(test_data, features)
    
    # 4. Analyser importance features sur J1-J4
    importance_analysis = analyze_feature_importance_j1_j4(test_data, features)
    
    # 5. Identifier features contextuelles
    contextual_features = identify_contextual_features()
    
    # 6. Générer recommandations pour J5+
    recommendations = generate_j5_plus_recommendations(error_analysis, drift_analysis, importance_analysis)
    
    # 7. Résumé exécutif
    print(f"\n📊 RÉSUMÉ EXÉCUTIF")
    print("=" * 50)
    if error_analysis:
        print(f"🎯 Performance J1-J4: {error_analysis['accuracy']:.1%}")
    print(f"📈 Features analysées: {len(features)}")
    if drift_analysis is not None:
        high_drift_count = len(drift_analysis[drift_analysis['drift_score'] > 0.5])
        print(f"🚨 Features avec drift élevé: {high_drift_count}")
    print(f"💡 Recommandations générées: {len(recommendations)}")
    print(f"🚀 Actions prioritaires: {len([r for r in recommendations if r['priority'] == 'HIGH'])}")
    
    print("\n✅ ANALYSE POST-MORTEM TERMINÉE!")
    print("🎯 Prochaine étape: Implémenter recommandations HIGH priority pour J5+")

if __name__ == "__main__":
    main()