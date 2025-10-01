#!/usr/bin/env python3
"""
🚀 PHASE 2 - FEATURE ENGINEERING INTELLIGENT

Implémentation des 4 nouvelles features "quick wins":
1. form_strength_adjusted - Form pondérée par force adversaires  
2. form_momentum - Accélération/décélération de la forme
3. fixture_density_14d - Fatigue équipes (matches 14 derniers jours)
4. european_hangover - Flag match après coupe Europe

BASELINE: RandomForest 50.0% avec 10 features optimisées
OBJECTIF: 50.0% → 52-55% avec features intelligentes
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
from datetime import datetime, timedelta
import json

def load_base_data():
    """Charger dataset de base avec validation"""
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    
    # Ajouter colonnes équipes nécessaires pour feature engineering  
    print(f"📊 Dataset original: {len(df)} matches")
    print(f"🗓️ Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def calculate_form_strength_adjusted(df):
    """
    Feature 1: form_strength_adjusted
    Form pondérée par la force ELO des adversaires récents
    """
    print("🔧 Feature 1: form_strength_adjusted")
    
    # Créer mapping ELO par équipe et date
    df_sorted = df.sort_values(['Date']).copy()
    
    # Colonnes pour nouvelle feature
    df_sorted['home_form_strength_adj'] = 0.0
    df_sorted['away_form_strength_adj'] = 0.0
    
    # Pour chaque match, calculer form ajustée sur 5 derniers matches
    window_size = 5
    
    for idx, row in df_sorted.iterrows():
        team_home = row['HomeTeam']
        team_away = row['AwayTeam']
        match_date = row['Date']
        
        # Matches précédents pour home team (5 derniers)
        home_history = df_sorted[
            (df_sorted['Date'] < match_date) & 
            ((df_sorted['HomeTeam'] == team_home) | (df_sorted['AwayTeam'] == team_home))
        ].tail(window_size)
        
        # Calcul form strength adjusted pour home
        if len(home_history) > 0:
            home_results = []
            home_opp_strengths = []
            
            for _, hist_match in home_history.iterrows():
                # Résultat du point de vue de team_home
                if hist_match['HomeTeam'] == team_home:
                    result = 1 if hist_match['FullTimeResult'] == 'H' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                    opponent = hist_match['AwayTeam']
                else:  # team_home était away
                    result = 1 if hist_match['FullTimeResult'] == 'A' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                    opponent = hist_match['HomeTeam']
                
                # Force de l'adversaire (proxy par elo_diff actuel - approximation)
                opp_strength = 0.5  # Default strength
                if 'elo_diff_normalized' in hist_match:
                    # Approximation de la force adversaire
                    opp_strength = 0.5 + abs(float(hist_match['elo_diff_normalized']) - 0.5) * 0.5
                
                home_results.append(result)
                home_opp_strengths.append(opp_strength)
            
            # Form strength adjusted = moyenne pondérée par force adversaire
            if len(home_results) > 0:
                weights = np.array(home_opp_strengths)
                if weights.sum() > 0:
                    home_form_adj = np.average(home_results, weights=weights)
                else:
                    home_form_adj = np.mean(home_results)
                df_sorted.loc[idx, 'home_form_strength_adj'] = home_form_adj
        
        # Même calcul pour away team
        away_history = df_sorted[
            (df_sorted['Date'] < match_date) & 
            ((df_sorted['HomeTeam'] == team_away) | (df_sorted['AwayTeam'] == team_away))
        ].tail(window_size)
        
        if len(away_history) > 0:
            away_results = []
            away_opp_strengths = []
            
            for _, hist_match in away_history.iterrows():
                if hist_match['HomeTeam'] == team_away:
                    result = 1 if hist_match['FullTimeResult'] == 'H' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                else:
                    result = 1 if hist_match['FullTimeResult'] == 'A' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                
                opp_strength = 0.5
                if 'elo_diff_normalized' in hist_match:
                    opp_strength = 0.5 + abs(float(hist_match['elo_diff_normalized']) - 0.5) * 0.5
                
                away_results.append(result)
                away_opp_strengths.append(opp_strength)
            
            if len(away_results) > 0:
                weights = np.array(away_opp_strengths)
                if weights.sum() > 0:
                    away_form_adj = np.average(away_results, weights=weights)
                else:
                    away_form_adj = np.mean(away_results)
                df_sorted.loc[idx, 'away_form_strength_adj'] = away_form_adj
    
    # Créer feature différentielle
    df_sorted['form_strength_adjusted'] = df_sorted['home_form_strength_adj'] - df_sorted['away_form_strength_adj']
    
    # Normaliser entre 0 et 1
    if df_sorted['form_strength_adjusted'].std() > 0:
        min_val = df_sorted['form_strength_adjusted'].min()
        max_val = df_sorted['form_strength_adjusted'].max()
        df_sorted['form_strength_adjusted'] = (df_sorted['form_strength_adjusted'] - min_val) / (max_val - min_val)
    
    print(f"✅ form_strength_adjusted calculée - Range: {df_sorted['form_strength_adjusted'].min():.3f} to {df_sorted['form_strength_adjusted'].max():.3f}")
    
    return df_sorted

def calculate_form_momentum(df):
    """
    Feature 2: form_momentum  
    Trend de la forme (s'améliore/se dégrade)
    """
    print("🔧 Feature 2: form_momentum")
    
    df['home_form_momentum'] = 0.0
    df['away_form_momentum'] = 0.0
    
    window_size = 4  # 4 matches pour calculer trend
    
    df_sorted = df.sort_values('Date').copy()
    
    for idx, row in df_sorted.iterrows():
        team_home = row['HomeTeam']
        team_away = row['AwayTeam']
        match_date = row['Date']
        
        # Home team momentum
        home_history = df_sorted[
            (df_sorted['Date'] < match_date) & 
            ((df_sorted['HomeTeam'] == team_home) | (df_sorted['AwayTeam'] == team_home))
        ].tail(window_size)
        
        if len(home_history) >= 3:  # Minimum pour calculer trend
            home_results = []
            for _, hist_match in home_history.iterrows():
                if hist_match['HomeTeam'] == team_home:
                    result = 1 if hist_match['FullTimeResult'] == 'H' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                else:
                    result = 1 if hist_match['FullTimeResult'] == 'A' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                home_results.append(result)
            
            # Calculer trend linéaire simple
            if len(home_results) >= 2:
                x = np.arange(len(home_results))
                momentum = np.polyfit(x, home_results, 1)[0]  # Slope de la régression linéaire
                df_sorted.loc[idx, 'home_form_momentum'] = momentum
        
        # Away team momentum
        away_history = df_sorted[
            (df_sorted['Date'] < match_date) & 
            ((df_sorted['HomeTeam'] == team_away) | (df_sorted['AwayTeam'] == team_away))
        ].tail(window_size)
        
        if len(away_history) >= 3:
            away_results = []
            for _, hist_match in away_history.iterrows():
                if hist_match['HomeTeam'] == team_away:
                    result = 1 if hist_match['FullTimeResult'] == 'H' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                else:
                    result = 1 if hist_match['FullTimeResult'] == 'A' else 0.5 if hist_match['FullTimeResult'] == 'D' else 0
                away_results.append(result)
            
            if len(away_results) >= 2:
                x = np.arange(len(away_results))
                momentum = np.polyfit(x, away_results, 1)[0]
                df_sorted.loc[idx, 'away_form_momentum'] = momentum
    
    # Feature différentielle
    df_sorted['form_momentum'] = df_sorted['home_form_momentum'] - df_sorted['away_form_momentum']
    
    # Normaliser 
    if df_sorted['form_momentum'].std() > 0:
        mean_val = df_sorted['form_momentum'].mean()
        std_val = df_sorted['form_momentum'].std()
        df_sorted['form_momentum'] = (df_sorted['form_momentum'] - mean_val) / (4 * std_val) + 0.5
        df_sorted['form_momentum'] = np.clip(df_sorted['form_momentum'], 0, 1)
    
    print(f"✅ form_momentum calculée - Range: {df_sorted['form_momentum'].min():.3f} to {df_sorted['form_momentum'].max():.3f}")
    
    return df_sorted

def calculate_fixture_density(df):
    """
    Feature 3: fixture_density_14d
    Nombre de matches par équipe dans les 14 derniers jours (fatigue)
    """
    print("🔧 Feature 3: fixture_density_14d")
    
    df['home_fixture_density'] = 0.0
    df['away_fixture_density'] = 0.0
    
    df_sorted = df.sort_values('Date').copy()
    
    for idx, row in df_sorted.iterrows():
        team_home = row['HomeTeam']
        team_away = row['AwayTeam']
        match_date = row['Date']
        
        # Date 14 jours avant
        start_date = match_date - timedelta(days=14)
        
        # Matches home team dans les 14 derniers jours
        home_recent = df_sorted[
            (df_sorted['Date'] >= start_date) & 
            (df_sorted['Date'] < match_date) &
            ((df_sorted['HomeTeam'] == team_home) | (df_sorted['AwayTeam'] == team_home))
        ]
        df_sorted.loc[idx, 'home_fixture_density'] = len(home_recent)
        
        # Matches away team dans les 14 derniers jours
        away_recent = df_sorted[
            (df_sorted['Date'] >= start_date) & 
            (df_sorted['Date'] < match_date) &
            ((df_sorted['HomeTeam'] == team_away) | (df_sorted['AwayTeam'] == team_away))
        ]
        df_sorted.loc[idx, 'away_fixture_density'] = len(away_recent)
    
    # Feature différentielle (équipe plus fatiguée a désavantage)
    df_sorted['fixture_density_14d'] = df_sorted['away_fixture_density'] - df_sorted['home_fixture_density']
    
    # Normaliser (plus de matches récents = plus de fatigue = score plus bas)
    if df_sorted['fixture_density_14d'].std() > 0:
        min_val = df_sorted['fixture_density_14d'].min()
        max_val = df_sorted['fixture_density_14d'].max()
        if max_val != min_val:
            df_sorted['fixture_density_14d'] = (df_sorted['fixture_density_14d'] - min_val) / (max_val - min_val)
    
    print(f"✅ fixture_density_14d calculée - Range: {df_sorted['fixture_density_14d'].min():.3f} to {df_sorted['fixture_density_14d'].max():.3f}")
    
    return df_sorted

def calculate_european_hangover(df):
    """
    Feature 4: european_hangover
    Flag binaire si équipe a joué en coupe Europe en milieu de semaine
    (Approximation: match le jeudi précédent)
    """
    print("🔧 Feature 4: european_hangover")
    
    df['home_european_hangover'] = 0.0
    df['away_european_hangover'] = 0.0
    
    df_sorted = df.sort_values('Date').copy()
    
    # Liste équipes participant généralement aux coupes européennes
    european_teams = [
        'Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Tottenham', 'Man United',
        'Newcastle', 'Brighton', 'Aston Villa', 'West Ham'
    ]
    
    for idx, row in df_sorted.iterrows():
        team_home = row['HomeTeam']
        team_away = row['AwayTeam']
        match_date = row['Date']
        
        # Si match le weekend après jeudi (jour 3 = jeudi)
        # Approximation: match weekend après jeudi = possible hangover
        if match_date.weekday() in [5, 6]:  # Samedi ou dimanche
            # Vérifier si jeudi précédent était jour de coupe
            thursday_before = match_date - timedelta(days=(match_date.weekday() - 3))
            
            # Home team hangover (si équipe européenne)
            if team_home in european_teams:
                df_sorted.loc[idx, 'home_european_hangover'] = 1.0
            
            # Away team hangover
            if team_away in european_teams:
                df_sorted.loc[idx, 'away_european_hangover'] = 1.0
    
    # Feature différentielle
    df_sorted['european_hangover'] = df_sorted['away_european_hangover'] - df_sorted['home_european_hangover']
    
    # Normaliser entre 0 et 1
    df_sorted['european_hangover'] = (df_sorted['european_hangover'] + 1) / 2
    
    hangover_count = (df_sorted['european_hangover'] != 0.5).sum()
    print(f"✅ european_hangover calculée - {hangover_count} matches avec hangover détecté")
    
    return df_sorted

def test_new_features():
    """
    Test des nouvelles features vs baseline
    """
    print()
    print("🚀 PHASE 2 - FEATURE ENGINEERING TEST")
    print("=" * 60)
    
    # Charger et préparer données
    df_base = load_base_data()
    
    # Calculer nouvelles features
    print("\n🔧 CALCUL NOUVELLES FEATURES:")
    df_enhanced = calculate_form_strength_adjusted(df_base)
    df_enhanced = calculate_form_momentum(df_enhanced)
    df_enhanced = calculate_fixture_density(df_enhanced) 
    df_enhanced = calculate_european_hangover(df_enhanced)
    
    # Split temporel
    cutoff_date = pd.Timestamp('2025-08-01')
    train_df = df_enhanced[df_enhanced['Date'] < cutoff_date].copy()
    test_df = df_enhanced[df_enhanced['Date'] >= cutoff_date].copy()
    
    # Features baseline (Phase 1)
    baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Nouvelles features
    new_features = [
        'form_strength_adjusted', 'form_momentum', 
        'fixture_density_14d', 'european_hangover'
    ]
    
    # Features combinées
    all_features = baseline_features + new_features
    
    # Nettoyer données
    train_clean = train_df.dropna(subset=all_features + ['FullTimeResult'])
    test_clean = test_df.dropna(subset=all_features + ['FullTimeResult'])
    
    print(f"\n📊 Dataset final: {len(train_clean)} train, {len(test_clean)} test")
    
    # Préparer données
    y_train = train_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    y_test = test_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print("\n🧪 TESTS COMPARATIFS:")
    print("-" * 50)
    
    # Modèle RandomForest standard
    rf_config = {
        'n_estimators': 300,
        'max_depth': 20,
        'max_features': 'sqrt',
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    results = {}
    
    # 1. Baseline (10 features)
    print("🏁 Test 1: Baseline (10 features)")
    X_train_base = train_clean[baseline_features]
    X_test_base = test_clean[baseline_features]
    
    rf_baseline = RandomForestClassifier(**rf_config)
    rf_baseline.fit(X_train_base, y_train)
    
    baseline_acc = accuracy_score(y_test, rf_baseline.predict(X_test_base))
    print(f"   Accuracy: {baseline_acc:.3f}")
    
    results['baseline'] = {
        'features': baseline_features,
        'accuracy': baseline_acc,
        'feature_count': len(baseline_features)
    }
    
    # 2. Test chaque nouvelle feature individuellement
    for new_feat in new_features:
        print(f"\n🧪 Test: Baseline + {new_feat}")
        
        combined_features = baseline_features + [new_feat]
        X_train_comb = train_clean[combined_features] 
        X_test_comb = test_clean[combined_features]
        
        rf_individual = RandomForestClassifier(**rf_config)
        rf_individual.fit(X_train_comb, y_train)
        
        individual_acc = accuracy_score(y_test, rf_individual.predict(X_test_comb))
        improvement = (individual_acc - baseline_acc) * 100
        
        status = "✅" if improvement > 0.5 else "⚠️" if improvement > 0 else "❌"
        print(f"   Accuracy: {individual_acc:.3f} ({improvement:+.1f}pp) {status}")
        
        results[f'baseline_plus_{new_feat}'] = {
            'features': combined_features,
            'accuracy': individual_acc,
            'improvement': improvement,
            'feature_count': len(combined_features)
        }
    
    # 3. Toutes nouvelles features
    print(f"\n🎯 Test: Toutes features (14 total)")
    X_train_all = train_clean[all_features]
    X_test_all = test_clean[all_features]
    
    rf_full = RandomForestClassifier(**rf_config)
    rf_full.fit(X_train_all, y_train)
    
    y_pred_full = rf_full.predict(X_test_all)
    full_acc = accuracy_score(y_test, y_pred_full)
    full_improvement = (full_acc - baseline_acc) * 100
    
    # Analyse détaillée
    class_report = classification_report(y_test, y_pred_full, target_names=['H', 'D', 'A'], output_dict=True)
    draw_recall = class_report['D']['recall']
    
    status = "🏆" if full_improvement > 2 else "✅" if full_improvement > 0.5 else "⚠️"
    print(f"   Accuracy: {full_acc:.3f} ({full_improvement:+.1f}pp) {status}")
    print(f"   Draw Recall: {draw_recall:.3f}")
    
    results['full_enhanced'] = {
        'features': all_features,
        'accuracy': full_acc,
        'improvement': full_improvement,
        'feature_count': len(all_features),
        'draw_recall': draw_recall,
        'class_report': class_report
    }
    
    # RECOMMANDATION FINALE
    print()
    print("=" * 60)
    print("🏆 RECOMMANDATION FINALE:")
    print("=" * 60)
    
    # Trouver meilleure configuration
    best_config = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_data = results[best_config]
    
    print(f"\n🥇 Meilleure configuration: {best_config}")
    print(f"📈 Performance: {best_data['accuracy']:.3f}")
    print(f"🎯 Amélioration: {(best_data['accuracy'] - baseline_acc)*100:+.1f}pp")
    print(f"📋 Features ({best_data['feature_count']}): {len(best_data['features'])}")
    
    # Sauvegarder meilleur modèle
    if best_config == 'full_enhanced':
        model_to_save = rf_full
        features_to_save = all_features
    else:
        # Re-entraîner meilleur modèle si pas le full
        best_features = best_data['features']
        X_train_best = train_clean[best_features]
        model_to_save = RandomForestClassifier(**rf_config)
        model_to_save.fit(X_train_best, y_train)
        features_to_save = best_features
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/phase2_enhanced_model_{timestamp}.joblib"
    
    joblib.dump(model_to_save, model_path)
    print(f"\n💾 Meilleur modèle sauvé: {model_path}")
    
    # Sauvegarder résultats détaillés
    final_results = {
        'timestamp': timestamp,
        'baseline_accuracy': baseline_acc,
        'best_configuration': best_config,
        'best_accuracy': best_data['accuracy'],
        'improvement_pp': (best_data['accuracy'] - baseline_acc) * 100,
        'recommended_features': features_to_save,
        'all_results': results
    }
    
    results_path = f"evaluation/phase2_feature_engineering_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"📋 Résultats détaillés: {results_path}")
    
    return features_to_save, best_data['accuracy'], final_results

if __name__ == "__main__":
    recommended_features, accuracy, results = test_new_features()
    
    print()
    print("🚀 PHASE 2 TERMINÉE")
    print(f"🎯 Features recommandées: {len(recommended_features)}")
    print(f"📈 Performance finale: {accuracy:.1%}")
    print("🏁 OPTIMISATION MODÈLE COMPLÉTÉE!")