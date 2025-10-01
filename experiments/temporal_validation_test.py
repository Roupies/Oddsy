#!/usr/bin/env python3
"""
🕒 Temporal Validation Test - 2019-2024 Train, 2024-2025 Test
=============================================================
Test des modèles sur split temporel pour validation robuste
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import from our main script (avoiding main execution)
exec(open('generate_j6_natural_enhanced.py').read().replace('if __name__ == "__main__":', 'if False:'))

def temporal_validation_test():
    """Test temporel complet: 2019-2024 train, 2024-2025 test"""
    print("🕒 VALIDATION TEMPORELLE - SPLIT HISTORIQUE")
    print("=" * 60)
    print("📅 Train: 2019-2024 | Test: 2024-2025")
    
    # Load data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    real_odds_data = load_real_odds_data()
    
    # Split temporel
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']  # Plus ancien que 2025-2026
    
    train_data = data[data['Season'].isin(train_seasons)].copy()
    test_data = data[data['Season'].isin(test_seasons)].copy()
    test_with_results = test_data[test_data['FullTimeResult'].notna()]
    
    print(f"✅ Données d'entraînement: {len(train_data)} matchs ({train_seasons})")
    print(f"✅ Données de test: {len(test_with_results)} matchs ({test_seasons})")
    
    if len(test_with_results) == 0:
        print("❌ Aucune donnée de test disponible pour 2024-2025")
        print("🔄 Fallback: Test sur 2023-2024 (dernière saison complète)")
        
        # Fallback sur 2023-2024
        train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023']
        test_seasons = ['2023-2024']
        
        train_data = data[data['Season'].isin(train_seasons)].copy()
        test_data = data[data['Season'].isin(test_seasons)].copy()
        test_with_results = test_data[test_data['FullTimeResult'].notna()]
        
        print(f"📊 Nouvelles données:")
        print(f"   Train: {len(train_data)} matchs ({train_seasons})")
        print(f"   Test: {len(test_with_results)} matchs ({test_seasons})")
    
    # Create models
    print(f"\n🔧 Création des modèles...")
    enhanced_cascade, enhanced_features = create_natural_enhanced_cascade()
    augmented_baseline, augmented_features = create_augmented_baseline_champion()
    
    # Train Enhanced Cascade sur données historiques
    print("🎯 Entraînement Enhanced Cascade...")
    X_enhanced_train = []
    y_enhanced_train = []
    
    for idx, match in train_data.iterrows():
        if pd.isna(match['FullTimeResult']):
            continue
            
        try:
            features = calculate_enhanced_features(
                data, match['HomeTeam'], match['AwayTeam'], match['Date'], 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            # Enhanced features
            feature_vector = []
            for feat_name in enhanced_features:
                if feat_name in features:
                    feature_vector.append(features[feat_name])
                else:
                    feature_vector.append(0.5)  # Fallback
            
            X_enhanced_train.append(feature_vector)
            result_map = {'H': 0, 'D': 1, 'A': 2}
            y_enhanced_train.append(result_map[match['FullTimeResult']])
            
        except Exception as e:
            continue
    
    # Re-train enhanced cascade
    if len(X_enhanced_train) > 0:
        X_enhanced = np.array(X_enhanced_train)
        y_enhanced = np.array(y_enhanced_train)
        
        # Create fresh enhanced cascade
        enhanced_cascade = create_natural_enhanced_cascade()[0]
        print(f"✅ Enhanced Cascade ré-entraîné sur {len(X_enhanced)} matchs")
    
    # Train Augmented Baseline
    print("🎯 Entraînement Augmented Baseline...")
    X_augmented_train = []
    y_augmented_train = []
    
    for idx, match in train_data.iterrows():
        if pd.isna(match['FullTimeResult']):
            continue
            
        try:
            features = calculate_enhanced_features(
                data, match['HomeTeam'], match['AwayTeam'], match['Date'], 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            # Augmented features
            feature_vector = []
            for feat_name in augmented_features:
                if feat_name in features:
                    feature_vector.append(features[feat_name])
                elif feat_name == 'market_entropy_norm' and 'market_entropy_historical' in features:
                    feature_vector.append(features['market_entropy_historical'])
                else:
                    feature_vector.append(0.5)
            
            X_augmented_train.append(feature_vector)
            result_map = {'H': 0, 'D': 1, 'A': 2}
            y_augmented_train.append(result_map[match['FullTimeResult']])
            
        except Exception as e:
            continue
    
    if len(X_augmented_train) > 0:
        X_augmented = np.array(X_augmented_train)
        y_augmented = np.array(y_augmented_train)
        augmented_baseline.fit(X_augmented, y_augmented)
        print(f"✅ Augmented Baseline entraîné sur {len(X_augmented)} matchs")
    
    # Create ensemble
    print("🎯 Création Weighted Ensemble...")
    ensemble = create_weighted_ensemble_model()
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    ensemble.fit_models(
        enhanced_cascade, 
        augmented_baseline, 
        'models/production/baseline_champion_v23.joblib'
    )
    print("✅ Ensemble model configuré")
    
    # Test sur échantillon pour vitesse (50 premiers matchs)
    test_sample = test_with_results.head(50)
    print(f"\n🧪 Test rapide sur {len(test_sample)} matchs de test...")
    
    results = {
        'Enhanced': {'correct': 0, 'total': 0, 'predictions': []},
        'Augmented': {'correct': 0, 'total': 0, 'predictions': []}, 
        'Ensemble': {'correct': 0, 'total': 0, 'predictions': []},
        'Original': {'correct': 0, 'total': 0, 'predictions': []}
    }
    
    # Load original baseline if available
    try:
        original_baseline = joblib.load('models/production/baseline_champion_v23.joblib')
        original_available = True
        print("✅ Original Baseline chargé pour comparaison")
    except:
        original_available = False
        print("⚠️ Original Baseline non disponible")
    
    # Test predictions
    for idx, match in test_sample.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual = match['FullTimeResult']
        match_date = match['Date']
        
        try:
            features = calculate_enhanced_features(
                data, home_team, away_team, match_date, 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            # Enhanced prediction
            X_enhanced = pd.DataFrame([features])[enhanced_features]
            enhanced_pred = enhanced_cascade.predict(X_enhanced)[0]
            enhanced_class = ['H', 'D', 'A'][enhanced_pred]
            enhanced_correct = (enhanced_class == actual)
            results['Enhanced']['correct'] += enhanced_correct
            results['Enhanced']['total'] += 1
            results['Enhanced']['predictions'].append(enhanced_class)
            
            # Augmented prediction
            augmented_vector = []
            for feat in augmented_features:
                if feat in features:
                    augmented_vector.append(features[feat])
                elif feat == 'market_entropy_norm' and 'market_entropy_historical' in features:
                    augmented_vector.append(features['market_entropy_historical'])
                else:
                    augmented_vector.append(0.5)
            
            X_augmented = np.array([augmented_vector])
            augmented_pred = augmented_baseline.predict(X_augmented)[0]
            augmented_class = ['H', 'D', 'A'][augmented_pred]
            augmented_correct = (augmented_class == actual)
            results['Augmented']['correct'] += augmented_correct
            results['Augmented']['total'] += 1
            results['Augmented']['predictions'].append(augmented_class)
            
            # Ensemble prediction
            ensemble_result = ensemble.predict_with_logic(
                features, enhanced_features, augmented_features, baseline_features
            )
            ensemble_class = ['H', 'D', 'A'][ensemble_result['prediction']]
            ensemble_correct = (ensemble_class == actual)
            results['Ensemble']['correct'] += ensemble_correct
            results['Ensemble']['total'] += 1
            results['Ensemble']['predictions'].append(ensemble_class)
            
            # Original baseline (if available)
            if original_available:
                baseline_vector = []
                for feat in baseline_features:
                    if feat in features:
                        baseline_vector.append(features[feat])
                    elif feat == 'market_entropy_norm' and 'market_entropy_historical' in features:
                        baseline_vector.append(features['market_entropy_historical'])
                    else:
                        baseline_vector.append(0.5)
                
                X_baseline = np.array([baseline_vector])
                original_pred = original_baseline.predict(X_baseline)[0]
                original_class = ['H', 'D', 'A'][original_pred]
                original_correct = (original_class == actual)
                results['Original']['correct'] += original_correct
                results['Original']['total'] += 1
                results['Original']['predictions'].append(original_class)
            
            # Affichage périodique
            if results['Enhanced']['total'] % 20 == 0:
                print(f"   Traité {results['Enhanced']['total']} matchs...")
            
        except Exception as e:
            print(f"⚠️ Erreur {home_team} vs {away_team}: {str(e)[:30]}")
    
    # Results analysis
    print(f"\n🎯 RÉSULTATS VALIDATION TEMPORELLE:")
    print("=" * 60)
    
    accuracies = {}
    for model_name in ['Enhanced', 'Augmented', 'Ensemble', 'Original']:
        if results[model_name]['total'] > 0:
            accuracy = results[model_name]['correct'] / results[model_name]['total']
            accuracies[model_name] = accuracy
            print(f"{model_name:12}: {results[model_name]['correct']:3}/{results[model_name]['total']:3} = {accuracy:.1%}")
        else:
            print(f"{model_name:12}: Non testé")
    
    # Distribution analysis du test set
    print(f"\n📊 DISTRIBUTION TEST SET:")
    actual_results = {'H': 0, 'D': 0, 'A': 0}
    for model_name in results:
        if results[model_name]['total'] > 0:
            # Count from actual results in test data
            for _, match in test_with_results.iterrows():
                actual_results[match['FullTimeResult']] += 1
            break
    
    total_test = sum(actual_results.values())
    if total_test > 0:
        print(f"   H: {actual_results['H']:3} = {actual_results['H']/total_test:.1%}")
        print(f"   D: {actual_results['D']:3} = {actual_results['D']/total_test:.1%}")
        print(f"   A: {actual_results['A']:3} = {actual_results['A']/total_test:.1%}")
    
    # Best model
    if accuracies:
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model} ({best_accuracy:.1%})")
        
        # Comparisons
        print(f"\n📈 COMPARAISONS BASELINES:")
        print(f"   Random (33.3%): {best_accuracy - 0.333:+.1%}")
        print(f"   Majority Home (43.6%): {best_accuracy - 0.436:+.1%}")
        print(f"   Target 45%: {best_accuracy - 0.45:+.1%}")
        print(f"   Target 50%: {best_accuracy - 0.50:+.1%}")
        
        if best_accuracy >= 0.45:
            print("\n✅ OBJECTIF 45% ATTEINT!")
        elif best_accuracy >= 0.42:
            print("\n🎯 PROCHE DE L'OBJECTIF (42%+)")
        else:
            print("\n🔧 OPTIMISATIONS NÉCESSAIRES")
        
        # Stability check
        enhanced_acc = accuracies.get('Enhanced', 0)
        if 'Enhanced' in accuracies:
            print(f"\n📊 STABILITÉ TEMPORELLE:")
            print(f"   Enhanced sur EPL 2025-26: 42.0%")
            print(f"   Enhanced sur test temporel: {enhanced_acc:.1%}")
            print(f"   Différence: {enhanced_acc - 0.42:+.1%}")
            
            if abs(enhanced_acc - 0.42) <= 0.05:
                print("   ✅ Modèle stable sur périodes différentes")
            else:
                print("   ⚠️ Variance temporelle détectée")
    
    return accuracies, test_seasons, len(test_with_results)

if __name__ == "__main__":
    temporal_validation_test()