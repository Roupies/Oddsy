#!/usr/bin/env python3
"""
🏆 Quick Model Comparison - Enhanced vs Augmented vs Ensemble
============================================================
Test the 3 improved models against the 33.3% Enhanced baseline
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import from our main script (avoiding main execution)
exec(open('generate_j6_natural_enhanced.py').read().replace('if __name__ == "__main__":', 'if False:'))

def quick_test_all_models():
    """Complete comparison of all models on 50 EPL 2025-26 matches"""
    print("🚀 COMPREHENSIVE MODEL TEST - 50 MATCHS EPL 2025-26")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    epl_2025 = data[data['Season'] == '2025-2026'].copy()
    epl_with_results = epl_2025[epl_2025['FullTimeResult'].notna()]
    real_odds_data = load_real_odds_data()
    
    # Test matches (ALL 50 EPL 2025-26 matches)
    test_matches = epl_with_results  # Use all available matches
    print(f"✅ Testing on {len(test_matches)} matches (FULL EPL 2025-26 dataset)")
    
    # Create models
    print("\n🔧 Creating models...")
    enhanced_cascade, enhanced_features = create_natural_enhanced_cascade()
    augmented_baseline, augmented_features = create_augmented_baseline_champion()
    
    # Quick train augmented baseline 
    historical_data = data[data['Season'] != '2025-2026'].copy()
    X_train_list = []
    y_train_list = []
    
    for idx, match in historical_data.head(1000).iterrows():  # Larger sample for better training
        if pd.isna(match['FullTimeResult']):
            continue
            
        try:
            features = calculate_enhanced_features(
                data, match['HomeTeam'], match['AwayTeam'], match['Date'], 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            feature_vector = []
            for feat_name in augmented_features:
                if feat_name in features:
                    feature_vector.append(features[feat_name])
                elif feat_name == 'market_entropy_norm' and 'market_entropy_historical' in features:
                    feature_vector.append(features['market_entropy_historical'])
                else:
                    feature_vector.append(0.5)
            
            X_train_list.append(feature_vector)
            result_map = {'H': 0, 'D': 1, 'A': 2}
            y_train_list.append(result_map[match['FullTimeResult']])
            
        except Exception as e:
            continue
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    augmented_baseline.fit(X_train, y_train)
    print("✅ Augmented baseline trained")
    
    # Create ensemble
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
    print("✅ Ensemble model created")
    
    # Test all models
    results = {
        'Enhanced': {'correct': 0, 'total': 0},
        'Augmented': {'correct': 0, 'total': 0}, 
        'Ensemble': {'correct': 0, 'total': 0}
    }
    
    print(f"\n🧪 Testing models...")
    
    for idx, match in test_matches.iterrows():
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
            
            # Ensemble prediction
            ensemble_result = ensemble.predict_with_logic(
                features, enhanced_features, augmented_features, baseline_features
            )
            ensemble_class = ['H', 'D', 'A'][ensemble_result['prediction']]
            ensemble_correct = (ensemble_class == actual)
            results['Ensemble']['correct'] += ensemble_correct
            results['Ensemble']['total'] += 1
            
            status_e = '✅' if enhanced_class == actual else '❌'
            status_a = '✅' if augmented_class == actual else '❌' 
            status_ens = '✅' if ensemble_class == actual else '❌'
            print(f"{home_team} vs {away_team}: E={enhanced_class}{status_e}, A={augmented_class}{status_a}, Ens={ensemble_class}{status_ens}, Actual={actual}")
            
        except Exception as e:
            print(f"⚠️ Error: {str(e)[:30]}")
    
    # Results  
    print(f"\n🎯 RÉSULTATS COMPLETS 50 MATCHS EPL 2025-26:")
    print("=" * 50)
    
    accuracies = {}
    for model_name in results:
        if results[model_name]['total'] > 0:
            accuracy = results[model_name]['correct'] / results[model_name]['total']
            accuracies[model_name] = accuracy
            print(f"{model_name:12}: {results[model_name]['correct']:2}/{results[model_name]['total']:2} = {accuracy:.1%}")
    
    # Best model
    if accuracies:
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]
        enhanced_acc = accuracies.get('Enhanced', 0)
        
        print(f"\n🏆 Meilleur Modèle: {best_model} ({best_accuracy:.1%})")
        print(f"📈 Amélioration vs Enhanced: {best_accuracy - enhanced_acc:+.1%}")
        
        # Distribution analysis
        print(f"\n📊 ANALYSE DISTRIBUTION EPL RÉELLE:")
        actual_dist = {'H': 0, 'D': 0, 'A': 0}
        for model_name in results:
            if results[model_name]['total'] > 0:
                # Count actual results
                break
        
        print(f"   H: ~43.6% EPL moyenne")  
        print(f"   D: ~23.0% EPL moyenne")
        print(f"   A: ~33.4% EPL moyenne")
        
        if best_accuracy > 0.45:  # Target improvement
            print("\n✅ OBJECTIF ATTEINT: >45% de précision!")
        elif best_accuracy > enhanced_acc:
            print("\n✅ AMÉLIORATION RÉUSSIE!")
        else:
            print("\n❌ Pas d'amélioration encore")
            
        # Comparison to baselines
        print(f"\n📈 COMPARAISON AUX BASELINES:")
        print(f"   Random (33.3%): {best_accuracy - 0.333:+.1%}")
        print(f"   Majority Home (43.6%): {best_accuracy - 0.436:+.1%}")
        print(f"   Production Target (50%): {best_accuracy - 0.50:+.1%}")
    
    return accuracies

if __name__ == "__main__":
    quick_test_all_models()