#!/usr/bin/env python3
"""
🎯 Hyperparameter Optimization - Enhanced Cascade
================================================
Optimisation des hyperparamètres pour passer de 42% à 45%+
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, make_scorer
import joblib
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Import from our main script
exec(open('generate_j6_natural_enhanced.py').read().replace('if __name__ == "__main__":', 'if False:'))

def create_optimized_enhanced_cascade(hyperparams=None):
    """Create Enhanced Cascade with optimized hyperparameters"""
    if hyperparams is None:
        # Default optimized parameters
        hyperparams = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 8,
            'min_samples_leaf': 3,
            'max_features': 0.7,
            'class_weight': {0: 1.0, 1: 1.8, 2: 2.2}  # H, D, A weights
        }
    
    print(f"🔧 Creating Enhanced Cascade with optimized hyperparams:")
    for param, value in hyperparams.items():
        print(f"   {param}: {value}")
    
    # Enhanced features for cascade
    enhanced_features = [
        'market_entropy_historical', 'odds_spread_normalized', 
        'draw_margin_normalized', 'form_variance_diff', 'rivalry_factor'
    ]
    
    classical_features = [
        'elo_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'form_diff_normalized', 'h2h_score', 'matchday_normalized',
        'home_xg_eff_10', 'away_xg_eff_10'
    ]
    
    # Stage 1: Draw Detection (Enhanced features)
    stage1_model = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        min_samples_split=hyperparams['min_samples_split'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        max_features=hyperparams['max_features'],
        class_weight='balanced',  # For binary classification
        random_state=42,
        n_jobs=-1
    )
    
    # Stage 2: H/A Classification (Classical features)
    stage2_model = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'] - 2,  # Slightly simpler
        min_samples_split=hyperparams['min_samples_split'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        max_features=hyperparams['max_features'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    class OptimizedEnhancedCascade:
        def __init__(self):
            self.stage1 = stage1_model
            self.stage2 = stage2_model
            self.enhanced_features = enhanced_features
            self.classical_features = classical_features
            self.all_features = enhanced_features + classical_features
            self.class_weights = hyperparams['class_weight']
            
        def fit(self, X, y):
            # Prepare data
            X_array = np.array(X)
            
            # Stage 1: Draw vs Non-Draw
            y_binary = (y == 1).astype(int)  # 1 for Draw, 0 for Non-Draw
            X_stage1 = X_array[:, :len(self.enhanced_features)]
            self.stage1.fit(X_stage1, y_binary)
            
            # Stage 2: Home vs Away (exclude draws)
            non_draw_mask = (y != 1)
            X_stage2 = X_array[non_draw_mask][:, len(self.enhanced_features):]
            y_stage2 = y[non_draw_mask]
            y_stage2_binary = (y_stage2 == 2).astype(int)  # 1 for Away, 0 for Home
            
            if len(np.unique(y_stage2_binary)) > 1:
                self.stage2.fit(X_stage2, y_stage2_binary)
            
        def predict(self, X):
            X_array = np.array(X)
            predictions = []
            
            for i in range(len(X_array)):
                sample = X_array[i:i+1]
                
                # Stage 1: Draw Detection
                X_stage1 = sample[:, :len(self.enhanced_features)]
                draw_proba = self.stage1.predict_proba(X_stage1)[0]
                
                # Apply custom class weights for final decision
                if draw_proba[1] > 0.35:  # Draw threshold
                    prediction = 1  # Draw
                else:
                    # Stage 2: H/A Classification
                    X_stage2 = sample[:, len(self.enhanced_features):]
                    ha_proba = self.stage2.predict_proba(X_stage2)[0]
                    
                    # Apply class weights
                    if ha_proba[1] > 0.45:  # Away threshold (lower due to class weight)
                        prediction = 2  # Away
                    else:
                        prediction = 0  # Home
                
                predictions.append(prediction)
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            X_array = np.array(X)
            probabilities = []
            
            for i in range(len(X_array)):
                sample = X_array[i:i+1]
                
                # Stage 1: Draw Detection
                X_stage1 = sample[:, :len(self.enhanced_features)]
                draw_proba = self.stage1.predict_proba(X_stage1)[0]
                
                if draw_proba[1] > 0.35:
                    # High draw probability
                    proba = [0.25, draw_proba[1], 0.25]
                else:
                    # Stage 2: H/A Classification
                    X_stage2 = sample[:, len(self.enhanced_features):]
                    ha_proba = self.stage2.predict_proba(X_stage2)[0]
                    
                    # Distribute non-draw probability
                    non_draw_prob = 1 - draw_proba[1] * 0.5  # Reduce draw influence
                    home_prob = non_draw_prob * ha_proba[0] * self.class_weights[0]
                    away_prob = non_draw_prob * ha_proba[1] * self.class_weights[2]
                    
                    # Normalize
                    total = home_prob + draw_proba[1] + away_prob
                    proba = [home_prob/total, draw_proba[1]/total, away_prob/total]
                
                probabilities.append(proba)
            
            return np.array(probabilities)
    
    return OptimizedEnhancedCascade(), enhanced_features + classical_features

def grid_search_hyperparameters():
    """Grid search pour optimiser les hyperparamètres"""
    print("🔍 GRID SEARCH HYPERPARAMETERS")
    print("=" * 50)
    
    # Load data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    real_odds_data = load_real_odds_data()
    
    # Prepare training data (historical)
    historical_data = data[data['Season'] != '2025-2026'].copy()
    
    X_train = []
    y_train = []
    
    print("📊 Préparation des données d'entraînement...")
    
    for idx, match in historical_data.head(1000).iterrows():  # Échantillon pour vitesse
        if pd.isna(match['FullTimeResult']):
            continue
            
        try:
            features = calculate_enhanced_features(
                data, match['HomeTeam'], match['AwayTeam'], match['Date'], 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            # Feature vector for all features
            feature_vector = []
            all_features = [
                'market_entropy_historical', 'odds_spread_normalized', 
                'draw_margin_normalized', 'form_variance_diff', 'rivalry_factor',
                'elo_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
                'form_diff_normalized', 'h2h_score', 'matchday_normalized',
                'home_xg_eff_10', 'away_xg_eff_10'
            ]
            
            for feat_name in all_features:
                if feat_name in features:
                    feature_vector.append(features[feat_name])
                else:
                    feature_vector.append(0.5)  # Default
            
            X_train.append(feature_vector)
            result_map = {'H': 0, 'D': 1, 'A': 2}
            y_train.append(result_map[match['FullTimeResult']])
            
        except Exception as e:
            continue
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"✅ Données d'entraînement: {len(X_train)} échantillons")
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [300, 500, 800],
        'max_depth': [12, 15, 18],
        'min_samples_split': [6, 8, 12],
        'min_samples_leaf': [2, 3, 4],
        'max_features': [0.6, 0.7, 0.8],
        'class_weight_away': [1.8, 2.0, 2.2, 2.5]  # Weight for Away class
    }
    
    best_score = 0
    best_params = None
    
    print(f"🔍 Testing {len(list(product(*param_grid.values())))} combinations...")
    
    # Manual grid search (more control)
    test_count = 0
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    for max_feat in param_grid['max_features']:
                        for away_weight in param_grid['class_weight_away']:
                            
                            test_count += 1
                            if test_count > 20:  # Limite pour vitesse
                                break
                            
                            # Create hyperparams
                            hyperparams = {
                                'n_estimators': n_est,
                                'max_depth': max_d,
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf,
                                'max_features': max_feat,
                                'class_weight': {0: 1.0, 1: 1.5, 2: away_weight}
                            }
                            
                            try:
                                # Create and test model
                                model, features = create_optimized_enhanced_cascade(hyperparams)
                                model.fit(X_train, y_train)
                                
                                # Quick validation on training data
                                predictions = model.predict(X_train)
                                score = accuracy_score(y_train, predictions)
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = hyperparams.copy()
                                    print(f"🎯 Nouveau meilleur: {score:.3f} - {hyperparams}")
                                
                            except Exception as e:
                                continue
    
    print(f"\n🏆 MEILLEURS HYPERPARAMÈTRES (Score: {best_score:.3f}):")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    return best_params

def test_optimized_model_on_epl2025():
    """Test du modèle optimisé sur EPL 2025-26"""
    print("\n🧪 TEST MODÈLE OPTIMISÉ SUR EPL 2025-26")
    print("=" * 50)
    
    # Get best hyperparameters
    best_params = grid_search_hyperparameters()
    
    # Load data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    epl_2025 = data[data['Season'] == '2025-2026'].copy()
    epl_with_results = epl_2025[epl_2025['FullTimeResult'].notna()]
    real_odds_data = load_real_odds_data()
    
    # Create optimized model
    optimized_model, opt_features = create_optimized_enhanced_cascade(best_params)
    
    # Train on historical data
    historical_data = data[data['Season'] != '2025-2026'].copy()
    X_train = []
    y_train = []
    
    print("🔄 Entraînement modèle optimisé...")
    
    for idx, match in historical_data.iterrows():
        if pd.isna(match['FullTimeResult']):
            continue
            
        try:
            features = calculate_enhanced_features(
                data, match['HomeTeam'], match['AwayTeam'], match['Date'], 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            feature_vector = []
            for feat_name in opt_features:
                if feat_name in features:
                    feature_vector.append(features[feat_name])
                else:
                    feature_vector.append(0.5)
            
            X_train.append(feature_vector)
            result_map = {'H': 0, 'D': 1, 'A': 2}
            y_train.append(result_map[match['FullTimeResult']])
            
        except Exception as e:
            continue
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    optimized_model.fit(X_train, y_train)
    
    print(f"✅ Modèle optimisé entraîné sur {len(X_train)} matchs")
    
    # Test on EPL 2025-26
    print(f"🧪 Test sur {len(epl_with_results)} matchs EPL 2025-26...")
    
    correct = 0
    total = 0
    class_results = {'H': {'correct': 0, 'total': 0}, 'D': {'correct': 0, 'total': 0}, 'A': {'correct': 0, 'total': 0}}
    
    for idx, match in epl_with_results.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual = match['FullTimeResult']
        match_date = match['Date']
        
        try:
            features = calculate_enhanced_features(
                data, home_team, away_team, match_date, 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            feature_vector = []
            for feat_name in opt_features:
                if feat_name in features:
                    feature_vector.append(features[feat_name])
                else:
                    feature_vector.append(0.5)
            
            X_test = np.array([feature_vector])
            pred = optimized_model.predict(X_test)[0]
            pred_class = ['H', 'D', 'A'][pred]
            
            is_correct = (pred_class == actual)
            correct += is_correct
            total += 1
            
            class_results[actual]['total'] += 1
            if is_correct:
                class_results[actual]['correct'] += 1
            
            if total % 10 == 0:
                print(f"   Traité {total} matchs...")
                
        except Exception as e:
            continue
    
    # Results
    if total > 0:
        accuracy = correct / total
        print(f"\n🎯 RÉSULTATS MODÈLE OPTIMISÉ:")
        print(f"Précision globale: {correct}/{total} = {accuracy:.1%}")
        
        print(f"\n📊 Par classe:")
        for class_name in ['H', 'D', 'A']:
            if class_results[class_name]['total'] > 0:
                class_acc = class_results[class_name]['correct'] / class_results[class_name]['total']
                print(f"  {class_name}: {class_results[class_name]['correct']}/{class_results[class_name]['total']} = {class_acc:.1%}")
        
        print(f"\n📈 Comparaison:")
        print(f"   Modèle original: 42.0%")
        print(f"   Modèle optimisé: {accuracy:.1%}")
        print(f"   Amélioration: {accuracy - 0.42:+.1%}")
        
        if accuracy >= 0.45:
            print("\n🎉 OBJECTIF 45% ATTEINT!")
        elif accuracy > 0.42:
            print("\n🎯 AMÉLIORATION RÉUSSIE!")
        
        return accuracy, best_params
    
    return 0, best_params

if __name__ == "__main__":
    test_optimized_model_on_epl2025()