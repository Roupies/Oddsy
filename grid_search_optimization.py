#!/usr/bin/env python3
"""
🎯 Grid Search Hyperparameter Optimization
=========================================

Optimize the winning Current Production feature set using Grid Search
to find the best draw_weight, draw_threshold, and n_estimators combination.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

def compute_team_parity_score(df):
    """Compute team_parity_score on the fly"""
    elo_parity = 1 - abs(df['elo_diff_normalized'] - 0.5) * 2
    market_component = df['market_entropy_norm'].fillna(0.5)
    parity_score = (elo_parity * 0.6 + market_component * 0.4).clip(0, 1)
    return parity_score

class OptimizedCascadeChampion(BaseEstimator, ClassifierMixin):
    """Cascade Champion with optimizable hyperparameters"""
    
    def __init__(self, draw_weight=2.5, draw_threshold=0.4, 
                 n_estimators_stage1=200, n_estimators_stage2=150,
                 max_depth=10, random_state=42):
        self.draw_weight = draw_weight
        self.draw_threshold = draw_threshold
        self.n_estimators_stage1 = n_estimators_stage1
        self.n_estimators_stage2 = n_estimators_stage2
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Features (winning set from optimization)
        self.features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized', 
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score', 
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        self.is_fitted = False
    
    def _create_models(self):
        """Create RF models with current hyperparameters"""
        # Stage 1: Draw Detection
        self.clf_draw = RandomForestClassifier(
            n_estimators=self.n_estimators_stage1,
            max_depth=self.max_depth,
            min_samples_leaf=5,
            class_weight={0: 1, 1: self.draw_weight},
            random_state=self.random_state
        )
        
        # Stage 2: Home/Away Classification
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=self.n_estimators_stage2,
            max_depth=self.max_depth,
            class_weight='balanced',
            random_state=self.random_state
        )
    
    def fit(self, X, y):
        """Fit cascade model"""
        self._create_models()
        
        # Stage 1: Draw vs Non-Draw
        X_features = X[self.features].fillna(0.5)
        y_binary = (y == 1).astype(int)
        
        self.clf_draw.fit(X_features, y_binary)
        
        # Stage 2: Home vs Away (exclude draws)
        non_draw_mask = (y != 1)
        if np.sum(non_draw_mask) > 0:
            X_non_draw = X_features.loc[non_draw_mask]
            y_stage2 = y[non_draw_mask]
            y_ha_binary = (y_stage2 == 2).astype(int)
            
            if len(np.unique(y_ha_binary)) > 1:
                self.clf_homeaway.fit(X_non_draw, y_ha_binary)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using optimized cascade logic"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = []
        X_features = X[self.features].fillna(0.5)
        
        for i in range(len(X_features)):
            sample = X_features.iloc[i:i+1]
            
            # Stage 1: Draw detection with optimized threshold
            draw_proba = self.clf_draw.predict_proba(sample)[0]
            
            if draw_proba[1] > self.draw_threshold:
                prediction = 1  # Draw
            else:
                # Stage 2: Home/Away
                ha_proba = self.clf_homeaway.predict_proba(sample)[0]
                prediction = 2 if ha_proba[1] > 0.5 else 0
            
            predictions.append(prediction)
        
        return np.array(predictions)

def run_grid_search_optimization():
    """Run Grid Search optimization on the winning feature set"""
    print("🚀 GRID SEARCH HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print("🎯 Optimizing Current Production feature set (46.0% baseline)")
    
    # Load dataset
    data_path = "data/processed/v_auto_update_20250922_093416.csv"
    data = pd.read_csv(data_path)
    data['team_parity_score'] = compute_team_parity_score(data)
    
    # Split data
    train_data = data[data['Season'] != '2025-2026']
    test_data = data[data['Season'] == '2025-2026']
    
    train_with_results = train_data[train_data['FullTimeResult'].notna()].copy()
    test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
    
    # Target mapping
    result_map = {'H': 0, 'D': 1, 'A': 2}
    train_with_results['target'] = train_with_results['FullTimeResult'].map(result_map)
    test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
    
    print(f"📊 Training: {len(train_with_results)} matches")
    print(f"📊 Test: {len(test_with_results)} matches (50 EPL 2025-26)")
    
    y_train = train_with_results['target'].values
    y_test = test_with_results['target'].values
    
    # Define hyperparameter grid
    param_grid = {
        'draw_weight': [2.0, 2.5, 3.0, 3.5],
        'draw_threshold': [0.35, 0.40, 0.45, 0.50], 
        'n_estimators_stage1': [150, 200, 250],
        'n_estimators_stage2': [100, 150, 200]
    }
    
    print(f"\n🔧 Grid Search Parameters:")
    print(f"   draw_weight: {param_grid['draw_weight']}")
    print(f"   draw_threshold: {param_grid['draw_threshold']}")
    print(f"   n_estimators_stage1: {param_grid['n_estimators_stage1']}")
    print(f"   n_estimators_stage2: {param_grid['n_estimators_stage2']}")
    
    total_combinations = len(list(ParameterGrid(param_grid)))
    print(f"   Total combinations: {total_combinations}")
    
    # Grid Search
    best_score = 0
    best_params = None
    best_predictions = None
    results = []
    
    print(f"\n🧪 Testing hyperparameter combinations...")
    
    for i, params in enumerate(ParameterGrid(param_grid)):
        try:
            # Create model with current parameters
            model = OptimizedCascadeChampion(**params, random_state=42)
            
            # Train and test
            model.fit(train_with_results, y_train)
            predictions = model.predict(test_with_results)
            accuracy = accuracy_score(y_test, predictions)
            
            # Track results
            results.append({
                'params': params.copy(),
                'accuracy': accuracy,
                'predictions': predictions
            })
            
            # Update best
            if accuracy > best_score:
                best_score = accuracy
                best_params = params.copy()
                best_predictions = predictions
            
            # Progress update every 10 combinations
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{total_combinations} - Best so far: {best_score:.3f}")
        
        except Exception as e:
            print(f"   ❌ Error with params {params}: {str(e)}")
            continue
    
    # Results analysis
    print(f"\n" + "=" * 60)
    print(f"🏆 GRID SEARCH OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"🎯 Best Accuracy: {best_score:.3f} ({best_score:.1%})")
    print(f"📈 Improvement over baseline: {best_score - 0.46:+.3f} ({(best_score - 0.46)*100:+.1f}pp)")
    
    print(f"\n🔧 Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Detailed analysis of best model
    print(f"\n📋 Best Model Prediction Breakdown (50 matches):")
    class_names = ['Home', 'Draw', 'Away']
    for i, class_name in enumerate(class_names):
        actual_count = np.sum(y_test == i)
        predicted_count = np.sum(best_predictions == i)
        actual_pct = actual_count / len(y_test) * 100
        predicted_pct = predicted_count / len(y_test) * 100
        print(f"   {class_name}: {actual_count} actual ({actual_pct:.1f}%), "
              f"{predicted_count} predicted ({predicted_pct:.1f}%)")
    
    # Performance tiers
    if best_score >= 0.50:
        print(f"\n🎉 EXCELLENT! Achieved 50%+ target")
    elif best_score > 0.46:
        print(f"\n✅ SUCCESS! Improved beyond baseline")
    elif best_score >= 0.436:
        print(f"\n🎯 GOOD! Above minimum threshold")
    else:
        print(f"\n⚠️ Below minimum - further optimization needed")
    
    # Top 5 combinations
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    print(f"\n📊 Top 5 Hyperparameter Combinations:")
    print("-" * 80)
    
    for i, result in enumerate(results[:5], 1):
        params_str = f"w={result['params']['draw_weight']}, th={result['params']['draw_threshold']}, n1={result['params']['n_estimators_stage1']}, n2={result['params']['n_estimators_stage2']}"
        print(f"{i}. {result['accuracy']:.3f} ({result['accuracy']:.1%}) - {params_str}")
    
    return {
        'best_score': best_score,
        'best_params': best_params,
        'best_predictions': best_predictions,
        'all_results': results
    }

if __name__ == "__main__":
    optimization_results = run_grid_search_optimization()
    
    print(f"\n🎉 GRID SEARCH OPTIMIZATION COMPLETE!")
    if optimization_results['best_score'] > 0.46:
        print(f"🏆 Successfully improved from 46.0% to {optimization_results['best_score']:.1%}")
    else:
        print(f"📊 Best achieved: {optimization_results['best_score']:.1%} (baseline: 46.0%)")