#!/usr/bin/env python3
"""
🎯 Cascade Champion Feature Set Optimization
===========================================

Systematic testing of 5 different feature sets to optimize
the Cascade Champion v2.0 beyond its current 46% accuracy.

Based on correlation analysis and enhanced feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class CascadeChampionOptimized(BaseEstimator, ClassifierMixin):
    """
    Optimized Cascade Champion with configurable feature sets
    """
    
    def __init__(self, stage1_features, stage2_features, draw_weight=2.5, 
                 draw_threshold=0.4, random_state=42):
        self.stage1_features = stage1_features
        self.stage2_features = stage2_features
        self.draw_weight = draw_weight
        self.draw_threshold = draw_threshold
        self.random_state = random_state
        
        # Stage 1: Draw Detection
        self.clf_draw = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: draw_weight},
            random_state=random_state
        )
        
        # Stage 2: Home/Away Classification
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=random_state
        )
        
        self.is_fitted = False
        self.all_features = stage1_features + stage2_features
    
    def fit(self, X, y):
        """Fit cascade model with stage-specific features"""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.all_features)
        
        # Stage 1: Draw vs Non-Draw
        X_stage1 = X[self.stage1_features].fillna(0.5)
        y_binary = (y == 1).astype(int)  # 1 for Draw, 0 for Non-Draw
        
        self.clf_draw.fit(X_stage1, y_binary)
        
        # Stage 2: Home vs Away (exclude draws)
        non_draw_mask = (y != 1)
        if np.sum(non_draw_mask) > 0:
            X_stage2 = X[self.stage2_features].fillna(0.5).loc[non_draw_mask]
            y_stage2 = y[non_draw_mask]
            y_ha_binary = (y_stage2 == 2).astype(int)  # 1 for Away, 0 for Home
            
            if len(np.unique(y_ha_binary)) > 1:
                self.clf_homeaway.fit(X_stage2, y_ha_binary)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using cascade logic"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.all_features)
        
        predictions = []
        
        for i in range(len(X)):
            sample = X.iloc[i:i+1]
            
            # Stage 1: Draw detection
            X_stage1 = sample[self.stage1_features].fillna(0.5)
            draw_proba = self.clf_draw.predict_proba(X_stage1)[0]
            
            if draw_proba[1] > self.draw_threshold:
                prediction = 1  # Draw
            else:
                # Stage 2: Home/Away classification
                X_stage2 = sample[self.stage2_features].fillna(0.5)
                ha_proba = self.clf_homeaway.predict_proba(X_stage2)[0]
                prediction = 2 if ha_proba[1] > 0.5 else 0  # Away : Home
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def get_stage1_performance(self, X, y):
        """Get Stage 1 (draw detection) performance separately"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.all_features)
        
        X_stage1 = X[self.stage1_features].fillna(0.5)
        y_binary = (y == 1).astype(int)  # 1 for Draw, 0 for Non-Draw
        
        stage1_predictions = self.clf_draw.predict(X_stage1)
        stage1_probabilities = self.clf_draw.predict_proba(X_stage1)
        
        return {
            'predictions': stage1_predictions,
            'probabilities': stage1_probabilities,
            'accuracy': accuracy_score(y_binary, stage1_predictions),
            'y_true': y_binary
        }

def define_feature_sets():
    """Define the 5 feature sets to test"""
    
    feature_sets = {
        'Set 1: Current Production': {
            'stage1_features': ['elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized', 
                               'corners_diff_normalized', 'form_diff_normalized', 'h2h_score', 
                               'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'],
            'stage2_features': []  # Same features for unified model
        },
        
        'Set 2: Draw-Specialized': {
            'stage1_features': ['market_entropy_norm', 'team_parity_score'],
            'stage2_features': ['elo_diff_normalized', 'shots_diff_normalized', 'form_diff_normalized',
                               'h2h_score', 'home_xg_eff_10', 'away_xg_eff_10']
        },
        
        'Set 3: Minimal Power': {
            'stage1_features': ['elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 
                               'away_xg_eff_10', 'shots_diff_normalized'],
            'stage2_features': []
        },
        
        'Set 4: Enhanced Hybrid': {
            'stage1_features': ['elo_diff_normalized', 'market_entropy_norm', 'team_parity_score',
                               'shots_diff_normalized', 'form_diff_normalized', 'h2h_score',
                               'home_xg_eff_10', 'away_xg_eff_10'],
            'stage2_features': []
        },
        
        'Set 5: Validated Enhanced': {
            'stage1_features': ['market_entropy_norm', 'team_parity_score', 'elo_diff_normalized',
                               'shots_diff_normalized', 'form_diff_normalized', 'h2h_score',
                               'home_xg_eff_10', 'away_xg_eff_10', 'matchday_normalized'],
            'stage2_features': []
        }
    }
    
    # For unified models (Sets 1, 3, 4, 5), use stage1 features for both stages
    for set_name, config in feature_sets.items():
        if not config['stage2_features']:
            config['stage2_features'] = config['stage1_features'].copy()
    
    return feature_sets

def test_feature_set(set_name, stage1_features, stage2_features, X_train, y_train, X_test, y_test):
    """Test a specific feature set configuration"""
    print(f"\n🧪 TESTING {set_name}")
    print("=" * 60)
    
    # Create and train model
    model = CascadeChampionOptimized(
        stage1_features=stage1_features,
        stage2_features=stage2_features,
        draw_weight=2.5,
        draw_threshold=0.4,
        random_state=42
    )
    
    print(f"📊 Stage 1 features ({len(stage1_features)}): {stage1_features[:3]}...")
    print(f"📊 Stage 2 features ({len(stage2_features)}): {stage2_features[:3]}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Test predictions
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"\n🎯 Overall Results:")
    print(f"   Accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
    
    # Stage 1 analysis (draw detection)
    if set_name == "Set 2: Draw-Specialized":
        print(f"\n🔬 Stage 1 (Draw Detection) Analysis:")
        stage1_results = model.get_stage1_performance(X_test, y_test)
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            stage1_results['y_true'], stage1_results['predictions'], average=None
        )
        
        print(f"   Binary Draw Detection:")
        print(f"     Non-Draw (0): P={precision[0]:.3f}, R={recall[0]:.3f}, F1={f1[0]:.3f}")
        print(f"     Draw (1):     P={precision[1]:.3f}, R={recall[1]:.3f}, F1={f1[1]:.3f}")
        print(f"   Stage 1 Accuracy: {stage1_results['accuracy']:.3f}")
    
    # Prediction breakdown
    print(f"\n📋 Prediction Breakdown:")
    class_names = ['Home', 'Draw', 'Away']
    for i, class_name in enumerate(class_names):
        actual_count = np.sum(y_test == i)
        predicted_count = np.sum(test_predictions == i)
        if len(y_test) > 0:
            actual_pct = actual_count / len(y_test) * 100
            predicted_pct = predicted_count / len(y_test) * 100
            print(f"   {class_name}: {actual_count} actual ({actual_pct:.1f}%), "
                  f"{predicted_count} predicted ({predicted_pct:.1f}%)")
    
    # Performance vs baseline
    improvement = test_accuracy - 0.46  # Current baseline
    if improvement > 0:
        print(f"   🎉 Improvement: +{improvement:.3f} ({improvement*100:+.1f}pp)")
    else:
        print(f"   📉 Below baseline: {improvement:.3f} ({improvement*100:+.1f}pp)")
    
    return {
        'set_name': set_name,
        'accuracy': test_accuracy,
        'predictions': test_predictions,
        'stage1_features': stage1_features,
        'stage2_features': stage2_features,
        'improvement': improvement
    }

def run_feature_optimization():
    """Run comprehensive feature set optimization"""
    print("🚀 CASCADE CHAMPION FEATURE OPTIMIZATION")
    print("=" * 60)
    
    # Load enhanced dataset
    data_path = "data/processed/v16_specialized_features_enhanced.csv"
    data = pd.read_csv(data_path)
    
    print(f"📊 Dataset: {len(data)} matches loaded")
    
    # Split data (same methodology as Cascade Champion v2.0)
    train_data = data[data['Season'] != '2025-2026']
    test_data = data[data['Season'] == '2025-2026']
    
    # Filter for matches with results
    train_with_results = train_data[train_data['FullTimeResult'].notna()].copy()
    test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
    
    # Target mapping
    result_map = {'H': 0, 'D': 1, 'A': 2}
    train_with_results['target'] = train_with_results['FullTimeResult'].map(result_map)
    test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
    
    print(f"📊 Training: {len(train_with_results)} matches")
    print(f"📊 Test (EPL 2025-26): {len(test_with_results)} matches")
    print(f"📊 Draw distribution: {test_with_results['target'].value_counts()[1]} draws ({test_with_results['target'].value_counts()[1]/len(test_with_results):.1%})")
    
    # Prepare data
    y_train = train_with_results['target'].values
    y_test = test_with_results['target'].values
    
    # Get feature sets
    feature_sets = define_feature_sets()
    
    # Test all feature sets
    results = []
    
    for set_name, config in feature_sets.items():
        try:
            result = test_feature_set(
                set_name, 
                config['stage1_features'], 
                config['stage2_features'],
                train_with_results,  # Pass full DataFrame
                y_train,
                test_with_results,   # Pass full DataFrame  
                y_test
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {set_name}: {str(e)}")
            continue
    
    # Summary analysis
    print(f"\n" + "=" * 60)
    print(f"🏆 FEATURE SET COMPARISON SUMMARY")
    print("=" * 60)
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<5} {'Set':<25} {'Accuracy':<10} {'vs Baseline':<12}")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}"
        improvement = f"{result['improvement']:+.3f}" if result['improvement'] != 0 else " 0.000"
        print(f"{rank_emoji:<5} {result['set_name']:<25} {result['accuracy']:.3f} ({result['accuracy']:.1%})  {improvement:<12}")
    
    # Winner analysis
    if results:
        winner = results[0]
        print(f"\n🎉 WINNER: {winner['set_name']}")
        print(f"🎯 Best Accuracy: {winner['accuracy']:.1%}")
        print(f"📈 Improvement: {winner['improvement']:+.3f} ({winner['improvement']*100:+.1f}pp)")
        
        if winner['accuracy'] > 0.46:
            print(f"✅ SUCCESS: Above baseline!")
        else:
            print(f"⚠️ Below baseline - needs hyperparameter optimization")
        
        return winner, results
    
    return None, results

if __name__ == "__main__":
    winner, all_results = run_feature_optimization()
    
    print(f"\n🎉 FEATURE OPTIMIZATION COMPLETE!")
    if winner:
        print(f"🏆 Best performer: {winner['set_name']} ({winner['accuracy']:.1%})")
    else:
        print(f"⚠️ No successful tests completed")