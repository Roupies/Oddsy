#!/usr/bin/env python3
"""
🎯 Cascade Champion Feature Optimization - 50 EPL 2025-26 Matches
================================================================

MANDATORY: Test on all 50 EPL 2025-26 matches as required.
Use original dataset + compute team_parity_score on the fly.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def compute_team_parity_score(df):
    """Compute team_parity_score on the fly"""
    # Distance to 0.5 for elo_diff (closer = more parity)
    elo_parity = 1 - abs(df['elo_diff_normalized'] - 0.5) * 2
    
    # Market entropy component (higher = more uncertainty)
    market_component = df['market_entropy_norm'].fillna(0.5)
    
    # Combined parity score
    parity_score = (elo_parity * 0.6 + market_component * 0.4).clip(0, 1)
    
    return parity_score

class CascadeChampionOptimized(BaseEstimator, ClassifierMixin):
    """Optimized Cascade Champion with configurable features"""
    
    def __init__(self, stage1_features, stage2_features=None, draw_weight=2.5, 
                 draw_threshold=0.4, random_state=42):
        self.stage1_features = stage1_features
        self.stage2_features = stage2_features or stage1_features
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
    
    def fit(self, X, y):
        """Fit cascade model"""
        # Stage 1: Draw vs Non-Draw
        X_stage1 = X[self.stage1_features].fillna(0.5)
        y_binary = (y == 1).astype(int)
        
        self.clf_draw.fit(X_stage1, y_binary)
        
        # Stage 2: Home vs Away (exclude draws)
        non_draw_mask = (y != 1)
        if np.sum(non_draw_mask) > 0:
            X_stage2 = X[self.stage2_features].fillna(0.5).loc[non_draw_mask]
            y_stage2 = y[non_draw_mask]
            y_ha_binary = (y_stage2 == 2).astype(int)
            
            if len(np.unique(y_ha_binary)) > 1:
                self.clf_homeaway.fit(X_stage2, y_ha_binary)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using cascade logic"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = []
        
        for i in range(len(X)):
            sample = X.iloc[i:i+1]
            
            # Stage 1: Draw detection
            X_stage1 = sample[self.stage1_features].fillna(0.5)
            draw_proba = self.clf_draw.predict_proba(X_stage1)[0]
            
            if draw_proba[1] > self.draw_threshold:
                prediction = 1  # Draw
            else:
                # Stage 2: Home/Away
                X_stage2 = sample[self.stage2_features].fillna(0.5)
                ha_proba = self.clf_homeaway.predict_proba(X_stage2)[0]
                prediction = 2 if ha_proba[1] > 0.5 else 0
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def get_stage1_performance(self, X, y):
        """Get Stage 1 draw detection performance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_stage1 = X[self.stage1_features].fillna(0.5)
        y_binary = (y == 1).astype(int)
        
        stage1_predictions = self.clf_draw.predict(X_stage1)
        
        return {
            'predictions': stage1_predictions,
            'y_true': y_binary,
            'accuracy': accuracy_score(y_binary, stage1_predictions)
        }

def test_feature_set_50matches(set_name, stage1_features, stage2_features, X_train, y_train, X_test, y_test):
    """Test feature set on mandatory 50 EPL 2025-26 matches"""
    print(f"\n🧪 TESTING {set_name}")
    print("=" * 60)
    
    model = CascadeChampionOptimized(
        stage1_features=stage1_features,
        stage2_features=stage2_features,
        draw_weight=2.5,
        draw_threshold=0.4,
        random_state=42
    )
    
    print(f"📊 Stage 1 features ({len(stage1_features)}): {stage1_features}")
    if stage2_features != stage1_features:
        print(f"📊 Stage 2 features ({len(stage2_features)}): {stage2_features}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Test on 50 matches
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"\n🎯 Results on 50 EPL 2025-26 matches:")
    print(f"   Accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
    
    # Stage 1 analysis for specialized set
    if "Specialized" in set_name:
        print(f"\n🔬 Stage 1 (Draw Detection) Analysis:")
        stage1_results = model.get_stage1_performance(X_test, y_test)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            stage1_results['y_true'], stage1_results['predictions'], average=None
        )
        
        print(f"   Binary Classification (Draw vs Non-Draw):")
        print(f"     Non-Draw: P={precision[0]:.3f}, R={recall[0]:.3f}, F1={f1[0]:.3f}")
        print(f"     Draw:     P={precision[1]:.3f}, R={recall[1]:.3f}, F1={f1[1]:.3f}")
        print(f"   Stage 1 Accuracy: {stage1_results['accuracy']:.3f}")
    
    # Detailed breakdown
    print(f"\n📋 Prediction Breakdown (50 matches):")
    class_names = ['Home', 'Draw', 'Away']
    for i, class_name in enumerate(class_names):
        actual_count = np.sum(y_test == i)
        predicted_count = np.sum(test_predictions == i)
        actual_pct = actual_count / len(y_test) * 100
        predicted_pct = predicted_count / len(y_test) * 100
        print(f"   {class_name}: {actual_count} actual ({actual_pct:.1f}%), "
              f"{predicted_count} predicted ({predicted_pct:.1f}%)")
    
    # vs baseline comparison
    baseline_46 = 0.46
    improvement = test_accuracy - baseline_46
    if improvement > 0:
        print(f"   🎉 vs 46% baseline: +{improvement:.3f} ({improvement*100:+.1f}pp)")
    else:
        print(f"   📉 vs 46% baseline: {improvement:.3f} ({improvement*100:+.1f}pp)")
    
    return {
        'set_name': set_name,
        'accuracy': test_accuracy,
        'predictions': test_predictions,
        'improvement': improvement
    }

def run_50match_optimization():
    """Run feature optimization on mandatory 50 EPL 2025-26 matches"""
    print("🚀 CASCADE OPTIMIZATION - 50 EPL 2025-26 MATCHES")
    print("=" * 60)
    
    # Load ORIGINAL dataset (has 50 EPL 2025-26 matches)
    data_path = "data/processed/v_auto_update_20250922_093416.csv"
    data = pd.read_csv(data_path)
    
    print(f"📊 Dataset: {len(data)} matches loaded")
    
    # Add team_parity_score on the fly
    print("🔧 Computing team_parity_score...")
    data['team_parity_score'] = compute_team_parity_score(data)
    
    # Split data  
    train_data = data[data['Season'] != '2025-2026']
    test_data = data[data['Season'] == '2025-2026']
    
    # Filter matches with results
    train_with_results = train_data[train_data['FullTimeResult'].notna()].copy()
    test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
    
    # Target mapping
    result_map = {'H': 0, 'D': 1, 'A': 2}
    train_with_results['target'] = train_with_results['FullTimeResult'].map(result_map)
    test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
    
    print(f"📊 Training: {len(train_with_results)} matches")
    print(f"📊 Test (EPL 2025-26): {len(test_with_results)} matches ← MANDATORY 50!")
    
    # Verify we have 50 test matches
    if len(test_with_results) != 50:
        print(f"❌ ERROR: Expected 50 EPL 2025-26 matches, got {len(test_with_results)}")
        return None, []
    
    draw_count = test_with_results['target'].value_counts()[1]
    print(f"📊 EPL 2025-26 draws: {draw_count} ({draw_count/50:.1%})")
    
    y_train = train_with_results['target'].values
    y_test = test_with_results['target'].values
    
    # Define feature sets (adapted for available features)
    feature_sets = {
        'Set 1: Current Production': {
            'stage1': ['elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized', 
                      'corners_diff_normalized', 'form_diff_normalized', 'h2h_score', 
                      'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'],
            'stage2': None
        },
        
        'Set 2: Draw-Specialized': {
            'stage1': ['market_entropy_norm', 'team_parity_score'],
            'stage2': ['elo_diff_normalized', 'shots_diff_normalized', 'form_diff_normalized',
                      'h2h_score', 'home_xg_eff_10', 'away_xg_eff_10']
        },
        
        'Set 3: Minimal Power': {
            'stage1': ['elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 
                      'away_xg_eff_10', 'shots_diff_normalized'],
            'stage2': None
        },
        
        'Set 4: Enhanced Hybrid': {
            'stage1': ['elo_diff_normalized', 'market_entropy_norm', 'team_parity_score',
                      'shots_diff_normalized', 'form_diff_normalized', 'h2h_score',
                      'home_xg_eff_10', 'away_xg_eff_10'],
            'stage2': None
        },
        
        'Set 5: Validated Enhanced': {
            'stage1': ['market_entropy_norm', 'team_parity_score', 'elo_diff_normalized',
                      'shots_diff_normalized', 'form_diff_normalized', 'h2h_score',
                      'home_xg_eff_10', 'away_xg_eff_10', 'matchday_normalized'],
            'stage2': None
        }
    }
    
    # Test all feature sets on 50 matches
    results = []
    
    for set_name, config in feature_sets.items():
        stage2_features = config['stage2'] or config['stage1']
        
        try:
            result = test_feature_set_50matches(
                set_name,
                config['stage1'], 
                stage2_features,
                train_with_results,
                y_train,
                test_with_results, 
                y_test
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {set_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"🏆 50-MATCH OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<5} {'Set':<25} {'Accuracy':<12} {'vs 46% Baseline':<15}")
    print("-" * 65)
    
    for i, result in enumerate(results, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}"
        improvement = f"{result['improvement']:+.3f}" if result['improvement'] != 0 else " 0.000"
        print(f"{rank_emoji:<5} {result['set_name']:<25} {result['accuracy']:.3f} ({result['accuracy']:.1%})  {improvement:<15}")
    
    # Winner analysis
    if results:
        winner = results[0]
        print(f"\n🎉 WINNER ON 50 EPL 2025-26 MATCHES:")
        print(f"🏆 {winner['set_name']}")
        print(f"🎯 Accuracy: {winner['accuracy']:.1%}")
        print(f"📈 vs Baseline: {winner['improvement']:+.3f} ({winner['improvement']*100:+.1f}pp)")
        
        if winner['accuracy'] > 0.46:
            print(f"✅ SUCCESS: Above 46% baseline!")
        elif winner['accuracy'] > 0.436:
            print(f"🎯 GOOD: Above 43.6% minimum!")  
        else:
            print(f"⚠️ Needs hyperparameter optimization")
        
        return winner, results
    
    return None, results

if __name__ == "__main__":
    winner, all_results = run_50match_optimization()
    
    print(f"\n🎉 50-MATCH OPTIMIZATION COMPLETE!")
    if winner:
        print(f"🏆 Best: {winner['set_name']} ({winner['accuracy']:.1%})")
    else:
        print(f"⚠️ No successful tests completed")