#!/usr/bin/env python3
"""
🎯 Test All Production Models on EPL 2025-2026
==============================================

Testing all production models to find the best performer:
- Baseline Champion v2.3 (from joblib)
- Cascade Champion v2.0 (from scripts)
- Latest optimized models
"""

import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('scripts/final')

def test_baseline_champion_v23():
    """Test the saved Baseline Champion v2.3"""
    print("🏆 TESTING BASELINE CHAMPION v2.3")
    print("=" * 50)
    
    try:
        # Load the production model
        baseline_model = joblib.load('models/production/baseline_champion_v23.joblib')
        
        # Load latest data
        data = pd.read_csv("data/processed/v_auto_update_20250922_093416.csv")
        
        # Filter EPL 2025-2026 test data
        test_data = data[data['Season'] == '2025-2026']
        test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
        
        # Target mapping
        result_map = {'H': 0, 'D': 1, 'A': 2}
        test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
        
        print(f"📊 Test samples: {len(test_with_results)}")
        
        # Model features
        model_features = ['form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 
                         'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
                         'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10']
        
        # Prepare features
        X_test_dict = {}
        for feature in model_features:
            if feature in test_with_results.columns:
                X_test_dict[feature] = test_with_results[feature].fillna(0.5)
            else:
                X_test_dict[feature] = np.full(len(test_with_results), 0.5)
        
        X_test = pd.DataFrame(X_test_dict)[model_features]
        y_test = test_with_results['target'].values
        
        # Predict
        predictions = baseline_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"🎯 Accuracy: {accuracy:.3f} ({accuracy:.1%})")
        
        return accuracy, predictions, y_test, "Baseline Champion v2.3"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0, None, None, "Baseline Champion v2.3 (Failed)"

def test_optimized_baseline():
    """Test a fresh optimized baseline model"""
    print("\n🏆 TESTING OPTIMIZED BASELINE")
    print("=" * 50)
    
    try:
        # Load data
        data = pd.read_csv("data/processed/v_auto_update_20250922_093416.csv")
        
        # Split data
        train_data = data[data['Season'] != '2025-2026']
        test_data = data[data['Season'] == '2025-2026']
        
        # Filter for matches with results
        train_with_results = train_data[train_data['FullTimeResult'].notna()].copy()
        test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
        
        # Target mapping
        result_map = {'H': 0, 'D': 1, 'A': 2}
        train_with_results['target'] = train_with_results['FullTimeResult'].map(result_map)
        test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
        
        print(f"📊 Train samples: {len(train_with_results)}")
        print(f"📊 Test samples: {len(test_with_results)}")
        
        # Use the same features as production model
        features = ['form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 
                   'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
                   'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10']
        
        # Prepare data
        X_train = train_with_results[features].fillna(0.5)
        y_train = train_with_results['target']
        X_test = test_with_results[features].fillna(0.5)
        y_test = test_with_results['target']
        
        # Create optimized model (similar to your production model)
        base_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Use calibration like the production model
        model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"🎯 Accuracy: {accuracy:.3f} ({accuracy:.1%})")
        
        return accuracy, predictions, y_test.values, "Optimized Baseline"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0, None, None, "Optimized Baseline (Failed)"

def compare_models_detailed(results):
    """Compare all model results in detail"""
    print("\n" + "=" * 60)
    print("🏆 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    
    best_accuracy = 0
    best_model = None
    
    for accuracy, predictions, y_test, model_name in results:
        if accuracy > 0:
            print(f"\n📊 {model_name}:")
            print(f"   Accuracy: {accuracy:.3f} ({accuracy:.1%})")
            
            if accuracy >= 0.50:
                print(f"   🎉 EXCELLENT! ≥ 50% target")
            elif accuracy >= 0.436:
                print(f"   🎯 GOOD! ≥ 43.6% baseline")
            else:
                print(f"   📊 Below baseline")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
            
            # Prediction breakdown
            if predictions is not None and y_test is not None:
                class_names = ['Home', 'Draw', 'Away']
                print(f"   Prediction breakdown:")
                for i, class_name in enumerate(class_names):
                    actual_count = np.sum(y_test == i)
                    predicted_count = np.sum(predictions == i)
                    if len(y_test) > 0:
                        actual_pct = actual_count / len(y_test) * 100
                        predicted_pct = predicted_count / len(y_test) * 100
                        print(f"     {class_name}: {actual_count} actual ({actual_pct:.1f}%), "
                              f"{predicted_count} predicted ({predicted_pct:.1f}%)")
        else:
            print(f"\n❌ {model_name}: FAILED")
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"🎯 BEST ACCURACY: {best_accuracy:.1%}")
    
    if best_accuracy >= 0.50:
        print(f"🎉 PRODUCTION READY! Exceeds 50% target")
    elif best_accuracy >= 0.436:
        print(f"🎯 GOOD PERFORMANCE! Above baseline")
    else:
        print(f"🔧 NEEDS IMPROVEMENT! Below baseline")

def main():
    """Test all production models"""
    print("🚀 COMPREHENSIVE PRODUCTION MODEL TEST")
    print("📊 Testing on EPL 2025-2026 season")
    print("=" * 60)
    
    results = []
    
    # Test Baseline Champion v2.3
    result1 = test_baseline_champion_v23()
    results.append(result1)
    
    # Test optimized baseline
    result2 = test_optimized_baseline()
    results.append(result2)
    
    # Compare all results
    compare_models_detailed(results)
    
    print(f"\n🎉 ALL TESTS COMPLETE!")

if __name__ == "__main__":
    main()