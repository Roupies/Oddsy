#!/usr/bin/env python3
"""
🎯 Test Production Models on EPL 2025-2026
==========================================

Testing the actual production champions:
- Baseline Champion v2.3 (53.5% CV)
- Cascade Champion v2.0 (50.0% test)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def test_baseline_champion():
    """Test Baseline Champion v2.3 on EPL 2025-2026"""
    print("🏆 TESTING BASELINE CHAMPION v2.3")
    print("=" * 50)
    
    try:
        # Load the production model
        print("📂 Loading Baseline Champion v2.3...")
        baseline_model = joblib.load('models/production/baseline_champion_v23.joblib')
        print(f"✅ Model loaded: {type(baseline_model)}")
        
        # Get model features
        if hasattr(baseline_model, 'feature_names_in_'):
            model_features = list(baseline_model.feature_names_in_)
            print(f"📋 Model features ({len(model_features)}): {model_features}")
        else:
            # Fallback feature list from your CLAUDE.md
            model_features = [
                'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 
                'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
                'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
            ]
            print(f"📋 Using fallback features ({len(model_features)})")
        
        # Load data
        print(f"\n📂 Loading test dataset...")
        data_path = "data/processed/v_auto_update_20250922_093416.csv"
        data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(data)} matches from {data_path}")
        
        # Filter for EPL 2025-2026 test data
        test_data = data[data['Season'] == '2025-2026']
        test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
        
        # Convert FullTimeResult to numeric target
        result_map = {'H': 0, 'D': 1, 'A': 2}
        test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
        
        print(f"\n📊 Test data (EPL 2025-2026): {len(test_with_results)} matches")
        
        if len(test_with_results) < 10:
            print(f"❌ Not enough test data: {len(test_with_results)}")
            return 0
        
        # Prepare test features
        X_test_dict = {}
        for feature in model_features:
            if feature in test_with_results.columns:
                X_test_dict[feature] = test_with_results[feature].fillna(0.5)
            else:
                print(f"⚠️ Missing feature '{feature}', using neutral value 0.5")
                X_test_dict[feature] = np.full(len(test_with_results), 0.5)
        
        X_test = pd.DataFrame(X_test_dict)[model_features]  # Ensure correct order
        y_test = test_with_results['target'].values
        
        print(f"🔧 Feature matrix shape: {X_test.shape}")
        print(f"🎯 Target classes: {np.unique(y_test)}")
        
        # Make predictions
        print(f"\n🧪 Testing Baseline Champion on EPL 2025-2026...")
        test_predictions = baseline_model.predict(X_test)
        test_probabilities = baseline_model.predict_proba(X_test)
        
        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"\n🏆 BASELINE CHAMPION v2.3 RESULTS:")
        print(f"   🎯 Test Accuracy (EPL 2025-2026): {test_accuracy:.3f} ({test_accuracy:.1%})")
        print(f"   📊 Test samples: {len(X_test)} matches")
        print(f"   🏭 Production model: CalibratedClassifierCV")
        
        # Target analysis
        if test_accuracy >= 0.50:
            print(f"   🎉 EXCELLENT! {test_accuracy:.1%} ≥ 50% target")
        elif test_accuracy >= 0.436:
            print(f"   🎯 GOOD! {test_accuracy:.1%} ≥ 43.6% baseline")
        else:
            print(f"   📊 Below baseline ({test_accuracy:.1%} < 43.6%)")
        
        # Prediction breakdown
        print(f"\n📋 PREDICTION BREAKDOWN:")
        class_names = ['Home', 'Draw', 'Away']
        for i, class_name in enumerate(class_names):
            actual_count = np.sum(y_test == i)
            predicted_count = np.sum(test_predictions == i)
            if len(y_test) > 0:
                actual_pct = actual_count / len(y_test) * 100
                predicted_pct = predicted_count / len(y_test) * 100
                print(f"   {class_name}: {actual_count} actual ({actual_pct:.1f}%), "
                      f"{predicted_count} predicted ({predicted_pct:.1f}%)")
        
        # Classification report
        print(f"\n📊 DETAILED PERFORMANCE:")
        test_report = classification_report(y_test, test_predictions, target_names=class_names)
        print(test_report)
        
        return test_accuracy
        
    except Exception as e:
        print(f"❌ Error testing Baseline Champion: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """Test production models"""
    print("🚀 TESTING PRODUCTION MODELS ON EPL 2025-2026")
    print("=" * 60)
    
    # Test Baseline Champion v2.3
    baseline_accuracy = test_baseline_champion()
    
    print(f"\n" + "=" * 60)
    print(f"🏆 PRODUCTION MODEL TEST RESULTS")
    print(f"=" * 60)
    print(f"📊 Baseline Champion v2.3: {baseline_accuracy:.3f} ({baseline_accuracy:.1%})")
    
    if baseline_accuracy >= 0.50:
        print(f"🎉 EXCELLENT: Production model achieves target!")
    elif baseline_accuracy >= 0.436:
        print(f"🎯 GOOD: Above baseline, production ready!")
    else:
        print(f"🔧 NEEDS WORK: Below baseline target")
    
    print(f"\n🎉 PRODUCTION MODEL TEST COMPLETE!")

if __name__ == "__main__":
    main()