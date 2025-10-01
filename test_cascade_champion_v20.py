#!/usr/bin/env python3
"""
🏆 Test Cascade Champion v2.0 - The Actual 50% Model
====================================================

Reconstructing the exact Cascade Champion v2.0 that achieved 50% on EPL 2025-26
Based on the production metadata file.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

class CascadeChampionV20(BaseEstimator, ClassifierMixin):
    """
    Exact reconstruction of Cascade Champion v2.0 from metadata.
    
    Architecture from metadata:
    - Stage 1: Draw Detection (RandomForest with draw_weight=2.5)
    - Stage 2: Home/Away Classification (RandomForest balanced)
    - Draw threshold: 0.4
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Stage 1: Draw Detection (from metadata)
        self.clf_draw = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: 2.5},  # non_draw: 1, draw: 2.5
            random_state=random_state
        )
        
        # Stage 2: Home/Away Classification (from metadata)
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=random_state
        )
        
        # Cascade parameters from metadata
        self.draw_threshold = 0.4
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the cascade model exactly as in production."""
        X = np.array(X)
        y = np.array(y)
        
        print(f"🔧 Training Cascade Champion v2.0 on {len(X)} samples")
        
        # Stage 1: Draw vs Non-Draw
        y_binary = (y == 1).astype(int)  # 1 for Draw, 0 for Non-Draw
        self.clf_draw.fit(X, y_binary)
        
        draw_distribution = np.mean(y_binary) * 100
        print(f"   Draw distribution: {draw_distribution:.1f}%")
        
        # Stage 2: Home vs Away (exclude draws)
        non_draw_mask = (y != 1)
        if np.sum(non_draw_mask) > 0:
            X_non_draw = X[non_draw_mask]
            y_non_draw = y[non_draw_mask]
            y_ha_binary = (y_non_draw == 2).astype(int)  # 1 for Away, 0 for Home
            
            self.clf_homeaway.fit(X_non_draw, y_ha_binary)
            print(f"   Home/Away trained on {len(X_non_draw)} non-draw samples")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using exact cascade logic from metadata."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = np.array(X)
        predictions = []
        
        for i in range(len(X)):
            sample = X[i:i+1]
            
            # Stage 1: Draw detection
            draw_proba = self.clf_draw.predict_proba(sample)[0]
            
            # Apply draw threshold (0.4 from metadata)
            if draw_proba[1] > self.draw_threshold:
                prediction = 1  # Draw
            else:
                # Stage 2: Home/Away classification
                ha_proba = self.clf_homeaway.predict_proba(sample)[0]
                prediction = 2 if ha_proba[1] > 0.5 else 0  # Away : Home
            
            predictions.append(prediction)
        
        return np.array(predictions)

def test_cascade_champion_v20():
    """Test the reconstructed Cascade Champion v2.0"""
    print("🏆 TESTING RECONSTRUCTED CASCADE CHAMPION v2.0")
    print("=" * 60)
    print("📋 Based on production metadata with 50% EPL 2025-26 accuracy")
    print()
    
    try:
        # Load data (same as metadata)
        data_path = "data/processed/v_auto_update_20250922_093416.csv"
        data = pd.read_csv(data_path)
        
        # Features from metadata (exact order)
        features = [
            "elo_diff_normalized",
            "market_entropy_norm", 
            "shots_diff_normalized",
            "corners_diff_normalized",
            "form_diff_normalized",
            "h2h_score",
            "matchday_normalized",
            "home_xg_eff_10",
            "away_xg_eff_10",
            "away_goals_sum_5"
        ]
        
        print(f"📊 Features (from metadata): {len(features)} features")
        print(f"   {features}")
        
        # Split data temporally (same as metadata)
        train_data = data[data['Season'] != '2025-2026']
        test_data = data[data['Season'] == '2025-2026']
        
        # Filter for matches with results
        train_with_results = train_data[train_data['FullTimeResult'].notna()].copy()
        test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
        
        # Target mapping
        result_map = {'H': 0, 'D': 1, 'A': 2}
        train_with_results['target'] = train_with_results['FullTimeResult'].map(result_map)
        test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
        
        print(f"\n📊 Data split (matching metadata):")
        print(f"   Training: {len(train_with_results)} matches")
        print(f"   Test EPL 2025-26: {len(test_with_results)} matches")
        
        # Prepare features (fill missing with 0 as in metadata)
        X_train_dict = {}
        X_test_dict = {}
        
        for feature in features:
            if feature in train_with_results.columns:
                X_train_dict[feature] = train_with_results[feature].fillna(0)
                X_test_dict[feature] = test_with_results[feature].fillna(0)
            else:
                print(f"⚠️ Missing feature '{feature}', using 0")
                X_train_dict[feature] = np.zeros(len(train_with_results))
                X_test_dict[feature] = np.zeros(len(test_with_results))
        
        X_train = pd.DataFrame(X_train_dict)[features].values
        y_train = train_with_results['target'].values
        X_test = pd.DataFrame(X_test_dict)[features].values  
        y_test = test_with_results['target'].values
        
        print(f"🔧 Feature matrix shape: Train {X_train.shape}, Test {X_test.shape}")
        
        # Create and train Cascade Champion v2.0
        print(f"\n🏗️ Creating Cascade Champion v2.0...")
        cascade_model = CascadeChampionV20(random_state=42)
        
        # Train model
        cascade_model.fit(X_train, y_train)
        
        # Test on EPL 2025-26
        print(f"\n🧪 Testing on EPL 2025-2026...")
        test_predictions = cascade_model.predict(X_test)
        
        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"\n🏆 CASCADE CHAMPION v2.0 RESULTS:")
        print(f"   🎯 Test Accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
        print(f"   📊 Expected from metadata: 50.0%")
        print(f"   📊 Test samples: {len(X_test)}")
        
        # Target analysis
        if test_accuracy >= 0.50:
            print(f"   🎉 TARGET ACHIEVED! {test_accuracy:.1%} ≥ 50%")
        elif test_accuracy >= 0.436:
            print(f"   🎯 ABOVE BASELINE! {test_accuracy:.1%} ≥ 43.6%")
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
        
        # Detailed performance
        print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
        report = classification_report(y_test, test_predictions, target_names=class_names)
        print(report)
        
        # Confusion matrix
        print(f"\n📋 CONFUSION MATRIX:")
        cm = confusion_matrix(y_test, test_predictions)
        print("        Pred: H    D    A")
        for i, (actual, row) in enumerate(zip(class_names, cm)):
            print(f"Real {actual}: {row[0]:4d} {row[1]:4d} {row[2]:4d}")
        
        # Compare with metadata
        print(f"\n🏆 COMPARISON WITH METADATA:")
        print(f"   Expected accuracy: 50.0%")
        print(f"   Achieved accuracy: {test_accuracy:.1%}")
        print(f"   Difference: {test_accuracy - 0.50:+.3f}")
        
        if abs(test_accuracy - 0.50) < 0.05:
            print(f"   ✅ SUCCESSFUL RECONSTRUCTION! Within 5% of expected")
        else:
            print(f"   ⚠️ Reconstruction differs from expected")
        
        return test_accuracy
        
    except Exception as e:
        print(f"❌ Error testing Cascade Champion v2.0: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    print("🚀 TESTING CASCADE CHAMPION v2.0 RECONSTRUCTION")
    print("=" * 60)
    
    accuracy = test_cascade_champion_v20()
    
    print(f"\n" + "=" * 60)
    print(f"🏆 CASCADE CHAMPION v2.0 TEST COMPLETE")
    print(f"=" * 60)
    print(f"📊 Final Accuracy: {accuracy:.3f} ({accuracy:.1%})")
    
    if accuracy >= 0.50:
        print(f"🎉 SUCCESS! Matches production target")
    elif accuracy >= 0.436:
        print(f"🎯 GOOD! Above baseline target") 
    else:
        print(f"🔧 NEEDS WORK! Below baseline")
        
    print(f"\n🎉 RECONSTRUCTION TEST COMPLETE!")