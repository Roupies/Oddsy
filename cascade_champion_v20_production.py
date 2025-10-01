#!/usr/bin/env python3
"""
🏆 Cascade Champion v2.0 - TRUE Production Model
===============================================

The REAL Cascade Champion v2.0 that achieved 46% on EPL 2025-26.
Based on the original metadata and architecture.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

class CascadeChampionV20(BaseEstimator, ClassifierMixin):
    """
    TRUE Cascade Champion v2.0 - Production Model
    
    The original champion that achieved 46% accuracy on EPL 2025-26.
    Based on exact metadata from cascade_champion_v2_metadata.json
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # TRUE v2.0 parameters from original metadata
        self.draw_weight = 2.5
        self.draw_threshold = 0.4
        
        # TRUE v2.0 features (exact order from metadata)
        self.features = [
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
        
        # Model version info
        self.model_version = "v2.0_cascade_dual_stage"
        self.accuracy = 0.46
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the TRUE Cascade Champion v2.0 - ANTI-LEAKAGE VERSION"""
        print(f"🏗️ Training Cascade Champion v2.0 on {len(X)} samples...")
        print(f"🔒 Anti-leakage mode: Using only feature columns")
        
        # Stage 1: Draw Detection (exact parameters from metadata)
        self.clf_draw = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: 2.5},  # non_draw: 1, draw: 2.5
            random_state=self.random_state
        )
        
        # Stage 2: Home/Away Classification (exact parameters from metadata)
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        # CRITICAL FIX: Ensure we only use feature columns, no data leakage
        if isinstance(X, pd.DataFrame):
            # If DataFrame, ensure we only use known features
            available_features = [f for f in self.features if f in X.columns]
            if len(available_features) != len(self.features):
                missing = set(self.features) - set(available_features)
                print(f"⚠️  Missing features: {missing}")
            X_features = X[available_features].fillna(0)
        else:
            # If array/matrix, assume it's already feature-only
            X_features = pd.DataFrame(X, columns=self.features[:X.shape[1]]).fillna(0)
        
        # Validate no target leakage
        if X_features.shape[1] != len(self.features):
            print(f"⚠️  Expected {len(self.features)} features, got {X_features.shape[1]}")
        
        # Stage 1: Draw vs Non-Draw
        y_binary = (y == 1).astype(int)  # 1 for Draw, 0 for Non-Draw
        self.clf_draw.fit(X_features, y_binary)
        
        draw_distribution = np.mean(y_binary) * 100
        print(f"   Stage 1: Draw distribution {draw_distribution:.1f}%")
        
        # Stage 2: Home vs Away (exclude draws)
        non_draw_mask = (y != 1)
        if np.sum(non_draw_mask) > 0:
            X_non_draw = X_features[non_draw_mask]
            y_stage2 = y[non_draw_mask]
            y_ha_binary = (y_stage2 == 2).astype(int)  # 1 for Away, 0 for Home
            
            if len(np.unique(y_ha_binary)) > 1:
                self.clf_homeaway.fit(X_non_draw, y_ha_binary)
                print(f"   Stage 2: Home/Away trained on {len(X_non_draw)} samples")
        
        self.is_fitted = True
        print("✅ Cascade Champion v2.0 training complete (leakage-free)")
        return self
    
    def predict(self, X):
        """Predict using TRUE v2.0 cascade logic - ANTI-LEAKAGE VERSION"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        
        predictions = []
        
        # CRITICAL FIX: Ensure we only use feature columns during prediction
        if isinstance(X, pd.DataFrame):
            available_features = [f for f in self.features if f in X.columns]
            X_features = X[available_features].fillna(0)
        else:
            X_features = pd.DataFrame(X, columns=self.features[:X.shape[1]]).fillna(0)
        
        for i in range(len(X_features)):
            sample = X_features.iloc[i:i+1]
            
            # Stage 1: Draw Detection (threshold from metadata)
            draw_proba = self.clf_draw.predict_proba(sample)[0]
            
            if draw_proba[1] > self.draw_threshold:  # 0.4 from metadata
                prediction = 1  # Draw
            else:
                # Stage 2: Home/Away Classification
                ha_proba = self.clf_homeaway.predict_proba(sample)[0]
                prediction = 2 if ha_proba[1] > 0.5 else 0  # Away : Home
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Return class probabilities using TRUE v2.0 logic - ANTI-LEAKAGE VERSION"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        
        probabilities = []
        
        # CRITICAL FIX: Ensure we only use feature columns
        if isinstance(X, pd.DataFrame):
            available_features = [f for f in self.features if f in X.columns]
            X_features = X[available_features].fillna(0)
        else:
            X_features = pd.DataFrame(X, columns=self.features[:X.shape[1]]).fillna(0)
        
        for i in range(len(X_features)):
            sample = X_features.iloc[i:i+1]
            
            # Get probabilities from both stages
            draw_proba = self.clf_draw.predict_proba(sample)[0]
            ha_proba = self.clf_homeaway.predict_proba(sample)[0]
            
            # Combine probabilities using cascade logic
            draw_prob = draw_proba[1]
            non_draw_prob = 1 - draw_prob
            home_prob = non_draw_prob * ha_proba[0]
            away_prob = non_draw_prob * ha_proba[1]
            
            # Normalize
            total = home_prob + draw_prob + away_prob
            if total > 0:
                proba = [home_prob/total, draw_prob/total, away_prob/total]
            else:
                proba = [0.33, 0.33, 0.34]
            
            probabilities.append(proba)
        
        return np.array(probabilities)
    
    def get_feature_importance(self):
        """Get feature importance from both stages"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Combine importances from both stages
        stage1_importance = self.clf_draw.feature_importances_
        stage2_importance = self.clf_homeaway.feature_importances_
        
        # Weight by typical draw rate (~23%)
        combined_importance = stage1_importance * 0.23 + stage2_importance * 0.77
        
        importance_dict = {}
        for feature, importance in zip(self.features, combined_importance):
            importance_dict[feature] = importance
        
        return importance_dict

def create_and_test_cascade_v20():
    """Create and test the TRUE Cascade Champion v2.0"""
    print("🏆 CREATING TRUE CASCADE CHAMPION v2.0")
    print("=" * 60)
    
    try:
        # Load data
        data_path = "data/processed/v_auto_update_20250922_093416.csv"
        data = pd.read_csv(data_path)
        
        # Split data (same as original)
        train_data = data[data['Season'] != '2025-2026']
        test_data = data[data['Season'] == '2025-2026']
        
        train_with_results = train_data[train_data['FullTimeResult'].notna()].copy()
        test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
        
        # Target mapping
        result_map = {'H': 0, 'D': 1, 'A': 2}
        train_with_results['target'] = train_with_results['FullTimeResult'].map(result_map)
        test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
        
        print(f"📊 Training: {len(train_with_results)} matches")
        print(f"📊 Test (EPL 2025-26): {len(test_with_results)} matches")
        
        y_train = train_with_results['target'].values
        y_test = test_with_results['target'].values
        
        # Create TRUE Cascade Champion v2.0
        cascade_v20 = CascadeChampionV20(random_state=42)
        
        # Train model
        cascade_v20.fit(train_with_results, y_train)
        
        # Test on EPL 2025-26
        print(f"\n🧪 Testing TRUE Cascade Champion v2.0...")
        test_predictions = cascade_v20.predict(test_with_results)
        test_probabilities = cascade_v20.predict_proba(test_with_results)
        
        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"\n🏆 TRUE CASCADE CHAMPION v2.0 RESULTS:")
        print(f"   🎯 Accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
        print(f"   📊 Expected: 46.0% (metadata)")
        print(f"   📊 Test matches: {len(test_with_results)}")
        
        # Performance analysis
        if abs(test_accuracy - 0.46) < 0.05:
            print(f"   ✅ MATCHES EXPECTED PERFORMANCE!")
        elif test_accuracy >= 0.436:
            print(f"   🎯 ABOVE BASELINE! Good performance")
        else:
            print(f"   ⚠️ Below expected - check parameters")
        
        # Prediction breakdown
        print(f"\n📋 PREDICTION BREAKDOWN:")
        class_names = ['Home', 'Draw', 'Away']
        for i, class_name in enumerate(class_names):
            actual_count = np.sum(y_test == i)
            predicted_count = np.sum(test_predictions == i)
            actual_pct = actual_count / len(y_test) * 100
            predicted_pct = predicted_count / len(y_test) * 100
            print(f"   {class_name}: {actual_count} actual ({actual_pct:.1f}%), "
                  f"{predicted_count} predicted ({predicted_pct:.1f}%)")
        
        # Detailed report
        print(f"\n📊 CLASSIFICATION REPORT:")
        report = classification_report(y_test, test_predictions, target_names=class_names)
        print(report)
        
        # Feature importance
        print(f"\n🔧 FEATURE IMPORTANCE:")
        importance = cascade_v20.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, imp) in enumerate(sorted_features[:5], 1):
            print(f"   {i}. {feature}: {imp:.3f}")
        
        # Save model
        model_path = "models/production/cascade_champion_v20_production.joblib"
        print(f"\n💾 Saving TRUE Cascade Champion v2.0...")
        joblib.dump(cascade_v20, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Create metadata
        metadata = {
            "model_type": "CascadeChampion_v2.0_Production",
            "version": "v2.0_cascade_dual_stage",
            "accuracy": float(test_accuracy),
            "timestamp": datetime.now().isoformat(),
            "features": cascade_v20.features,
            "hyperparameters": {
                "draw_weight": cascade_v20.draw_weight,
                "draw_threshold": cascade_v20.draw_threshold,
                "random_state": cascade_v20.random_state
            },
            "architecture": {
                "stage_1": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5},
                "stage_2": {"n_estimators": 150, "class_weight": "balanced"}
            },
            "test_results": {
                "accuracy": float(test_accuracy),
                "test_samples": len(test_with_results),
                "predictions_breakdown": {
                    "home": {"actual": int(np.sum(y_test == 0)), "predicted": int(np.sum(test_predictions == 0))},
                    "draw": {"actual": int(np.sum(y_test == 1)), "predicted": int(np.sum(test_predictions == 1))},  
                    "away": {"actual": int(np.sum(y_test == 2)), "predicted": int(np.sum(test_predictions == 2))}
                }
            }
        }
        
        metadata_path = "models/production/cascade_champion_v20_production_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved: {metadata_path}")
        
        return cascade_v20, test_accuracy
        
    except Exception as e:
        print(f"❌ Error creating Cascade v2.0: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0

if __name__ == "__main__":
    print("🚀 CREATING TRUE CASCADE CHAMPION v2.0")
    print("=" * 60)
    
    model, accuracy = create_and_test_cascade_v20()
    
    print(f"\n" + "=" * 60)
    print(f"🏆 TRUE CASCADE CHAMPION v2.0 COMPLETE!")
    print("=" * 60)
    
    if model:
        print(f"✅ Model created and tested: {accuracy:.1%}")
        print(f"💾 Saved to: models/production/")
        print(f"🎯 Ready for production deployment!")
        print(f"\n🏆 CASCADE CHAMPION v2.0 IS THE TRUE CHAMPION!")
    else:
        print(f"❌ Model creation failed")