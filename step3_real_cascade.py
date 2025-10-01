#!/usr/bin/env python3
"""
🎯 STEP 3: Real Enhanced Cascade Test
===================================

Using the ACTUAL Enhanced Cascade model with best parameters:
{'away_boost': 2.6, 'max_depth': 14, 'max_features': 1.0, 
 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 400}
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Best parameters from automation pipeline
best_hyperparams = {
    'away_boost': 2.6, 
    'max_depth': 14, 
    'max_features': 1.0, 
    'min_samples_leaf': 1, 
    'min_samples_split': 5, 
    'n_estimators': 400
}

def create_optimized_enhanced_cascade(hyperparams=None):
    """Create Enhanced Cascade with optimized hyperparameters"""
    
    # Use provided hyperparameters
    if hyperparams is None:
        hyperparams = best_hyperparams
    
    # Feature sets - UPDATED to use market_entropy_norm
    enhanced_features = [
        'market_entropy_norm', 'odds_spread_normalized', 
        'draw_margin_normalized', 'form_variance_diff', 'rivalry_factor'
    ]
    
    classical_features = [
        'elo_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'form_diff_normalized', 'h2h_score', 'matchday_normalized',
        'home_xg_eff_10', 'away_xg_eff_10'
    ]
    
    class OptimizedEnhancedCascade:
        def __init__(self):
            # Stage 1: Draw Detection (Enhanced features)
            self.stage1 = RandomForestClassifier(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                max_features=hyperparams['max_features'],
                class_weight='balanced',  # Critical for draw detection
                random_state=42,
                n_jobs=-1
            )
            
            # Stage 2: H/A Classification (Classical features)
            self.stage2 = RandomForestClassifier(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'] - 2,
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                max_features=hyperparams['max_features'],
                class_weight={0: 1.0, 1: hyperparams['away_boost']},  # Away boost
                random_state=42,
                n_jobs=-1
            )
            
            self.enhanced_features = enhanced_features
            self.classical_features = classical_features
            self.all_features = enhanced_features + classical_features
            self.hyperparams = hyperparams
            self.is_fitted = False
        
        def fit(self, X, y):
            """Fit both stages of the cascade"""
            X_array = np.array(X) if not isinstance(X, np.ndarray) else X
            
            # Stage 1: Draw vs Non-Draw
            y_binary = (y == 1).astype(int)  # 1 for Draw, 0 for Non-Draw
            X_stage1 = X_array[:, :len(self.enhanced_features)]
            
            self.stage1.fit(X_stage1, y_binary)
            
            # Stage 2: Home vs Away (exclude draws)
            non_draw_mask = (y != 1)
            if np.sum(non_draw_mask) > 0:
                X_stage2 = X_array[non_draw_mask][:, len(self.enhanced_features):]
                y_stage2 = y[non_draw_mask]
                y_stage2_binary = (y_stage2 == 2).astype(int)  # 1 for Away, 0 for Home
                
                if len(np.unique(y_stage2_binary)) > 1:  # Ensure we have both classes
                    self.stage2.fit(X_stage2, y_stage2_binary)
            
            self.is_fitted = True
        
        def predict(self, X):
            """Predict using 2-stage cascade with optimized thresholds"""
            if not self.is_fitted:
                raise ValueError("Model must be fitted before predicting")
                
            X_array = np.array(X) if not isinstance(X, np.ndarray) else X
            predictions = []
            
            for i in range(len(X_array)):
                sample = X_array[i:i+1]
                
                # Stage 1: Draw Detection
                X_stage1 = sample[:, :len(self.enhanced_features)]
                draw_proba = self.stage1.predict_proba(X_stage1)[0]
                
                # Optimized draw threshold (lower to reduce over-prediction)
                if draw_proba[1] > 0.30:  # Reduced from 0.35
                    prediction = 1  # Draw
                else:
                    # Stage 2: H/A Classification
                    X_stage2 = sample[:, len(self.enhanced_features):]
                    ha_proba = self.stage2.predict_proba(X_stage2)[0]
                    
                    # Apply Away boost with optimized threshold
                    away_boosted = ha_proba[1] * self.hyperparams['away_boost']
                    if away_boosted > 0.40:  # Optimized Away threshold
                        prediction = 2  # Away
                    else:
                        prediction = 0  # Home
                
                predictions.append(prediction)
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            """Return class probabilities"""
            if not self.is_fitted:
                raise ValueError("Model must be fitted before predicting")
                
            X_array = np.array(X) if not isinstance(X, np.ndarray) else X
            probabilities = []
            
            for i in range(len(X_array)):
                sample = X_array[i:i+1]
                
                # Get probabilities from both stages
                X_stage1 = sample[:, :len(self.enhanced_features)]
                draw_proba = self.stage1.predict_proba(X_stage1)[0]
                
                X_stage2 = sample[:, len(self.enhanced_features):]
                ha_proba = self.stage2.predict_proba(X_stage2)[0]
                
                # Combine probabilities with Away boost
                non_draw_prob = 1 - draw_proba[1]
                home_prob = non_draw_prob * ha_proba[0]
                away_prob = non_draw_prob * ha_proba[1] * self.hyperparams['away_boost']
                
                # Normalize
                total = home_prob + draw_proba[1] + away_prob
                if total > 0:
                    proba = [home_prob/total, draw_proba[1]/total, away_prob/total]
                else:
                    proba = [0.33, 0.33, 0.34]
                
                probabilities.append(proba)
            
            return np.array(probabilities)
    
    return OptimizedEnhancedCascade()

def test_real_cascade():
    """Test the actual Enhanced Cascade model"""
    print("🎯 TESTING REAL ENHANCED CASCADE MODEL")
    print("="*60)
    print("📋 Using optimal parameters from automation pipeline:")
    for param, value in best_hyperparams.items():
        print(f"   {param}: {value}")
    
    try:
        # Load data
        print(f"\n📂 Loading training dataset...")
        data_path = "data/processed/v_auto_update_20250922_093416.csv"
        data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(data)} matches from {data_path}")
        
        # Split data
        train_data = data[data['Season'] != '2025-2026']
        test_data = data[data['Season'] == '2025-2026']
        
        # Filter for matches with results and convert FullTimeResult to target
        train_with_results = train_data[train_data['FullTimeResult'].notna()].copy()
        test_with_results = test_data[test_data['FullTimeResult'].notna()].copy()
        
        # Convert FullTimeResult to numeric target
        result_map = {'H': 0, 'D': 1, 'A': 2}
        train_with_results['target'] = train_with_results['FullTimeResult'].map(result_map)
        test_with_results['target'] = test_with_results['FullTimeResult'].map(result_map)
        
        print(f"\n📊 Data split:")
        print(f"   Training (2019-2025): {len(train_with_results)} matches")
        print(f"   Test (2025-2026): {len(test_with_results)} matches")
        
        # Define ACTUAL cascade feature order
        # Enhanced features (first 5) + Classical features (next 8) = 13 total
        enhanced_features = [
            'market_entropy_norm', 'odds_spread_normalized', 
            'draw_margin_normalized', 'form_variance_diff', 'rivalry_factor'
        ]
        
        classical_features = [
            'elo_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'form_diff_normalized', 'h2h_score', 'matchday_normalized',
            'home_xg_eff_10', 'away_xg_eff_10'
        ]
        
        all_cascade_features = enhanced_features + classical_features
        
        # For this simplified test, use available features and fill missing ones
        available_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10'
        ]
        
        # Create feature matrix with proper cascade order
        def create_cascade_features(df, features_available):
            """Create features in cascade order, filling missing with defaults"""
            X = []
            for _, row in df.iterrows():
                # Enhanced features (first 5)
                enhanced = [
                    row.get('market_entropy_norm', 0.5),  # market_entropy_norm
                    0.5,  # odds_spread_normalized (missing)
                    0.5,  # draw_margin_normalized (missing)
                    0.5,  # form_variance_diff (missing)
                    0.5   # rivalry_factor (missing)
                ]
                
                # Classical features (next 8)
                classical = [
                    row.get('elo_diff_normalized', 0.5),
                    row.get('shots_diff_normalized', 0.5),
                    row.get('corners_diff_normalized', 0.5),
                    row.get('form_diff_normalized', 0.5),
                    row.get('h2h_score', 0.5),
                    row.get('matchday_normalized', 0.5),
                    row.get('home_xg_eff_10', 0.5),
                    row.get('away_xg_eff_10', 0.5)
                ]
                
                X.append(enhanced + classical)
            return np.array(X)
        
        # Prepare training data
        X_train = create_cascade_features(train_with_results, available_features)
        y_train = train_with_results['target'].values
        
        # Prepare test data  
        X_test = create_cascade_features(test_with_results, available_features)
        y_test = test_with_results['target'].values
        
        print(f"\n🔧 Enhanced Cascade Features:")
        print(f"   Total features: {X_train.shape[1]} (5 enhanced + 8 classical)")
        print(f"   Enhanced stage: {enhanced_features}")
        print(f"   Classical stage: {classical_features}")
        
        # Create and train the REAL Enhanced Cascade
        print(f"\n🏗️ Creating Enhanced Cascade with optimal hyperparameters...")
        cascade_model = create_optimized_enhanced_cascade(best_hyperparams)
        
        print(f"📈 Training Enhanced Cascade on {len(X_train)} samples...")
        cascade_model.fit(X_train, y_train)
        print("✅ Enhanced Cascade training completed!")
        
        # Test on EPL 2025-2026
        print(f"\n🧪 Testing Enhanced Cascade on EPL 2025-2026...")
        test_predictions = cascade_model.predict(X_test)
        
        # Calculate accuracy
        final_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"\n🏆 ENHANCED CASCADE RESULTS:")
        print(f"   🎯 Test Accuracy (EPL 2025-2026): {final_accuracy:.3f} ({final_accuracy:.1%})")
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   📊 Test samples: {len(X_test)}")
        
        # Target analysis
        if final_accuracy >= 0.436:
            print(f"   🎯 BASELINE ACHIEVED! {final_accuracy:.1%} ≥ 43.6%")
        else:
            print(f"   📊 Below baseline target ({final_accuracy:.1%} < 43.6%)")
            
        if final_accuracy >= 0.50:
            print(f"   🎉 OBJECTIVE ACHIEVED! {final_accuracy:.1%} ≥ 50%")
        elif final_accuracy >= 0.45:
            print(f"   🎯 TARGET ACHIEVED! {final_accuracy:.1%} ≥ 45%")
        else:
            print(f"   🔧 Below target ({final_accuracy:.1%} < 45%)")
        
        # Detailed breakdown
        print(f"\n📋 ENHANCED CASCADE PREDICTION BREAKDOWN:")
        class_names = ['Home', 'Draw', 'Away']
        for i, class_name in enumerate(class_names):
            actual_count = np.sum(y_test == i)
            predicted_count = np.sum(test_predictions == i)
            if len(y_test) > 0:
                actual_pct = actual_count / len(y_test) * 100
                predicted_pct = predicted_count / len(y_test) * 100
                print(f"   {class_name}: {actual_count} actual ({actual_pct:.1f}%), "
                      f"{predicted_count} predicted ({predicted_pct:.1f}%)")
        
        # Detailed classification report
        print(f"\n📊 ENHANCED CASCADE PERFORMANCE:")
        test_report = classification_report(y_test, test_predictions, target_names=class_names)
        print(test_report)
        
        # Compare with your champions
        print(f"\n🏆 COMPARISON WITH YOUR CHAMPIONS:")
        print(f"   Enhanced Cascade (this test): {final_accuracy:.1%}")
        print(f"   Baseline Champion v2.3: 53.5% (CV)")
        print(f"   Cascade Champion v2.0: 50.0% (EPL 2025-26)")
        
        if final_accuracy >= 0.50:
            print(f"   🎉 MATCHES CASCADE CHAMPION!")
        elif final_accuracy >= 0.45:
            print(f"   🎯 Good performance, room for improvement")
        else:
            print(f"   🔧 Needs optimization to match production models")
        
        return final_accuracy
        
    except Exception as e:
        print(f"❌ Error testing Enhanced Cascade: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    print("🚀 TESTING REAL ENHANCED CASCADE MODEL")
    print("="*60)
    
    final_score = test_real_cascade()
    
    print(f"\n" + "="*60)
    print(f"🏆 REAL CASCADE TEST COMPLETE")
    print(f"="*60)
    print(f"📊 Enhanced Cascade Accuracy: {final_score:.3f} ({final_score:.1%})")
    
    if final_score >= 0.50:
        print(f"🎉 SUCCESS: Matches your Cascade Champion performance!")
    elif final_score >= 0.436:
        print(f"🎯 GOOD: Above baseline, potential for optimization!")
    else:
        print(f"🔧 NEEDS WORK: Below baseline, requires improvement")
        
    print(f"\n🎉 REAL CASCADE MODEL TEST COMPLETE!")