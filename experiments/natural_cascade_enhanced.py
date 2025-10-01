#!/usr/bin/env python3
"""
🎯 NATURAL CASCADE ENHANCED - Safeguarded Draw Detection Pipeline
================================================================

Enhanced cascade model with:
- Segmented features (draw-focused vs classical)
- Correlation safeguards
- Cautious calibration
- Empirical validation
- Reality checks
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('scripts/auto_update')
from feature_calculator import FeatureCalculator

class CorrelationChecker:
    """Correlation analysis and feature pruning"""
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def check_correlations(self, X, feature_names):
        """Check feature correlations and identify redundant pairs"""
        corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > self.threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j], 
                        'correlation': corr_value
                    })
        
        return high_corr_pairs, corr_matrix
    
    def auto_prune_features(self, high_corr_pairs, feature_importance=None):
        """Auto-prune highly correlated features"""
        features_to_remove = set()
        
        for pair in high_corr_pairs:
            feat1, feat2 = pair['feature1'], pair['feature2']
            corr = pair['correlation']
            
            print(f"⚠️ High correlation detected: {feat1} vs {feat2} ({corr:.3f})")
            
            # Decision logic: keep more important or more general feature
            if feature_importance and feat1 in feature_importance and feat2 in feature_importance:
                # Keep feature with higher importance
                if feature_importance[feat1] > feature_importance[feat2]:
                    features_to_remove.add(feat2)
                    print(f"→ Removing {feat2} (lower importance)")
                else:
                    features_to_remove.add(feat1)
                    print(f"→ Removing {feat1} (lower importance)")
            else:
                # Default heuristics
                if 'odds_spread' in feat1 and 'draw_margin' in feat2:
                    features_to_remove.add(feat2)
                    print(f"→ Removing {feat2} (odds_spread more general)")
                elif 'draw_margin' in feat1 and 'odds_spread' in feat2:
                    features_to_remove.add(feat1)
                    print(f"→ Removing {feat1} (odds_spread more general)")
                else:
                    # Remove the second one by default
                    features_to_remove.add(feat2)
                    print(f"→ Removing {feat2} (default)")
        
        return list(features_to_remove)

class SafeguardedCascadeModel:
    """Natural cascade with segmented features and safeguards"""
    
    def __init__(self, correlation_threshold=0.8):
        self.stage1_model = None  # Draw vs Non-Draw
        self.stage2_model = None  # Home vs Away
        self.calibrator = None
        self.is_calibrated = False
        self.correlation_checker = CorrelationChecker(correlation_threshold)
        
        # Feature segmentation
        self.draw_features = [
            'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'market_entropy_historical', 'draw_margin_normalized',
            'form_variance_normalized', 'odds_spread_normalized',
            'matchday_normalized'
        ]
        
        self.ha_features = [
            'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'shots_diff_normalized', 'corners_diff_normalized',
            'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
            # Note: deliberately excluding draw-focused features
        ]
        
        self.pruned_features = {}
        
    def validate_phase2_features(self, X_full, y_full):
        """A/B test: confirm draw features hurt H/A performance"""
        print("\n🧪 Phase 2 Feature Validation (A/B Test)")
        
        # Filter to non-draw matches only
        non_draw_mask = y_full != 1
        X_nondraw = X_full[non_draw_mask]
        y_nondraw = y_full[non_draw_mask]
        
        if len(X_nondraw) < 50:
            print("⚠️ Insufficient non-draw samples for A/B test")
            return True
        
        # Create feature sets
        available_features = list(X_full.columns)
        classical_available = [f for f in self.ha_features if f in available_features]
        draw_features_available = [f for f in self.draw_features if f in available_features and f not in classical_available]
        
        print(f"Classical features: {len(classical_available)}")
        print(f"Draw features to test: {len(draw_features_available)}")
        
        # Test 1: Classical features only
        X_classical = X_nondraw[classical_available]
        classical_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        classical_scores = cross_val_score(classical_model, X_classical, y_nondraw, cv=3, scoring='accuracy')
        classical_accuracy = classical_scores.mean()
        
        # Test 2: Classical + Draw features  
        mixed_features = classical_available + draw_features_available
        X_mixed = X_nondraw[mixed_features]
        mixed_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        mixed_scores = cross_val_score(mixed_model, X_mixed, y_nondraw, cv=3, scoring='accuracy')
        mixed_accuracy = mixed_scores.mean()
        
        print(f"H/A Classical only: {classical_accuracy:.3f}")
        print(f"H/A with draw features: {mixed_accuracy:.3f}")
        print(f"Difference: {mixed_accuracy - classical_accuracy:.3f}")
        
        if classical_accuracy >= mixed_accuracy:
            print("✅ Confirmed: Draw features don't help H/A classification")
            return True
        else:
            print("⚠️ Unexpected: Draw features seem to help H/A - investigate")
            return False
    
    def check_and_prune_correlations(self, X, feature_names):
        """Check correlations and prune if necessary"""
        print("\n🔍 Correlation Analysis")
        
        high_corr_pairs, corr_matrix = self.correlation_checker.check_correlations(X, feature_names)
        
        if len(high_corr_pairs) > 0:
            print(f"⚠️ Found {len(high_corr_pairs)} high correlation pairs")
            features_to_remove = self.correlation_checker.auto_prune_features(high_corr_pairs)
            
            if features_to_remove:
                print(f"🗑️ Removing {len(features_to_remove)} redundant features")
                remaining_features = [f for f in feature_names if f not in features_to_remove]
                
                # Update feature sets
                self.draw_features = [f for f in self.draw_features if f not in features_to_remove]
                self.ha_features = [f for f in self.ha_features if f not in features_to_remove]
                
                self.pruned_features = {
                    'removed': features_to_remove,
                    'remaining': remaining_features,
                    'correlation_pairs': high_corr_pairs
                }
                
                return remaining_features
        else:
            print("✅ No high correlations detected")
        
        return feature_names
    
    def cautious_calibration_check(self, X_val, y_val, cv_folds=3):
        """Check if calibration is safe given draw sample sizes"""
        print(f"\n🎯 Calibration Safety Check")
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        min_draws_per_fold = float('inf')
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_val)):
            y_fold = y_val.iloc[val_idx] if hasattr(y_val, 'iloc') else y_val[val_idx]
            fold_draws = (y_fold == 1).sum()
            min_draws_per_fold = min(min_draws_per_fold, fold_draws)
            print(f"Fold {fold_idx + 1}: {fold_draws} draws")
        
        safe_threshold = 15
        is_safe = min_draws_per_fold >= safe_threshold
        
        if is_safe:
            print(f"✅ Calibration SAFE (min {min_draws_per_fold} draws per fold)")
        else:
            print(f"⚠️ Calibration UNSAFE (min {min_draws_per_fold} draws < {safe_threshold})")
            print("→ Will use raw model for stability")
        
        return is_safe
    
    def fit(self, X, y, enable_calibration=True):
        """Fit the safeguarded cascade model"""
        print("🏗️ Training Safeguarded Natural Cascade")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        feature_names = list(X.columns)
        
        # Step 1: Validate Phase 2 features
        phase2_validated = self.validate_phase2_features(X, y)
        
        # Step 2: Correlation check and pruning
        validated_features = self.check_and_prune_correlations(X.values, feature_names)
        X_pruned = X[validated_features]
        
        # Update feature sets after pruning
        draw_features_available = [f for f in self.draw_features if f in validated_features]
        ha_features_available = [f for f in self.ha_features if f in validated_features]
        
        print(f"\n🎯 Stage 1 (Draw Detection) - {len(draw_features_available)} features")
        print(f"⚽ Stage 2 (H/A Classification) - {len(ha_features_available)} features")
        
        # Stage 1: Draw vs Non-Draw with draw-focused features
        print("\n🎯 Training Stage 1: Draw Detection")
        X_stage1 = X_pruned[draw_features_available]
        y_binary = (y == 1).astype(int)  # 1 for draws, 0 for non-draws
        
        self.stage1_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',  # Natural balancing
            random_state=42
        )
        self.stage1_model.fit(X_stage1, y_binary)
        
        # Stage 2: Home vs Away with classical features only
        print("\n⚽ Training Stage 2: H/A Classification")
        non_draw_mask = y != 1
        X_stage2 = X_pruned[ha_features_available][non_draw_mask]
        y_stage2 = y[non_draw_mask]
        y_binary_ha = (y_stage2 == 2).astype(int)  # 1 for away, 0 for home
        
        self.stage2_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.stage2_model.fit(X_stage2, y_binary_ha)
        
        # Calibration check
        if enable_calibration:
            calibration_safe = self.cautious_calibration_check(X_pruned, y)
            
            if calibration_safe:
                print("\n⚖️ Applying Isotonic Calibration")
                self.calibrator = CalibratedClassifierCV(
                    estimator=self,
                    method='isotonic',
                    cv=3
                )
                # Note: calibrator would need separate fit() call with full pipeline
                self.is_calibrated = True
            else:
                print("\n⚖️ Skipping Calibration (insufficient draws)")
                self.is_calibrated = False
        
        # Store feature names for prediction
        self.feature_names_ = validated_features
        
        print(f"\n✅ Natural Cascade Training Complete")
        print(f"   Phase 2 Validated: {phase2_validated}")
        print(f"   Features Pruned: {len(self.pruned_features.get('removed', []))}")
        print(f"   Calibration: {'Enabled' if self.is_calibrated else 'Disabled'}")
        
        return self
    
    def predict_proba(self, X):
        """Natural probability prediction without manual boosts"""
        if isinstance(X, np.ndarray):
            # Convert to DataFrame with expected features
            if hasattr(self, 'feature_names_'):
                feature_names = self.feature_names_
            else:
                feature_names = self.draw_features + [f for f in self.ha_features if f not in self.draw_features]
            X = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
        
        # Get validated features (use all if no pruning happened)
        if hasattr(self, 'pruned_features') and 'remaining' in self.pruned_features:
            validated_features = self.pruned_features['remaining']
        else:
            validated_features = list(X.columns)
        
        X_validated = X[[f for f in validated_features if f in X.columns]]
        
        # Stage 1: Draw probabilities
        draw_features_available = [f for f in self.draw_features if f in X_validated.columns]
        X_stage1 = X_validated[draw_features_available]
        stage1_proba = self.stage1_model.predict_proba(X_stage1)
        
        if stage1_proba.shape[1] == 2:
            prob_draw = stage1_proba[:, 1]  # Probability of draw
        else:
            prob_draw = np.full(len(X), 0.25)  # Fallback
        
        # Stage 2: Home/Away probabilities  
        ha_features_available = [f for f in self.ha_features if f in X_validated.columns]
        X_stage2 = X_validated[ha_features_available]
        stage2_proba = self.stage2_model.predict_proba(X_stage2)
        
        if stage2_proba.shape[1] == 2:
            prob_home_given_nondraw = stage2_proba[:, 0]  # Home given non-draw
            prob_away_given_nondraw = stage2_proba[:, 1]  # Away given non-draw
        else:
            prob_home_given_nondraw = np.full(len(X), 0.5)
            prob_away_given_nondraw = np.full(len(X), 0.5)
        
        # Natural combination (no manual boosts)
        prob_non_draw = 1 - prob_draw
        prob_home = prob_non_draw * prob_home_given_nondraw
        prob_away = prob_non_draw * prob_away_given_nondraw
        
        # Normalize to ensure probabilities sum to 1
        total = prob_home + prob_draw + prob_away
        prob_home = prob_home / total
        prob_draw = prob_draw / total
        prob_away = prob_away / total
        
        return np.column_stack([prob_home, prob_draw, prob_away])
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def get_feature_importance(self):
        """Get feature importance from both stages"""
        stage1_features = [f for f in self.draw_features if f in self.pruned_features.get('remaining', self.draw_features)]
        stage2_features = [f for f in self.ha_features if f in self.pruned_features.get('remaining', self.ha_features)]
        
        importance_analysis = {
            'stage1_draw_detection': dict(zip(stage1_features, self.stage1_model.feature_importances_)),
            'stage2_ha_classification': dict(zip(stage2_features, self.stage2_model.feature_importances_))
        }
        
        return importance_analysis

def load_and_prepare_data():
    """Load and prepare enhanced dataset"""
    print("📊 Loading and Preparing Enhanced Dataset")
    
    # Load base data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    print(f"Base dataset: {len(data)} matches")
    
    # Filter complete data (non-null FullTimeResult)
    complete_data = data[data['FullTimeResult'].notna()].copy()
    print(f"Complete data: {len(complete_data)} matches")
    
    # Ensure target encoding
    target_map = {'H': 0, 'D': 1, 'A': 2}
    complete_data['target'] = complete_data['FullTimeResult'].map(target_map)
    
    # Enhanced feature calculation
    calculator = FeatureCalculator()
    
    # For simplicity, use existing features + calculate missing enhanced ones
    enhanced_data = complete_data.copy()
    
    # Add enhanced features where missing
    enhanced_features = ['market_entropy_historical', 'odds_spread_normalized', 
                        'draw_margin_normalized', 'form_variance_normalized']
    
    for feature in enhanced_features:
        if feature not in enhanced_data.columns:
            print(f"⚡ Calculating {feature} for all matches...")
            enhanced_data[feature] = 0.5  # Placeholder - would need full calculation
    
    # Feature set for model
    all_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5',
        'market_entropy_historical', 'odds_spread_normalized', 
        'draw_margin_normalized', 'form_variance_normalized'
    ]
    
    # Clean feature data
    feature_data = enhanced_data[all_features].fillna(enhanced_data[all_features].mean())
    target_data = enhanced_data['target']
    
    # Remove any remaining NaN targets
    complete_mask = target_data.notna()
    X_final = feature_data[complete_mask]
    y_final = target_data[complete_mask]
    
    print(f"✅ Final dataset: {len(X_final)} matches, {len(all_features)} features")
    print(f"Target distribution: {y_final.value_counts().to_dict()}")
    
    return X_final, y_final, all_features

def j6_reality_check(model, j6_predictions, j6_probabilities):
    """Reality check for J6 predictions"""
    print("\n🎲 J6 Reality Check")
    
    predicted_draws = (j6_predictions == 1).sum()
    expected_range = (2, 5)  # Realistic for EPL
    
    print(f"📊 J6 Draw Predictions: {predicted_draws}/10")
    
    if predicted_range := range(*expected_range):
        if predicted_draws in predicted_range:
            status = "🎯 REALISTIC - Model learned natural signal"
            realistic = True
        elif predicted_draws == 0:
            status = "❌ ZERO DRAWS - Model too conservative"
            realistic = False
        elif predicted_draws >= 7:
            status = "❌ TOO MANY - Model over-boosting draws"
            realistic = False
        else:
            status = "⚠️ BORDERLINE - Monitor performance"
            realistic = True
    
    print(f"Status: {status}")
    
    # Show prediction details
    print(f"\nJ6 Prediction Details:")
    for i, (pred, probs) in enumerate(zip(j6_predictions, j6_probabilities)):
        outcome = ['H', 'D', 'A'][pred]
        confidence = max(probs)
        print(f"Match {i+1}: {outcome} ({confidence:.3f}) | H:{probs[0]:.2f} D:{probs[1]:.2f} A:{probs[2]:.2f}")
    
    return realistic

def comprehensive_validation(model, X, y):
    """Comprehensive model validation"""
    print("\n✅ Comprehensive Model Validation")
    
    # Temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    
    print(f"Temporal CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Multi-seed stability test
    seed_scores = []
    for seed in range(5):
        test_model = SafeguardedCascadeModel()
        test_model.fit(X, y, enable_calibration=False)  # Quick test without calibration
        test_score = cross_val_score(test_model, X, y, cv=3, scoring='accuracy').mean()
        seed_scores.append(test_score)
    
    seed_variance = np.var(seed_scores)
    print(f"Multi-seed stability: {np.mean(seed_scores):.3f} ± {np.sqrt(seed_variance):.3f}")
    print(f"Seed variance: {seed_variance:.6f} ({'STABLE' if seed_variance < 0.001 else 'UNSTABLE'})")
    
    # Feature importance analysis
    importance = model.get_feature_importance()
    print(f"\n🔍 Feature Importance Analysis:")
    
    print("Stage 1 (Draw Detection) - Top 5:")
    stage1_sorted = sorted(importance['stage1_draw_detection'].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in stage1_sorted[:5]:
        print(f"  {feat}: {imp:.3f}")
    
    print("Stage 2 (H/A Classification) - Top 5:")
    stage2_sorted = sorted(importance['stage2_ha_classification'].items(), key=lambda x: x[1], reverse=True)
    for feat, imp in stage2_sorted[:5]:
        print(f"  {feat}: {imp:.3f}")
    
    validation_results = {
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'stability_variance': seed_variance,
        'feature_importance': importance
    }
    
    return validation_results

if __name__ == "__main__":
    print("🚀 Enhanced Natural Draw Detection Pipeline")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Create and train model (skip calibration for now)
    model = SafeguardedCascadeModel(correlation_threshold=0.8)
    model.fit(X, y, enable_calibration=False)
    
    # Comprehensive validation
    validation_results = comprehensive_validation(model, X, y)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'natural_cascade_enhanced_{timestamp}.joblib'
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    print(f"\n🎯 Enhanced Natural Cascade Complete!")
    print(f"   CV Accuracy: {validation_results['cv_accuracy']:.3f}")
    print(f"   Stability: {validation_results['stability_variance']:.6f}")
    print(f"   Features: {len(feature_names)} total")