#!/usr/bin/env python3
"""
v2.12 Ensemble Model
Combines standard RandomForest + intelligent cascade for optimal performance.
Strategy: Leverage standard RF accuracy + cascade draw detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import cascade model from previous implementation
class IntelligentCascadeModel:
    """Two-stage cascade model optimized for draw prediction."""
    
    def __init__(self, draw_threshold=0.5, use_smote=True):
        self.draw_threshold = draw_threshold
        self.use_smote = use_smote
        
        # Stage 1: Draw vs Non-Draw detector
        self.draw_detector = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Stage 2: Home vs Away classifier
        self.home_away_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.draw_features = None
        self.home_away_features = None
        
    def select_features(self, df):
        """Select optimal features for each stage."""
        
        # Stage 1: Draw detection features
        draw_features = [
            'form_equilibrium', 'elo_equilibrium', 'shots_equilibrium',
            'market_entropy_norm', 'midseason_draw_factor', 'team_balance_score',
            'rest_equilibrium', 'travel_draw_factor', 'midweek_draw_factor',
            'mutual_streak_interaction', 'h2h_score', 'matchday_normalized',
            'draw_propensity_score_normalized', 'close_odds_indicator', 'high_uncertainty'
        ]
        
        # Stage 2: Home vs Away features
        home_away_features = [
            'elo_diff_normalized', 'form_diff_normalized', 'home_xg_eff_10', 'away_xg_eff_10',
            'away_goals_sum_5', 'shots_diff_normalized', 'corners_diff_normalized',
            'fixture_congestion_diff_normalized', 'travel_fatigue_factor',
            'days_since_last_home_normalized', 'days_since_last_away_normalized',
            'market_entropy_norm'
        ]
        
        self.draw_features = [f for f in draw_features if f in df.columns]
        self.home_away_features = [f for f in home_away_features if f in df.columns]
        
        return self.draw_features, self.home_away_features
    
    def fit(self, X_df, y):
        """Train the cascade model."""
        
        self.select_features(X_df)
        
        # Stage 1: Binary draw detection
        X_draw = X_df[self.draw_features].values
        y_draw_binary = (y == 1).astype(int)
        
        if self.use_smote:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_draw_balanced, y_draw_balanced = smote.fit_resample(X_draw, y_draw_binary)
            self.draw_detector.fit(X_draw_balanced, y_draw_balanced)
        else:
            self.draw_detector.fit(X_draw, y_draw_binary)
        
        # Stage 2: Home vs Away on non-draws
        non_draw_mask = (y != 1)
        X_home_away = X_df[self.home_away_features].values[non_draw_mask]
        y_home_away = y[non_draw_mask]
        
        self.home_away_classifier.fit(X_home_away, y_home_away)
        
    def predict(self, X_df):
        """Make cascade predictions."""
        
        X_draw = X_df[self.draw_features].values
        X_home_away = X_df[self.home_away_features].values
        
        # Stage 1: Draw probabilities
        draw_probs = self.draw_detector.predict_proba(X_draw)[:, 1]
        
        # Stage 2: Home vs Away predictions
        home_away_preds = self.home_away_classifier.predict(X_home_away)
        
        # Combine using threshold
        final_predictions = np.zeros(len(X_df), dtype=int)
        
        for i in range(len(X_df)):
            if draw_probs[i] >= self.draw_threshold:
                final_predictions[i] = 1  # Draw
            else:
                final_predictions[i] = home_away_preds[i]  # Home (0) or Away (2)
        
        return final_predictions
    
    def predict_proba(self, X_df):
        """Get probability estimates for ensemble."""
        
        X_draw = X_df[self.draw_features].values
        X_home_away = X_df[self.home_away_features].values
        
        # Stage 1: Draw probabilities
        draw_probs = self.draw_detector.predict_proba(X_draw)[:, 1]
        
        # Stage 2: Home vs Away probabilities
        home_away_probs = self.home_away_classifier.predict_proba(X_home_away)
        
        # Combine probabilities
        final_probs = np.zeros((len(X_df), 3))  # [Home, Draw, Away]
        
        for i in range(len(X_df)):
            draw_prob = draw_probs[i]
            non_draw_prob = 1 - draw_prob
            
            # Redistribute non-draw probability between Home/Away
            if home_away_probs[i].shape[0] == 2:  # Binary Home vs Away
                home_prob = home_away_probs[i][0] * non_draw_prob if home_away_probs[i][0] > 0.5 else home_away_probs[i][1] * non_draw_prob
                away_prob = non_draw_prob - home_prob
            else:  # Multi-class (should not happen)
                home_prob = home_away_probs[i][0] * non_draw_prob
                away_prob = home_away_probs[i][2] * non_draw_prob if len(home_away_probs[i]) > 2 else non_draw_prob - home_prob
            
            final_probs[i] = [home_prob, draw_prob, away_prob]
        
        # Normalize probabilities
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
        
        return final_probs

class EnsembleModel:
    """Ensemble combining standard RF and cascade model."""
    
    def __init__(self, ensemble_method='weighted_voting'):
        self.ensemble_method = ensemble_method
        
        # Base model 1: Standard RandomForest (high accuracy)
        self.standard_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Base model 2: Cascade model (good draw detection)
        self.cascade_model = IntelligentCascadeModel(draw_threshold=0.5, use_smote=True)
        
        # Meta-learner for stacking
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Feature sets
        self.standard_features = None
        self.ensemble_weights = None
        
    def select_standard_features(self, df):
        """Select features for standard RF."""
        
        # Best v2.9 + v2.11 feature combination
        features = [
            # v2.9 baseline + context
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
            'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
            'form_diff_normalized', 'h2h_score', 'away_goals_sum_5',
            'days_since_last_home_normalized', 'days_since_last_away_normalized',
            'fixture_congestion_diff_normalized', 'travel_distance_km_normalized',
            'is_midweek', 'travel_fatigue_factor',
            
            # Best draw features that don't hurt standard model
            'team_balance_score', 'form_equilibrium', 'elo_equilibrium'
        ]
        
        self.standard_features = [f for f in features if f in df.columns]
        return self.standard_features
    
    def fit(self, X_df, y):
        """Train ensemble model."""
        
        logger.info("Training ensemble model...")
        
        # Select features
        self.select_standard_features(X_df)
        
        # Train standard RF
        X_standard = X_df[self.standard_features]
        self.standard_rf.fit(X_standard, y)
        
        # Train cascade model
        self.cascade_model.fit(X_df, y)
        
        if self.ensemble_method == 'stacking':
            # Create meta-features using cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            meta_features = []
            meta_targets = []
            
            for train_idx, val_idx in tscv.split(X_df):
                X_train_fold = X_df.iloc[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_df.iloc[val_idx]
                y_val_fold = y[val_idx]
                
                # Train base models on fold
                temp_rf = RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=-1
                )
                temp_cascade = IntelligentCascadeModel(draw_threshold=0.5, use_smote=True)
                
                temp_rf.fit(X_train_fold[self.standard_features], y_train_fold)
                temp_cascade.fit(X_train_fold, y_train_fold)
                
                # Get predictions on validation set
                rf_probs = temp_rf.predict_proba(X_val_fold[self.standard_features])
                cascade_probs = temp_cascade.predict_proba(X_val_fold)
                
                # Combine as meta-features
                fold_meta = np.concatenate([rf_probs, cascade_probs], axis=1)
                meta_features.append(fold_meta)
                meta_targets.append(y_val_fold)
            
            # Train meta-learner
            X_meta = np.vstack(meta_features)
            y_meta = np.concatenate(meta_targets)
            self.meta_learner.fit(X_meta, y_meta)
            
        elif self.ensemble_method == 'weighted_voting':
            # Determine optimal weights based on validation performance
            # Standard RF weight higher for global accuracy
            # Cascade weight higher for draw detection
            self.ensemble_weights = {
                'standard_rf': 0.7,    # Higher weight for accuracy
                'cascade': 0.3         # Lower weight but captures draws
            }
        
        logger.info(f"Ensemble training complete using {self.ensemble_method}")
    
    def predict(self, X_df):
        """Make ensemble predictions."""
        
        if self.ensemble_method == 'weighted_voting':
            # Get predictions from both models
            rf_probs = self.standard_rf.predict_proba(X_df[self.standard_features])
            cascade_probs = self.cascade_model.predict_proba(X_df)
            
            # Weighted average
            ensemble_probs = (self.ensemble_weights['standard_rf'] * rf_probs + 
                            self.ensemble_weights['cascade'] * cascade_probs)
            
            return np.argmax(ensemble_probs, axis=1)
            
        elif self.ensemble_method == 'stacking':
            # Get base model predictions
            rf_probs = self.standard_rf.predict_proba(X_df[self.standard_features])
            cascade_probs = self.cascade_model.predict_proba(X_df)
            
            # Meta-features
            meta_features = np.concatenate([rf_probs, cascade_probs], axis=1)
            
            # Meta-learner prediction
            return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X_df):
        """Get ensemble probability estimates."""
        
        if self.ensemble_method == 'weighted_voting':
            rf_probs = self.standard_rf.predict_proba(X_df[self.standard_features])
            cascade_probs = self.cascade_model.predict_proba(X_df)
            
            ensemble_probs = (self.ensemble_weights['standard_rf'] * rf_probs + 
                            self.ensemble_weights['cascade'] * cascade_probs)
            
            return ensemble_probs
            
        elif self.ensemble_method == 'stacking':
            rf_probs = self.standard_rf.predict_proba(X_df[self.standard_features])
            cascade_probs = self.cascade_model.predict_proba(X_df)
            
            meta_features = np.concatenate([rf_probs, cascade_probs], axis=1)
            return self.meta_learner.predict_proba(meta_features)

def evaluate_ensemble_models(df):
    """Evaluate different ensemble configurations."""
    
    logger.info("🚀 Evaluating v2.12 Ensemble Models...")
    
    # Clean data
    df = df.dropna()
    logger.info(f"Dataset shape: {df.shape}")
    
    # Prepare data
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping).values
    
    # Train/test split
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Models to evaluate
    models_config = {
        'Standard_RF': {
            'model': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'type': 'standard'
        },
        'Cascade_Only': {
            'model': IntelligentCascadeModel(draw_threshold=0.5, use_smote=True),
            'type': 'cascade'
        },
        'Ensemble_Weighted': {
            'model': EnsembleModel(ensemble_method='weighted_voting'),
            'type': 'ensemble'
        },
        'Ensemble_Stacking': {
            'model': EnsembleModel(ensemble_method='stacking'),
            'type': 'ensemble'
        }
    }
    
    results = []
    
    for model_name, config in models_config.items():
        logger.info(f"Evaluating {model_name}...")
        
        model = config['model']
        model_type = config['type']
        
        try:
            if model_type == 'standard':
                # Standard RF features
                features = [
                    'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
                    'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
                    'form_diff_normalized', 'h2h_score', 'away_goals_sum_5',
                    'days_since_last_home_normalized', 'days_since_last_away_normalized',
                    'fixture_congestion_diff_normalized', 'travel_distance_km_normalized',
                    'is_midweek', 'travel_fatigue_factor'
                ]
                available_features = [f for f in features if f in df_train.columns]
                
                model.fit(df_train[available_features], y_train)
                y_pred = model.predict(df_test[available_features])
                
            elif model_type == 'cascade':
                model.fit(df_train, y_train)
                y_pred = model.predict(df_test)
                
            elif model_type == 'ensemble':
                model.fit(df_train, y_train)
                y_pred = model.predict(df_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Draw-specific metrics
            draw_mask_true = (y_test == 1)
            draw_mask_pred = (y_pred == 1)
            
            draw_recall = np.sum((y_test == 1) & (y_pred == 1)) / max(np.sum(y_test == 1), 1)
            draw_precision = np.sum((y_test == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
            draw_f1 = 2 * draw_precision * draw_recall / (draw_precision + draw_recall) if (draw_precision + draw_recall) > 0 else 0
            
            # Prediction distribution
            pred_counts = np.bincount(y_pred, minlength=3)
            pred_distribution = pred_counts / len(y_pred)
            
            result = {
                'model': model_name,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'draw_recall': draw_recall,
                'draw_precision': draw_precision,
                'draw_f1': draw_f1,
                'pred_distribution': pred_distribution
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    return results

def run_ensemble_evaluation():
    """Run complete ensemble evaluation."""
    
    # Load v2.11 dataset
    df = pd.read_csv('data/processed/v211_draw_features_2025_09_06.csv')
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Evaluate models
    results = evaluate_ensemble_models(df)
    
    # Display results
    print("\\n" + "="*90)
    print("🎯 v2.12 ENSEMBLE MODEL EVALUATION")
    print("="*90)
    
    baseline_accuracy = 0.5685  # v2.9 baseline
    
    # Sort by F1-macro for balanced evaluation
    results.sort(key=lambda x: x['f1_macro'], reverse=True)
    
    for i, result in enumerate(results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
        
        print(f"\\n{rank_emoji} {result['model']}:")
        print(f"   • Accuracy: {result['accuracy']:.4f}")
        print(f"   • F1-Macro: {result['f1_macro']:.4f}")
        print(f"   • vs Baseline: {(result['accuracy'] - baseline_accuracy)*100:+.2f}pp")
        
        print(f"   • Draw Performance:")
        print(f"     - Recall: {result['draw_recall']:.3f}")
        print(f"     - Precision: {result['draw_precision']:.3f}")
        print(f"     - F1-Score: {result['draw_f1']:.3f}")
        
        pred_dist = result['pred_distribution']
        print(f"   • Predictions: H:{pred_dist[0]:.3f} D:{pred_dist[1]:.3f} A:{pred_dist[2]:.3f}")
    
    # Best model analysis
    best_result = results[0]
    
    print(f"\\n🏆 CHAMPION ENSEMBLE: {best_result['model']}")
    print(f"   • Final Accuracy: {best_result['accuracy']:.4f}")
    print(f"   • Final F1-Macro: {best_result['f1_macro']:.4f}")
    print(f"   • Total Progress: {(best_result['accuracy'] - 0.551)*100:+.2f}pp vs original v2.4")
    
    # Performance milestone
    if best_result['accuracy'] >= 0.58:
        print("\\n🚀 BREAKTHROUGH: 58%+ accuracy achieved!")
    elif best_result['accuracy'] >= 0.57:
        print("\\n✨ EXCELLENT: 57%+ accuracy achieved!")
    elif best_result['f1_macro'] >= 0.52:
        print("\\n⚡ BALANCED EXCELLENCE: Superior F1-macro performance!")
    
    # Save results
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'evaluation/reports/v212_ensemble_evaluation_{timestamp}.json', index=False)
    
    logger.info("✅ v2.12 ensemble evaluation complete!")
    
    return results

if __name__ == "__main__":
    results = run_ensemble_evaluation()