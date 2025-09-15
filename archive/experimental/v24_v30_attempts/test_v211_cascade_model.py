#!/usr/bin/env python3
"""
v2.11 Intelligent Cascade Model
Improved two-stage cascade using draw-specific features and lessons from v2.4.
Focus: Better draw detection without sacrificing global accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentCascadeModel:
    """Two-stage cascade model optimized for draw prediction."""
    
    def __init__(self, draw_threshold=0.4, use_smote=True):
        self.draw_threshold = draw_threshold
        self.use_smote = use_smote
        
        # Stage 1: Draw vs Non-Draw detector
        self.draw_detector = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            class_weight='balanced',  # Handle draw imbalance
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
        
        # Stage 1: Draw detection features (equilibrium focused)
        draw_features = [
            # Core draw indicators
            'elo_equilibrium', 'team_balance_score', 'draw_propensity_score_normalized',
            'shots_equilibrium', 'form_equilibrium', 'rest_equilibrium',
            
            # Market signals
            'market_entropy_norm', 'close_odds_indicator', 'high_uncertainty',
            
            # Context factors
            'travel_draw_factor', 'midseason_draw_factor', 'midweek_draw_factor',
            'mutual_streak_interaction',
            
            # Traditional features that help draw detection
            'matchday_normalized', 'h2h_score'
        ]
        
        # Stage 2: Home vs Away features (dominance focused)
        home_away_features = [
            # Team strength
            'elo_diff_normalized', 'form_diff_normalized',
            
            # Performance metrics
            'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5',
            'shots_diff_normalized', 'corners_diff_normalized',
            
            # Context advantages
            'fixture_congestion_diff_normalized', 'travel_fatigue_factor',
            'days_since_last_home_normalized', 'days_since_last_away_normalized',
            
            # Market intelligence
            'market_entropy_norm'
        ]
        
        # Filter available features
        self.draw_features = [f for f in draw_features if f in df.columns]
        self.home_away_features = [f for f in home_away_features if f in df.columns]
        
        logger.info(f"Selected {len(self.draw_features)} draw features and {len(self.home_away_features)} home/away features")
        
        return self.draw_features, self.home_away_features
    
    def fit(self, df):
        """Train the cascade model."""
        
        # Select features
        self.select_features(df)
        
        # Prepare targets
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y_full = df['FullTimeResult'].map(target_mapping).values
        
        # Stage 1: Train draw detector (binary: Draw=1, Non-Draw=0)
        X_draw = df[self.draw_features].values
        y_draw_binary = (y_full == 1).astype(int)  # 1 for Draw, 0 for Non-Draw
        
        # Optional SMOTE for draw detection
        if self.use_smote:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_draw_balanced, y_draw_balanced = smote.fit_resample(X_draw, y_draw_binary)
            logger.info(f"SMOTE applied: {len(X_draw)} -> {len(X_draw_balanced)} samples")
            self.draw_detector.fit(X_draw_balanced, y_draw_balanced)
        else:
            self.draw_detector.fit(X_draw, y_draw_binary)
        
        # Stage 2: Train home vs away classifier (only on non-draw samples)
        non_draw_mask = (y_full != 1)
        X_home_away = df[self.home_away_features].values[non_draw_mask]
        y_home_away = y_full[non_draw_mask]  # 0=Home, 2=Away
        
        self.home_away_classifier.fit(X_home_away, y_home_away)
        
        logger.info("Cascade model training complete")
        
    def predict(self, df):
        """Make cascade predictions."""
        
        X_draw = df[self.draw_features].values
        X_home_away = df[self.home_away_features].values
        
        # Stage 1: Draw prediction probabilities
        draw_probs = self.draw_detector.predict_proba(X_draw)[:, 1]  # Probability of draw
        
        # Stage 2: Home vs Away predictions
        home_away_preds = self.home_away_classifier.predict(X_home_away)
        
        # Combine predictions using threshold
        final_predictions = np.zeros(len(df), dtype=int)
        
        for i in range(len(df)):
            if draw_probs[i] >= self.draw_threshold:
                final_predictions[i] = 1  # Draw
            else:
                final_predictions[i] = home_away_preds[i]  # Home (0) or Away (2)
        
        return final_predictions, draw_probs
    
    def get_feature_importance(self):
        """Get feature importance from both stages."""
        
        draw_importance = dict(zip(self.draw_features, self.draw_detector.feature_importances_))
        home_away_importance = dict(zip(self.home_away_features, self.home_away_classifier.feature_importances_))
        
        return {
            'draw_detector': draw_importance,
            'home_away_classifier': home_away_importance
        }

def evaluate_cascade_model(df, thresholds_to_test=[0.3, 0.4, 0.5, 0.6]):
    """Evaluate cascade model with different thresholds."""
    
    logger.info("🚀 Evaluating v2.11 Intelligent Cascade Model...")
    
    # Clean data
    df = df.dropna()
    logger.info(f"Dataset shape after cleaning: {df.shape}")
    
    # Prepare target
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_true = df['FullTimeResult'].map(target_mapping).values
    
    # Time series split for training
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    y_test = y_true[train_size:]
    
    results = []
    
    # Test different thresholds
    for threshold in thresholds_to_test:
        logger.info(f"Testing threshold: {threshold}")
        
        # Train cascade model
        cascade_model = IntelligentCascadeModel(draw_threshold=threshold, use_smote=True)
        cascade_model.fit(df_train)
        
        # Predict on test set
        y_pred, draw_probs = cascade_model.predict(df_test)
        
        # Calculate metrics
        global_accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Draw-specific metrics
        draw_mask_true = (y_test == 1)
        draw_mask_pred = (y_pred == 1)
        
        draw_recall = np.sum((y_test == 1) & (y_pred == 1)) / max(np.sum(y_test == 1), 1)
        draw_precision = np.sum((y_test == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
        draw_f1 = 2 * draw_precision * draw_recall / (draw_precision + draw_recall) if (draw_precision + draw_recall) > 0 else 0
        
        # Home/Away metrics
        non_draw_mask = (y_test != 1)
        if np.sum(non_draw_mask) > 0:
            y_test_ha = y_test[non_draw_mask]
            y_pred_ha = y_pred[non_draw_mask]
            home_away_accuracy = accuracy_score(y_test_ha, y_pred_ha)
        else:
            home_away_accuracy = 0
        
        # Class distribution
        pred_counts = np.bincount(y_pred, minlength=3)
        true_counts = np.bincount(y_test, minlength=3)
        
        result = {
            'threshold': threshold,
            'global_accuracy': global_accuracy,
            'f1_macro': f1_macro,
            'draw_recall': draw_recall,
            'draw_precision': draw_precision,
            'draw_f1': draw_f1,
            'home_away_accuracy': home_away_accuracy,
            'pred_distribution': pred_counts / len(y_pred),
            'true_distribution': true_counts / len(y_test),
            'draw_prob_mean': draw_probs.mean(),
            'draw_prob_std': draw_probs.std()
        }
        
        results.append(result)
    
    return results, cascade_model

def run_cascade_evaluation():
    """Run complete cascade evaluation."""
    
    # Load v2.11 dataset
    df = pd.read_csv('data/processed/v211_draw_features_2025_09_06.csv')
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Evaluate cascade model
    results, best_model = evaluate_cascade_model(df)
    
    # Display results
    print("\\n" + "="*90)
    print("🎯 v2.11 INTELLIGENT CASCADE MODEL EVALUATION")
    print("="*90)
    
    baseline_accuracy = 0.5685  # v2.9 baseline
    v24_cascade_accuracy = 0.530  # v2.4 cascade reference
    
    # Sort by F1-macro (balanced performance metric)
    results.sort(key=lambda x: x['f1_macro'], reverse=True)
    
    for i, result in enumerate(results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
        
        print(f"\\n{rank_emoji} Threshold {result['threshold']}:")
        print(f"   • Global Accuracy: {result['global_accuracy']:.4f}")
        print(f"   • F1-Macro: {result['f1_macro']:.4f}")
        print(f"   • vs v2.9 Baseline: {(result['global_accuracy'] - baseline_accuracy)*100:+.2f}pp")
        print(f"   • vs v2.4 Cascade: {(result['global_accuracy'] - v24_cascade_accuracy)*100:+.2f}pp")
        
        print(f"   • Draw Performance:")
        print(f"     - Recall: {result['draw_recall']:.3f}")
        print(f"     - Precision: {result['draw_precision']:.3f}")
        print(f"     - F1-Score: {result['draw_f1']:.3f}")
        
        print(f"   • Home/Away Accuracy: {result['home_away_accuracy']:.3f}")
        
        # Prediction distribution
        pred_dist = result['pred_distribution']
        true_dist = result['true_distribution']
        print(f"   • Prediction Distribution: H:{pred_dist[0]:.3f} D:{pred_dist[1]:.3f} A:{pred_dist[2]:.3f}")
        print(f"   • True Distribution:       H:{true_dist[0]:.3f} D:{true_dist[1]:.3f} A:{true_dist[2]:.3f}")
    
    # Best model analysis
    best_result = results[0]
    
    print(f"\\n🏆 OPTIMAL CASCADE CONFIGURATION:")
    print(f"   • Threshold: {best_result['threshold']}")
    print(f"   • Global Accuracy: {best_result['global_accuracy']:.4f}")
    print(f"   • Total Improvement: {(best_result['global_accuracy'] - 0.551)*100:+.2f}pp vs original v2.4")
    print(f"   • Draw F1 vs v2.4: {best_result['draw_f1']:.3f} (v2.4 had ~0.34 recall)")
    
    # Performance assessment
    if best_result['global_accuracy'] >= baseline_accuracy:
        print("\\n✅ CASCADE SUCCESS: Maintains accuracy while improving draw detection!")
    elif best_result['global_accuracy'] >= baseline_accuracy - 0.01:
        print("\\n⚡ BALANCED TRADE-OFF: Minor accuracy loss for major draw improvement")
    else:
        print("\\n⚠️  ACCURACY TRADE-OFF: Significant loss for draw gains")
    
    # Feature importance from best model
    if hasattr(best_model, 'get_feature_importance'):
        importance = best_model.get_feature_importance()
        
        print("\\n🔍 TOP DRAW DETECTION FEATURES:")
        draw_imp_sorted = sorted(importance['draw_detector'].items(), key=lambda x: x[1], reverse=True)
        for feature, imp in draw_imp_sorted[:5]:
            print(f"   • {feature}: {imp:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'evaluation/reports/v211_cascade_evaluation_{timestamp}.json', index=False)
    
    logger.info("✅ v2.11 cascade evaluation complete!")
    
    return results

if __name__ == "__main__":
    results = run_cascade_evaluation()