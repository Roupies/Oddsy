#!/usr/bin/env python3
"""
CONFIDENCE THRESHOLD ANALYZER
============================

Analyzes the Domain Adaptation model's prediction confidence patterns
to understand why it's overly conservative (100% precision, 12.5% recall).

Key Questions:
1. What confidence levels does the model output?
2. Which matches does it "ignore" (low confidence)?
3. What features characterize confident vs uncertain predictions?
4. How can we adjust thresholds to boost recall while maintaining precision?

Created: 2025-09-14
Purpose: Fine-tune precision/recall balance after concept drift resolution
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🔍 CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 60)
    print("Analyzing Domain Adaptation model's conservative behavior...")
    print("Goal: Understand 'ignored predictions' to boost recall\n")
    
    # Load domain adaptation model and data
    print("📊 Loading Domain Adaptation model and data...")
    
    model_path = "models/domain_adaptation/domain_adapted_model_20250914_191240.joblib"
    if not Path(model_path).exists():
        # Find the most recent domain adaptation model
        model_dir = Path("models/domain_adaptation/")
        model_files = list(model_dir.glob("domain_adapted_model_*.joblib"))
        if model_files:
            model_path = str(sorted(model_files)[-1])
        else:
            print("❌ No domain adaptation model found!")
            return
    
    model_data = joblib.load(model_path)
    model = model_data['model']  # Extract actual model from dictionary
    features_used = model_data.get('features', [])
    feature_transformers = model_data.get('feature_transformers', {})
    print(f"✅ Loaded model: {Path(model_path).name}")
    print(f"✅ Model features: {len(features_used)}")
    
    # Load EPL 2025-26 data
    data_path = "data/processed/premier_league_2025_26_all_matches_played.csv"
    if not Path(data_path).exists():
        print("❌ EPL 2025-26 processed data not found!")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded EPL 2025-26 data: {len(df)} matches")
    
    # Filter established teams only (same logic as domain adaptation)
    promoted_teams = ['Leeds United', 'Sunderland', 'Burnley']
    df_established = df[~df['HomeTeam'].isin(promoted_teams) & 
                       ~df['AwayTeam'].isin(promoted_teams)].copy()
    
    print(f"📊 Established teams only: {len(df_established)} matches")
    
    # Prepare features (same as domain adaptation model)
    feature_columns = [
        'elo_diff_normalized', 'form_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_xg_eff_10', 'shots_diff_normalized',
        'corners_diff_normalized', 'h2h_score', 'away_goals_sum_5',
        'uncertainty_amplified_v2', 'elo_reliability', 'elo_adjusted',
        'away_dominance_signal', 'home_fortress_breach', 'regime_change_signal',
        'correlation_breakdown'
    ]
    
    # Check which features exist
    existing_features = [col for col in feature_columns if col in df_established.columns]
    missing_features = set(feature_columns) - set(existing_features)
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print("Using available features only...")
        feature_columns = existing_features
    
    X = df_established[feature_columns]
    y_true = (df_established['FullTimeResult'] == 'H').astype(int)  # 1 = Home Win, 0 = Not Home Win
    
    print(f"📊 Features used: {len(feature_columns)}")
    print(f"📊 Target distribution: {y_true.mean():.1%} home wins")
    
    # Engineer missing drift-aware features (same as domain adaptation)
    print("\n🛠️ Engineering missing drift-aware features...")
    
    # uncertainty_amplified_v2
    if 'uncertainty_amplified_v2' not in df_established.columns:
        df_established['uncertainty_amplified_v2'] = (
            df_established['market_entropy_norm'] * 2.0 + 
            (1 - df_established['elo_diff_normalized'].abs()) * 1.5
        ).clip(0, 3)
    
    # elo_reliability 
    if 'elo_reliability' not in df_established.columns:
        df_established['elo_reliability'] = (df_established['elo_diff_normalized'] * 0.8 + 0.1).clip(0, 1)
    
    # elo_adjusted
    if 'elo_adjusted' not in df_established.columns:
        df_established['elo_adjusted'] = df_established['elo_diff_normalized'] * df_established['elo_reliability']
    
    # away_dominance_signal
    if 'away_dominance_signal' not in df_established.columns:
        df_established['away_dominance_signal'] = (
            (1 - df_established['elo_diff_normalized']) * 0.6 + 
            df_established['away_goals_sum_5'] / 15.0 * 0.4
        ).clip(0, 1)
    
    # home_fortress_breach
    if 'home_fortress_breach' not in df_established.columns:
        df_established['home_fortress_breach'] = (
            df_established['away_dominance_signal'] * 
            (1 - df_established['home_xg_eff_10']) * 2.0
        ).clip(0, 1)
    
    # regime_change_signal
    if 'regime_change_signal' not in df_established.columns:
        df_established['regime_change_signal'] = (
            abs(df_established['form_diff_normalized'] - 0.5) * 2.0 * 
            df_established['uncertainty_amplified_v2'] / 3.0
        ).clip(0, 1)
    
    # correlation_breakdown
    if 'correlation_breakdown' not in df_established.columns:
        df_established['correlation_breakdown'] = (
            abs(df_established['elo_diff_normalized'] - df_established['shots_diff_normalized']) * 2.0
        ).clip(0, 1)
    
    print("✅ Added missing drift-aware features")
    
    # For now, use basic features that exist in the data
    basic_features = [
        'elo_diff_normalized', 'form_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_xg_eff_10', 'shots_diff_normalized',
        'corners_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # Add engineered features
    engineered_features = [
        'uncertainty_amplified_v2', 'elo_reliability', 'elo_adjusted',
        'away_dominance_signal', 'home_fortress_breach', 'regime_change_signal',
        'correlation_breakdown'
    ]
    
    available_features = [f for f in basic_features + engineered_features 
                         if f in df_established.columns]
    
    print(f"✅ Using {len(available_features)} features for analysis")
    print("📊 Note: Using simplified features (not recalibrated) for confidence analysis")
    
    X = df_established[available_features]
    feature_columns = available_features
    
    # Get prediction probabilities
    print("\n🎯 Analyzing model confidence patterns...")
    
    try:
        # Get probabilities for Home Win (class 1)
        probabilities = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"❌ Error getting probabilities: {e}")
        return
    
    predictions = model.predict(X)
    
    # Analyze confidence distribution
    print("\n📈 CONFIDENCE DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    confidence_stats = {
        'mean': probabilities.mean(),
        'median': np.median(probabilities),
        'std': probabilities.std(),
        'min': probabilities.min(),
        'max': probabilities.max(),
        'q25': np.percentile(probabilities, 25),
        'q75': np.percentile(probabilities, 75)
    }
    
    print(f"Confidence Statistics (Home Win Probability):")
    for stat, value in confidence_stats.items():
        print(f"  {stat:>8}: {value:.4f}")
    
    # Identify current decision threshold
    # Find threshold that produces current predictions
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_match = 0
    
    for thresh in thresholds:
        thresh_predictions = (probabilities >= thresh).astype(int)
        matches = (thresh_predictions == predictions).sum()
        if matches > best_match:
            best_match = matches
            best_threshold = thresh
    
    print(f"\n🎯 Estimated decision threshold: {best_threshold:.3f}")
    print(f"   (Matches {best_match}/{len(predictions)} predictions)")
    
    # Analyze confident vs uncertain predictions
    print("\n🔍 CONFIDENT vs UNCERTAIN PREDICTIONS ANALYSIS")
    print("-" * 60)
    
    # Define confidence levels
    high_confidence = probabilities >= 0.7
    medium_confidence = (probabilities >= 0.4) & (probabilities < 0.7)
    low_confidence = probabilities < 0.4
    
    confidence_analysis = pd.DataFrame({
        'confidence_level': ['High (≥0.7)', 'Medium (0.4-0.7)', 'Low (<0.4)'],
        'count': [high_confidence.sum(), medium_confidence.sum(), low_confidence.sum()],
        'percentage': [high_confidence.mean()*100, medium_confidence.mean()*100, low_confidence.mean()*100],
        'avg_prob': [probabilities[high_confidence].mean() if high_confidence.any() else 0,
                    probabilities[medium_confidence].mean() if medium_confidence.any() else 0,
                    probabilities[low_confidence].mean() if low_confidence.any() else 0]
    })
    
    print("Confidence Level Distribution:")
    print(confidence_analysis.to_string(index=False, float_format='%.2f'))
    
    # Feature analysis for different confidence levels
    print("\n📊 FEATURE PATTERNS BY CONFIDENCE LEVEL")
    print("-" * 50)
    
    # Analyze key uncertainty features
    uncertainty_features = [f for f in ['uncertainty_amplified_v2', 'regime_change_signal', 
                           'market_entropy_norm', 'elo_reliability'] if f in feature_columns]
    
    if uncertainty_features:
        print("\nUncertainty Features Analysis:")
        for feature in uncertainty_features:
            print(f"\n{feature}:")
            if high_confidence.any():
                high_mean = X.loc[high_confidence, feature].mean()
                print(f"  High Confidence: {high_mean:.4f}")
            if medium_confidence.any():
                med_mean = X.loc[medium_confidence, feature].mean()
                print(f"  Medium Confidence: {med_mean:.4f}")
            if low_confidence.any():
                low_mean = X.loc[low_confidence, feature].mean()
                print(f"  Low Confidence: {low_mean:.4f}")
    
    # Threshold optimization analysis
    print("\n🎯 THRESHOLD OPTIMIZATION ANALYSIS")
    print("-" * 45)
    print("Testing different thresholds to balance precision/recall...")
    
    threshold_results = []
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        thresh_pred = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
        tp = ((thresh_pred == 1) & (y_true == 1)).sum()
        fp = ((thresh_pred == 1) & (y_true == 0)).sum()
        fn = ((thresh_pred == 0) & (y_true == 1)).sum()
        tn = ((thresh_pred == 0) & (y_true == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        predictions_made = (thresh_pred == 1).sum()
        
        threshold_results.append({
            'threshold': threshold,
            'predictions_made': predictions_made,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    
    print("\nThreshold Performance Analysis:")
    print(threshold_df.round(3).to_string(index=False))
    
    # Find optimal thresholds
    best_f1_idx = threshold_df['f1_score'].idxmax()
    best_precision_idx = threshold_df['precision'].idxmax()
    best_recall_idx = threshold_df['recall'].idxmax()
    
    print(f"\n🏆 OPTIMAL THRESHOLDS:")
    print(f"Best F1-Score: {threshold_df.loc[best_f1_idx, 'threshold']:.3f} "
          f"(F1: {threshold_df.loc[best_f1_idx, 'f1_score']:.3f})")
    print(f"Best Precision: {threshold_df.loc[best_precision_idx, 'threshold']:.3f} "
          f"(Precision: {threshold_df.loc[best_precision_idx, 'precision']:.3f})")
    print(f"Best Recall: {threshold_df.loc[best_recall_idx, 'threshold']:.3f} "
          f"(Recall: {threshold_df.loc[best_recall_idx, 'recall']:.3f})")
    
    # Analyze "ignored" predictions
    print(f"\n🤐 ANALYZING 'IGNORED' PREDICTIONS")
    print("-" * 40)
    
    # Using current estimated threshold
    confident_predictions = probabilities >= best_threshold
    ignored_predictions = probabilities < best_threshold
    
    print(f"Confident predictions: {confident_predictions.sum()} ({confident_predictions.mean():.1%})")
    print(f"Ignored predictions: {ignored_predictions.sum()} ({ignored_predictions.mean():.1%})")
    
    if ignored_predictions.any():
        print(f"\nWhat characterizes 'ignored' matches?")
        print(f"Average probability: {probabilities[ignored_predictions].mean():.3f}")
        
        # Analyze ignored matches features
        ignored_features = X[ignored_predictions].mean()
        confident_features = X[confident_predictions].mean() if confident_predictions.any() else pd.Series()
        
        if not confident_features.empty:
            print(f"\nFeature differences (Ignored vs Confident):")
            feature_diff = ignored_features - confident_features
            top_differences = feature_diff.abs().nlargest(5)
            
            for feature in top_differences.index:
                diff = feature_diff[feature]
                print(f"  {feature}: {diff:+.4f}")
    
    # Save results
    results_dir = Path("results/confidence_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'matches_analyzed': len(df_established),
        'confidence_statistics': confidence_stats,
        'estimated_threshold': best_threshold,
        'confidence_distribution': confidence_analysis.to_dict('records'),
        'threshold_optimization': threshold_df.to_dict('records'),
        'optimal_thresholds': {
            'best_f1': {
                'threshold': threshold_df.loc[best_f1_idx, 'threshold'],
                'f1_score': threshold_df.loc[best_f1_idx, 'f1_score'],
                'precision': threshold_df.loc[best_f1_idx, 'precision'],
                'recall': threshold_df.loc[best_f1_idx, 'recall']
            }
        },
        'recommendations': []
    }
    
    # Add recommendations
    if threshold_df.loc[best_f1_idx, 'f1_score'] > 0.22:  # Current F1 is 0.22
        results['recommendations'].append({
            'type': 'threshold_adjustment',
            'action': f"Lower threshold to {threshold_df.loc[best_f1_idx, 'threshold']:.3f}",
            'expected_improvement': f"F1-Score: {threshold_df.loc[best_f1_idx, 'f1_score']:.3f} "
                                   f"(+{threshold_df.loc[best_f1_idx, 'f1_score'] - 0.22:.3f})"
        })
    
    if ignored_predictions.mean() > 0.5:
        results['recommendations'].append({
            'type': 'feature_engineering',
            'action': "Reduce uncertainty amplification in features",
            'rationale': f"{ignored_predictions.mean():.1%} of predictions ignored due to low confidence"
        })
    
    results_file = results_dir / f"confidence_analysis_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    
    print(f"\n🎉 CONFIDENCE ANALYSIS COMPLETED!")
    print(f"📊 Key Insight: Model ignores {ignored_predictions.mean():.1%} of predictions")
    print(f"🎯 Recommendation: Test threshold {threshold_df.loc[best_f1_idx, 'threshold']:.3f} "
          f"for F1-Score {threshold_df.loc[best_f1_idx, 'f1_score']:.3f}")

if __name__ == "__main__":
    main()