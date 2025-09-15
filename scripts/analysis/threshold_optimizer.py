#!/usr/bin/env python3
"""
THRESHOLD OPTIMIZER FOR EPL 2025-26
===================================

Analyzes prediction confidence and optimizes thresholds to balance precision/recall.
Uses the original v2.3 model to understand confidence patterns on EPL 2025-26 data.

Goal: Find optimal threshold to improve recall while maintaining reasonable precision.

Created: 2025-09-14
Purpose: Fine-tune precision/recall balance after identifying conservative behavior
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🎯 THRESHOLD OPTIMIZATION FOR EPL 2025-26")
    print("=" * 60)
    print("Goal: Balance precision/recall by optimizing decision thresholds")
    print("Using v2.3 model for cleaner analysis\n")
    
    # Load v2.3 model
    print("📊 Loading v2.3 production model...")
    model_path = "models/v23_retrained_2025_09_11_154613.joblib"
    
    if not Path(model_path).exists():
        print("❌ v2.3 model not found!")
        return
    
    model = joblib.load(model_path)
    print(f"✅ Loaded v2.3 model: {Path(model_path).name}")
    
    # Load EPL 2025-26 data
    print("📊 Loading EPL 2025-26 data...")
    data_path = "data/processed/premier_league_2025_26_all_matches_played.csv"
    
    if not Path(data_path).exists():
        print("❌ EPL 2025-26 data not found!")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded EPL 2025-26: {len(df)} matches")
    
    # Filter established teams only
    promoted_teams = ['Leeds United', 'Sunderland', 'Burnley']
    df_established = df[~df['HomeTeam'].isin(promoted_teams) & 
                       ~df['AwayTeam'].isin(promoted_teams)].copy()
    
    print(f"📊 Established teams: {len(df_established)} matches")
    
    # Prepare v2.3 features
    v23_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    # Check available features
    available_features = [f for f in v23_features if f in df_established.columns]
    missing_features = set(v23_features) - set(available_features)
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        # Add matchday_normalized if missing
        if 'matchday_normalized' in missing_features:
            df_established['matchday_normalized'] = 0.0  # All matches are from early season
            available_features.append('matchday_normalized')
            print("✅ Added matchday_normalized (set to 0.0 for early season)")
    
    X = df_established[available_features]
    y_true = (df_established['FullTimeResult'] == 'H').astype(int)  # 1 = Home Win, 0 = Not Home Win
    
    print(f"📊 Features: {len(available_features)}")
    print(f"📊 Target distribution: {y_true.mean():.1%} home wins")
    
    # Get prediction probabilities
    print("\n🎯 Analyzing v2.3 model confidence on EPL 2025-26...")
    
    try:
        probabilities = model.predict_proba(X)[:, 0]  # Probability of Home Win (class 0 in v2.3)
        default_predictions = model.predict(X)
        default_home_predictions = (default_predictions == 0).astype(int)  # Convert to binary
        
    except Exception as e:
        print(f"❌ Error getting probabilities: {e}")
        return
    
    print(f"✅ Generated probabilities for {len(probabilities)} matches")
    
    # Analyze current model behavior
    print("\n📈 CURRENT MODEL BEHAVIOR ANALYSIS")
    print("-" * 50)
    
    current_accuracy = accuracy_score(y_true, default_home_predictions)
    current_precision = precision_score(y_true, default_home_predictions)
    current_recall = recall_score(y_true, default_home_predictions)
    current_f1 = f1_score(y_true, default_home_predictions)
    
    print(f"Current performance on EPL 2025-26:")
    print(f"  Accuracy:  {current_accuracy:.3f}")
    print(f"  Precision: {current_precision:.3f}")  
    print(f"  Recall:    {current_recall:.3f}")
    print(f"  F1-Score:  {current_f1:.3f}")
    
    home_predictions_made = default_home_predictions.sum()
    print(f"  Home predictions made: {home_predictions_made}/{len(y_true)} ({home_predictions_made/len(y_true):.1%})")
    
    # Analyze confidence distribution
    print(f"\n📊 CONFIDENCE DISTRIBUTION")
    print("-" * 35)
    
    confidence_stats = {
        'mean': probabilities.mean(),
        'median': np.median(probabilities),
        'std': probabilities.std(),
        'min': probabilities.min(),
        'max': probabilities.max(),
        'q25': np.percentile(probabilities, 25),
        'q75': np.percentile(probabilities, 75)
    }
    
    print("Home Win Probability Statistics:")
    for stat, value in confidence_stats.items():
        print(f"  {stat:>8}: {value:.4f}")
    
    # Confidence level analysis
    high_conf_home = probabilities >= 0.7
    medium_conf_home = (probabilities >= 0.3) & (probabilities < 0.7)
    low_conf_home = probabilities < 0.3  # High confidence for Away/Draw
    
    print(f"\nConfidence Distribution:")
    print(f"  High Home Confidence (≥0.7): {high_conf_home.sum():2d} ({high_conf_home.mean():.1%})")
    print(f"  Medium Confidence (0.3-0.7):  {medium_conf_home.sum():2d} ({medium_conf_home.mean():.1%})")
    print(f"  High Away/Draw Conf (<0.3):   {low_conf_home.sum():2d} ({low_conf_home.mean():.1%})")
    
    # Threshold optimization
    print(f"\n🎯 THRESHOLD OPTIMIZATION")
    print("-" * 35)
    print("Testing thresholds from 0.2 to 0.8 for Home Win predictions...")
    
    threshold_results = []
    
    for threshold in np.arange(0.2, 0.85, 0.05):
        # Predict Home Win if probability >= threshold
        thresh_predictions = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, thresh_predictions)
        precision = precision_score(y_true, thresh_predictions, zero_division=0)
        recall = recall_score(y_true, thresh_predictions, zero_division=0)
        f1 = f1_score(y_true, thresh_predictions, zero_division=0)
        
        predictions_made = thresh_predictions.sum()
        prediction_rate = predictions_made / len(y_true)
        
        threshold_results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions_made': predictions_made,
            'prediction_rate': prediction_rate
        })
    
    results_df = pd.DataFrame(threshold_results)
    
    print("\nThreshold Performance Analysis:")
    print(results_df.round(3).to_string(index=False))
    
    # Find optimal thresholds
    best_f1_idx = results_df['f1_score'].idxmax()
    best_recall_idx = results_df['recall'].idxmax()
    best_balance_idx = (results_df['precision'] * results_df['recall']).idxmax()
    
    print(f"\n🏆 OPTIMAL THRESHOLDS:")
    print(f"Best F1-Score:    {results_df.loc[best_f1_idx, 'threshold']:.2f} "
          f"(F1: {results_df.loc[best_f1_idx, 'f1_score']:.3f}, "
          f"P: {results_df.loc[best_f1_idx, 'precision']:.3f}, "
          f"R: {results_df.loc[best_f1_idx, 'recall']:.3f})")
    
    print(f"Best Recall:      {results_df.loc[best_recall_idx, 'threshold']:.2f} "
          f"(Recall: {results_df.loc[best_recall_idx, 'recall']:.3f}, "
          f"P: {results_df.loc[best_recall_idx, 'precision']:.3f}, "
          f"F1: {results_df.loc[best_recall_idx, 'f1_score']:.3f})")
    
    print(f"Best Balance:     {results_df.loc[best_balance_idx, 'threshold']:.2f} "
          f"(P×R: {results_df.loc[best_balance_idx, 'precision'] * results_df.loc[best_balance_idx, 'recall']:.3f})")
    
    # Detailed analysis of best threshold
    best_threshold = results_df.loc[best_f1_idx, 'threshold']
    best_predictions = (probabilities >= best_threshold).astype(int)
    
    print(f"\n🔍 DETAILED ANALYSIS: Threshold {best_threshold:.2f}")
    print("-" * 45)
    
    # Confusion matrix analysis
    tp = ((best_predictions == 1) & (y_true == 1)).sum()
    fp = ((best_predictions == 1) & (y_true == 0)).sum()
    fn = ((best_predictions == 0) & (y_true == 1)).sum()
    tn = ((best_predictions == 0) & (y_true == 0)).sum()
    
    print(f"Confusion Matrix:")
    print(f"  True Positives (Correct Home):  {tp:2d}")
    print(f"  False Positives (Wrong Home):   {fp:2d}")
    print(f"  True Negatives (Correct N-Home): {tn:2d}")
    print(f"  False Negatives (Missed Home):  {fn:2d}")
    
    # Match analysis
    correct_home_matches = df_established[(best_predictions == 1) & (y_true == 1)]
    wrong_home_matches = df_established[(best_predictions == 1) & (y_true == 0)]
    missed_home_matches = df_established[(best_predictions == 0) & (y_true == 1)]
    
    if len(correct_home_matches) > 0:
        print(f"\n✅ CORRECTLY PREDICTED HOME WINS ({len(correct_home_matches)}):")
        for _, match in correct_home_matches.head(3).iterrows():
            prob = probabilities[df_established.index == match.name].iloc[0]
            print(f"  {match['HomeTeam']} vs {match['AwayTeam']} (prob: {prob:.3f})")
    
    if len(wrong_home_matches) > 0:
        print(f"\n❌ INCORRECTLY PREDICTED HOME WINS ({len(wrong_home_matches)}):")
        for _, match in wrong_home_matches.head(3).iterrows():
            prob = probabilities[df_established.index == match.name].iloc[0]
            actual = match['FullTimeResult']
            print(f"  {match['HomeTeam']} vs {match['AwayTeam']} -> {actual} (prob: {prob:.3f})")
    
    if len(missed_home_matches) > 0:
        print(f"\n😞 MISSED HOME WINS ({len(missed_home_matches)}):")
        for _, match in missed_home_matches.head(3).iterrows():
            prob = probabilities[df_established.index == match.name].iloc[0]
            print(f"  {match['HomeTeam']} vs {match['AwayTeam']} (prob: {prob:.3f})")
    
    # Feature analysis for different confidence levels
    print(f"\n📊 FEATURE ANALYSIS BY CONFIDENCE")
    print("-" * 40)
    
    if len(available_features) >= 5:
        top_features = ['elo_diff_normalized', 'market_entropy_norm', 
                       'form_diff_normalized', 'shots_diff_normalized']
        top_features = [f for f in top_features if f in available_features]
        
        print("Feature averages by confidence level:")
        for feature in top_features[:3]:  # Show top 3 to keep output manageable
            print(f"\n{feature}:")
            if high_conf_home.any():
                print(f"  High Home Conf: {X.loc[high_conf_home, feature].mean():.4f}")
            if medium_conf_home.any():
                print(f"  Medium Conf:    {X.loc[medium_conf_home, feature].mean():.4f}")
            if low_conf_home.any():
                print(f"  High Away Conf: {X.loc[low_conf_home, feature].mean():.4f}")
    
    # Save results
    results_dir = Path("results/threshold_optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_used': 'v2.3_production',
        'matches_analyzed': len(df_established),
        'current_performance': {
            'accuracy': current_accuracy,
            'precision': current_precision,
            'recall': current_recall,
            'f1_score': current_f1
        },
        'confidence_statistics': confidence_stats,
        'threshold_analysis': results_df.to_dict('records'),
        'optimal_thresholds': {
            'best_f1': {
                'threshold': results_df.loc[best_f1_idx, 'threshold'],
                'f1_score': results_df.loc[best_f1_idx, 'f1_score'],
                'precision': results_df.loc[best_f1_idx, 'precision'],
                'recall': results_df.loc[best_f1_idx, 'recall']
            },
            'best_recall': {
                'threshold': results_df.loc[best_recall_idx, 'threshold'],
                'recall': results_df.loc[best_recall_idx, 'recall'],
                'precision': results_df.loc[best_recall_idx, 'precision'],
                'f1_score': results_df.loc[best_recall_idx, 'f1_score']
            }
        },
        'recommendations': []
    }
    
    # Add recommendations
    improvement_potential = results_df.loc[best_f1_idx, 'f1_score'] - current_f1
    if improvement_potential > 0.05:
        results['recommendations'].append({
            'type': 'threshold_adjustment',
            'action': f"Adjust threshold to {best_threshold:.2f}",
            'expected_improvement': f"F1-Score: +{improvement_potential:.3f}",
            'trade_off': f"Prediction rate: {results_df.loc[best_f1_idx, 'prediction_rate']:.1%}"
        })
    
    results_file = results_dir / f"threshold_optimization_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    
    # Final recommendations
    print(f"\n🎉 THRESHOLD OPTIMIZATION COMPLETED!")
    print(f"📊 Current F1-Score: {current_f1:.3f}")
    print(f"🎯 Optimal F1-Score: {results_df.loc[best_f1_idx, 'f1_score']:.3f} (threshold: {best_threshold:.2f})")
    
    if improvement_potential > 0.05:
        print(f"✅ RECOMMENDATION: Use threshold {best_threshold:.2f}")
        print(f"   Expected improvement: +{improvement_potential:.3f} F1-Score")
    else:
        print(f"📊 Current threshold appears optimal for this dataset")

if __name__ == "__main__":
    main()