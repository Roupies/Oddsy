#!/usr/bin/env python3
"""
ROI Loss Diagnostic - Analyze why the model is losing money
Focus: Calibration issues, overconfidence, market efficiency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_roi_failure():
    """Comprehensive analysis of why ROI simulation failed."""
    
    logger.info("🔍 Starting ROI failure diagnostic...")
    
    # Load the betting history from our simulation
    try:
        with open('evaluation/reports/real_odds_roi_simulation_2025_09_06_185942.json', 'r') as f:
            roi_results = json.load(f)
        
        print("="*80)
        print("🔍 ROI FAILURE DIAGNOSTIC")
        print("="*80)
        
        print(f"\n💸 LOSS BREAKDOWN:")
        print(f"   • Total Loss: £{roi_results['total_profit']:.0f}")
        print(f"   • Hit Rate: {roi_results['hit_rate']:.1%}")
        print(f"   • Average Edge: {roi_results['avg_edge']:.1%}")
        print(f"   • Average Odds: {roi_results['avg_odds']:.2f}")
        
        # Problem analysis
        avg_edge = roi_results['avg_edge']
        hit_rate = roi_results['hit_rate'] 
        avg_odds = roi_results['avg_odds']
        
        print(f"\n🎯 PROBLEM ANALYSIS:")
        
        # Problem 1: Edge vs Hit Rate mismatch
        expected_win_prob = 1 / avg_odds
        print(f"   1. CALIBRATION ISSUE:")
        print(f"      • Model thinks edge: {avg_edge:.1%}")
        print(f"      • Actual hit rate: {hit_rate:.1%}") 
        print(f"      • Market implied: {expected_win_prob:.1%}")
        print(f"      • Model overconfident by: {(avg_edge - (hit_rate - expected_win_prob))*100:.1f}pp")
        
        # Problem 2: Kelly sizing with bad calibration
        theoretical_kelly = avg_edge / (avg_odds - 1) if avg_odds > 1 else 0
        print(f"   2. KELLY SIZING ISSUE:")
        print(f"      • Theoretical Kelly: {theoretical_kelly:.1%}")
        print(f"      • With overconfidence → bet sizes too large")
        
        # Problem 3: Selection bias
        print(f"   3. SELECTION BIAS:")
        print(f"      • Only betting when model very confident")
        print(f"      • High odds bets (avg {avg_odds:.2f}) are hardest to predict")
        print(f"      • Missing easy, low-odds value bets")
        
    except FileNotFoundError:
        logger.warning("ROI results file not found, running diagnostic on model directly")
    
    return analyze_model_calibration()

def analyze_model_calibration():
    """Analyze model probability calibration issues."""
    
    logger.info("Analyzing model calibration...")
    
    # Recreate the model and data
    import glob
    
    # Load odds data
    odds_files = glob.glob('/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/*.csv')
    all_odds = []
    for file in sorted(odds_files):
        season_data = pd.read_csv(file)
        season = file.split('_')[-2] + '_' + file.split('_')[-1].replace('.csv', '')
        season_data['Season'] = season
        all_odds.append(season_data)
    
    odds_df = pd.concat(all_odds, ignore_index=True)
    odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Load features
    features_df = pd.read_csv('data/processed/v13_xg_corrected_features_latest.csv')
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    # Merge
    merged_df = features_df.merge(
        odds_df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']],
        on=['Date', 'HomeTeam', 'AwayTeam'],
        how='inner'
    )
    
    # Prepare data
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    available_features = [f for f in features if f in merged_df.columns]
    df_clean = merged_df.dropna(subset=available_features + ['B365H', 'B365D', 'B365A'])
    
    # Split data
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean[:split_idx]
    df_test = df_clean[split_idx:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    X_train = df_train[available_features]
    y_train = df_train['FullTimeResult'].map(target_mapping)
    
    model.fit(X_train, y_train)
    
    # Predict on test set
    X_test = df_test[available_features]
    y_test = df_test['FullTimeResult'].map(target_mapping)
    model_probs = model.predict_proba(X_test)
    
    print(f"\n📊 CALIBRATION ANALYSIS:")
    print(f"   • Test set size: {len(df_test)} matches")
    
    # Analyze calibration for each class
    class_names = ['Home', 'Draw', 'Away']
    
    for class_idx, class_name in enumerate(class_names):
        y_binary = (y_test == class_idx).astype(int)
        prob_pred = model_probs[:, class_idx]
        
        # Basic calibration metrics
        prob_bins = np.arange(0, 1.1, 0.1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(prob_bins)-1):
            mask = (prob_pred >= prob_bins[i]) & (prob_pred < prob_bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((prob_bins[i] + prob_bins[i+1]) / 2)
                bin_accuracies.append(y_binary[mask].mean())
                bin_counts.append(mask.sum())
        
        # Calculate ECE (Expected Calibration Error)
        ece = 0
        total_samples = len(y_binary)
        for j in range(len(bin_centers)):
            weight = bin_counts[j] / total_samples
            ece += weight * abs(bin_centers[j] - bin_accuracies[j])
        
        print(f"   • {class_name} ECE: {ece:.3f}")
        
        # Show worst calibrated bins
        if len(bin_centers) > 0:
            worst_bin_idx = np.argmax([abs(bin_centers[j] - bin_accuracies[j]) for j in range(len(bin_centers))])
            worst_predicted = bin_centers[worst_bin_idx]
            worst_actual = bin_accuracies[worst_bin_idx]
            print(f"     - Worst bin: {worst_predicted:.1f} predicted, {worst_actual:.1f} actual (off by {abs(worst_predicted-worst_actual):.2f})")
    
    # Market efficiency comparison
    print(f"\n🎯 MARKET EFFICIENCY:")
    
    total_value_bets = 0
    profitable_value_bets = 0
    
    for i, (_, row) in enumerate(df_test.reset_index().iterrows()):
        if i >= len(model_probs):
            break
            
        # Model vs market comparison
        model_prob_home, model_prob_draw, model_prob_away = model_probs[i]
        
        odds_home = row['B365H']
        odds_draw = row['B365D']
        odds_away = row['B365A']
        
        if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
            continue
            
        # Market probabilities
        prob_home_market = 1 / odds_home
        prob_draw_market = 1 / odds_draw
        prob_away_market = 1 / odds_away
        
        # Normalize
        total_market = prob_home_market + prob_draw_market + prob_away_market
        prob_home_market /= total_market
        prob_draw_market /= total_market
        prob_away_market /= total_market
        
        # Check edges
        edges = [
            model_prob_home - prob_home_market,
            model_prob_draw - prob_draw_market, 
            model_prob_away - prob_away_market
        ]
        
        outcomes = ['H', 'D', 'A']
        actual = row['FullTimeResult']
        
        for j, edge in enumerate(edges):
            if edge > 0.05:  # 5% edge
                total_value_bets += 1
                if actual == outcomes[j]:
                    profitable_value_bets += 1
    
    value_bet_accuracy = profitable_value_bets / total_value_bets if total_value_bets > 0 else 0
    
    print(f"   • Value bets found: {total_value_bets}")
    print(f"   • Value bets won: {profitable_value_bets}")
    print(f"   • Value bet accuracy: {value_bet_accuracy:.1%}")
    print(f"   • Expected from edges: ~55-60% (if well calibrated)")
    
    # Recommendations
    print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
    print(f"   1. CALIBRATE MODEL:")
    print(f"      • Use Platt scaling or Isotonic regression")
    print(f"      • ECE too high → probabilities not trustworthy")
    
    print(f"   2. LOWER EDGE THRESHOLD:")
    print(f"      • Try 2-3% minimum edge instead of 5%")
    print(f"      • Focus on volume with smaller edges")
    
    print(f"   3. BET ON FAVORITES:")
    print(f"      • High-odds bets are hardest to predict")
    print(f"      • Try betting on Home favorites with small edges")
    
    print(f"   4. ENSEMBLE APPROACH:")
    print(f"      • Combine with market probabilities")
    print(f"      • Use market as prior, model as adjustment")
    
    return {
        'ece_home': ece if 'ece' in locals() else None,
        'value_bet_accuracy': value_bet_accuracy,
        'total_value_bets': total_value_bets
    }

if __name__ == "__main__":
    results = analyze_roi_failure()