#!/usr/bin/env python3
"""
📊 Feature Correlation Analysis with Draw Outcomes
=================================================

Analyze correlation between enhanced features and draw outcomes
to validate their potential for draw detection improvement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_correlations():
    """Analyze correlation between features and draw outcomes"""
    print("📊 FEATURE CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Load enhanced dataset
    data_path = "data/processed/v16_specialized_features_enhanced.csv"
    data = pd.read_csv(data_path)
    
    # Filter matches with results and create draw target
    data_with_results = data[data['FullTimeResult'].notna()].copy()
    data_with_results['is_draw'] = (data_with_results['FullTimeResult'] == 'D').astype(int)
    
    print(f"📈 Dataset: {len(data_with_results)} matches with results")
    print(f"📊 Draws: {data_with_results['is_draw'].sum()} ({data_with_results['is_draw'].mean():.1%})")
    
    # All features to analyze
    all_features = [
        # Current production features
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized', 
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score', 
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5',
        
        # Enhanced features
        'elo_variance_recent', 'team_parity_score', 'market_odds_spread', 
        'low_scoring_potential', 'is_promoted'
    ]
    
    # Calculate correlations with draw outcome
    correlations = []
    for feature in all_features:
        if feature in data_with_results.columns:
            # Handle missing values
            feature_data = data_with_results[feature].fillna(data_with_results[feature].median())
            draw_data = data_with_results['is_draw']
            
            # Calculate Pearson correlation
            corr, p_value = pearsonr(feature_data, draw_data)
            
            correlations.append({
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'feature_type': 'enhanced' if feature in ['elo_variance_recent', 'team_parity_score', 
                                                         'market_odds_spread', 'low_scoring_potential', 
                                                         'is_promoted'] else 'current'
            })
        else:
            print(f"⚠️ Missing feature: {feature}")
    
    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    print(f"\n🎯 FEATURE CORRELATIONS WITH DRAWS")
    print("=" * 60)
    print(f"{'Feature':<25} {'Corr':<8} {'P-val':<8} {'Sig':<5} {'Type':<10}")
    print("-" * 60)
    
    for _, row in corr_df.head(15).iterrows():
        sig_marker = "✅" if row['significant'] else "❌"
        print(f"{row['feature']:<25} {row['correlation']:>6.3f} {row['p_value']:>6.3f} {sig_marker:<5} {row['feature_type']:<10}")
    
    # Enhanced features analysis
    enhanced_features = corr_df[corr_df['feature_type'] == 'enhanced'].copy()
    print(f"\n🔬 ENHANCED FEATURES ANALYSIS")
    print("=" * 40)
    
    for _, row in enhanced_features.iterrows():
        performance = "🎯 Strong" if row['abs_correlation'] > 0.1 else "📊 Moderate" if row['abs_correlation'] > 0.05 else "⚠️ Weak"
        print(f"{row['feature']:<25} {row['correlation']:>6.3f} {performance}")
    
    # Identify best features for each category
    print(f"\n🏆 TOP FEATURES BY CATEGORY")
    print("=" * 40)
    
    # Top current features (>0.05 correlation)
    top_current = corr_df[(corr_df['feature_type'] == 'current') & 
                         (corr_df['abs_correlation'] > 0.05)]['feature'].tolist()
    print(f"🔷 Top Current Features: {top_current[:5]}")
    
    # Top enhanced features (>0.05 correlation)
    top_enhanced = corr_df[(corr_df['feature_type'] == 'enhanced') & 
                          (corr_df['abs_correlation'] > 0.05)]['feature'].tolist()
    print(f"🔶 Top Enhanced Features: {top_enhanced}")
    
    # Create validated enhanced feature set
    validated_enhanced = corr_df[corr_df['abs_correlation'] > 0.05]['feature'].tolist()
    print(f"\n✅ VALIDATED FEATURES (>0.05 correlation): {len(validated_enhanced)}")
    print(f"   {validated_enhanced}")
    
    # Feature recommendations for each set
    print(f"\n📋 FEATURE SET RECOMMENDATIONS")
    print("=" * 40)
    
    # Set 2: Draw-Specialized (Stage 1 enhanced + Stage 2 power)
    stage1_features = ['market_entropy_norm'] + [f for f in top_enhanced if f != 'is_promoted']
    stage2_features = ['elo_diff_normalized', 'shots_diff_normalized', 'form_diff_normalized',
                      'h2h_score', 'home_xg_eff_10', 'away_xg_eff_10']
    
    print(f"🎯 Set 2 - Stage 1 (Draw): {stage1_features}")
    print(f"🎯 Set 2 - Stage 2 (H/A): {stage2_features}")
    
    # Set 4: Enhanced Hybrid (best current + best enhanced)
    hybrid_features = top_current[:6] + top_enhanced[:3]
    print(f"🔶 Set 4 - Hybrid: {hybrid_features}")
    
    # Set 5: Validated Enhanced (all good features)
    print(f"✅ Set 5 - Validated: {validated_enhanced[:12]}") # Limit to 12 for stability
    
    return {
        'correlations': corr_df,
        'top_current': top_current,
        'top_enhanced': top_enhanced, 
        'validated_features': validated_enhanced,
        'stage1_features': stage1_features,
        'stage2_features': stage2_features,
        'hybrid_features': hybrid_features
    }

if __name__ == "__main__":
    results = analyze_feature_correlations()
    print(f"\n🎉 CORRELATION ANALYSIS COMPLETE!")
    print(f"📊 Enhanced features validated for cascade optimization")