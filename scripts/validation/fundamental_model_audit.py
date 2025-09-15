#!/usr/bin/env python3
"""
Fundamental Model Audit - Question Everything
Deep investigation into the core assumptions and validity of the v2.4 model.

Key Questions:
1. Are we testing on real or simulated data?
2. Does cascade architecture make sense if it never predicts draws?
3. What is the true baseline performance on REAL data only?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def analyze_data_reality():
    """Determine what data is real vs simulated/projected."""
    print("🔍 ANALYZING DATA REALITY")
    print("=" * 50)
    
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"📊 Dataset Overview:")
    print(f"Total matches: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Analyze by season
    season_analysis = df.groupby('Season').agg({
        'Date': ['min', 'max', 'count']
    }).round(2)
    season_analysis.columns = ['Start', 'End', 'Matches']
    
    print(f"\n📅 Season Analysis:")
    print(season_analysis)
    
    # Current date analysis
    from datetime import datetime
    today = datetime.now()
    print(f"\n⏰ Today's date: {today.strftime('%Y-%m-%d')}")
    
    # Classify data as real or projected
    real_data = df[df['Date'] <= today]
    future_data = df[df['Date'] > today]
    
    print(f"\n🔍 Reality Check:")
    print(f"Real/Historical data: {len(real_data)} matches (up to {real_data['Date'].max().strftime('%Y-%m-%d')})")
    if len(future_data) > 0:
        print(f"Future/Simulated data: {len(future_data)} matches (from {future_data['Date'].min().strftime('%Y-%m-%d')} onwards)")
    else:
        print("Future/Simulated data: 0 matches")
    
    # Check our test split
    test_cutoff = pd.to_datetime('2023-05-01')
    test_data = df[df['Date'] >= test_cutoff]
    test_real = test_data[test_data['Date'] <= today]
    test_future = test_data[test_data['Date'] > today]
    
    print(f"\n⚠️  TEST SET ANALYSIS:")
    print(f"Test set total: {len(test_data)} matches")
    print(f"Test set real: {len(test_real)} matches ({len(test_real)/len(test_data)*100:.1f}%)")
    print(f"Test set simulated: {len(test_future)} matches ({len(test_future)/len(test_data)*100:.1f}%)")
    
    if len(test_future) > 0:
        print(f"🚨 CRITICAL ISSUE: Testing on {len(test_future)} SIMULATED matches!")
        print("This invalidates performance claims on future data.")
    
    return {
        'total_matches': len(df),
        'real_matches': len(real_data),
        'future_matches': len(future_data),
        'test_real_matches': len(test_real),
        'test_future_matches': len(test_future),
        'last_real_date': real_data['Date'].max(),
        'real_data': real_data,
        'test_real_data': test_real
    }

def test_on_real_data_only(real_analysis):
    """Test performance on REAL data only."""
    print(f"\n🎯 TESTING ON REAL DATA ONLY")
    print("=" * 50)
    
    df = real_analysis['real_data']
    
    print(f"Real data: {len(df)} matches up to {real_analysis['last_real_date'].strftime('%Y-%m-%d')}")
    
    # Find a reasonable split within real data
    # Use 2023-05-01 as cutoff if we have real data after that, otherwise use 80/20 split
    cutoff_date = pd.to_datetime('2023-05-01')
    
    if df['Date'].max() > cutoff_date:
        train_df = df[df['Date'] < cutoff_date].copy()
        test_df = df[df['Date'] >= cutoff_date].copy()
        split_type = "Temporal (2023-05-01 cutoff)"
    else:
        # If all data is before 2023, use 80/20 split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        split_type = "80/20 chronological"
    
    print(f"\n📊 Real Data Split ({split_type}):")
    print(f"Train: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test: {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    # Check if we have enough test data
    if len(test_df) < 50:
        print(f"⚠️  WARNING: Only {len(test_df)} test matches - results may be unreliable")
    
    # Features
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # Test different approaches
    results = {}
    
    # 1. Simple RandomForest (no cascade)
    print(f"\n1️⃣ SIMPLE RANDOMFOREST (No Cascade):")
    X_train_simple = train_df[features].fillna(train_df[features].median())
    y_train_simple = train_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    X_test_simple = test_df[features].fillna(train_df[features].median())
    y_test_simple = test_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    simple_model = RandomForestClassifier(n_estimators=100, random_state=42)
    simple_model.fit(X_train_simple, y_train_simple)
    y_pred_simple = simple_model.predict(X_test_simple)
    accuracy_simple = accuracy_score(y_test_simple, y_pred_simple)
    
    results['simple'] = accuracy_simple
    print(f"Accuracy: {accuracy_simple:.1%}")
    
    # Map back to H/D/A for analysis
    pred_simple_mapped = {0: 'H', 1: 'D', 2: 'A'}
    y_pred_simple_hda = [pred_simple_mapped[p] for p in y_pred_simple]
    
    print("Class distribution:")
    unique, counts = np.unique(y_pred_simple_hda, return_counts=True)
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count} ({count/len(y_pred_simple_hda)*100:.1f}%)")
    
    # 2. Cascade with reasonable threshold
    print(f"\n2️⃣ CASCADE WITH BALANCED THRESHOLD:")
    
    # Stage 1: Draw detection
    y_train_draw = (train_df['FullTimeResult'] == 'D').astype(int)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_simple, y_train_draw)
    
    stage1_model = RandomForestClassifier(n_estimators=100, random_state=42)
    stage1_model.fit(X_train_balanced, y_train_balanced)
    stage1_proba = stage1_model.predict_proba(X_test_simple)[:, 1]
    
    # Try different thresholds
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        draw_mask = stage1_proba >= threshold
        y_pred_cascade = np.full(len(X_test_simple), 'D', dtype=object)
        
        # Stage 2 for non-draws
        if (~draw_mask).sum() > 0:
            train_non_draw = train_df[train_df['FullTimeResult'] != 'D'].copy()
            X_train_s2 = train_non_draw[features].fillna(train_non_draw[features].median())
            y_train_s2 = train_non_draw['FullTimeResult'].map({'H': 0, 'A': 1})
            
            stage2_model = RandomForestClassifier(n_estimators=100, random_state=42)
            stage2_model.fit(X_train_s2, y_train_s2)
            stage2_pred = stage2_model.predict(X_test_simple[~draw_mask])
            
            y_pred_cascade[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
        
        accuracy_cascade = accuracy_score(test_df['FullTimeResult'], y_pred_cascade)
        draw_pct = draw_mask.mean()
        
        print(f"  Threshold {threshold}: {accuracy_cascade:.1%} accuracy, {draw_pct:.1%} draws")
        
        if accuracy_cascade > best_accuracy:
            best_accuracy = accuracy_cascade
            best_threshold = threshold
    
    results['cascade'] = best_accuracy
    print(f"Best cascade: {best_accuracy:.1%} at threshold {best_threshold}")
    
    # 3. Test current "optimal" threshold 0.8
    print(f"\n3️⃣ CURRENT MODEL (Threshold 0.8):")
    draw_mask_08 = stage1_proba >= 0.8
    y_pred_08 = np.full(len(X_test_simple), 'D', dtype=object)
    
    if (~draw_mask_08).sum() > 0:
        stage2_pred_08 = stage2_model.predict(X_test_simple[~draw_mask_08])
        y_pred_08[~draw_mask_08] = np.where(stage2_pred_08 == 0, 'H', 'A')
    
    accuracy_08 = accuracy_score(test_df['FullTimeResult'], y_pred_08)
    results['current'] = accuracy_08
    
    print(f"Accuracy: {accuracy_08:.1%}")
    print(f"Draw predictions: {draw_mask_08.sum()}/{len(draw_mask_08)} ({draw_mask_08.mean():.1%})")
    
    return results

def analyze_cascade_logic():
    """Analyze if cascade architecture makes sense."""
    print(f"\n🤔 ANALYZING CASCADE ARCHITECTURE LOGIC")
    print("=" * 50)
    
    print("Questions to address:")
    print("1. Does 2-stage cascade outperform simple 3-class classification?")
    print("2. If we never predict draws, why have a draw detector?")
    print("3. Is the complexity justified by performance gains?")
    
    # Load data for analysis
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Analyze draw prediction rates at different thresholds
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    print(f"\n📊 True distribution in test set:")
    test_dist = test_df['FullTimeResult'].value_counts(normalize=True)
    for result, pct in test_dist.items():
        print(f"  {result}: {pct:.1%}")
    
    print(f"\nIf we predict 0% draws but true rate is {test_dist.get('D', 0):.1%}:")
    print(f"  Maximum possible accuracy ≈ {(test_dist.get('H', 0) + test_dist.get('A', 0)):.1%}")
    print(f"  We're leaving {test_dist.get('D', 0):.1%} on the table by never predicting draws")
    
    return True

def main():
    """Main fundamental audit."""
    print("🔍 FUNDAMENTAL MODEL AUDIT")
    print("=" * 60)
    print("Questioning core assumptions about data, architecture, and performance")
    print()
    
    # 1. Analyze data reality
    real_analysis = analyze_data_reality()
    
    # 2. Test on real data only
    if real_analysis['real_matches'] > 100:
        real_results = test_on_real_data_only(real_analysis)
        
        print(f"\n📊 PERFORMANCE COMPARISON ON REAL DATA:")
        print(f"Simple RandomForest: {real_results['simple']:.1%}")
        print(f"Best Cascade: {real_results['cascade']:.1%}")
        print(f"Current Model (0.8): {real_results['current']:.1%}")
        
        if real_results['simple'] > real_results['cascade']:
            print(f"🚨 CRITICAL: Simple model beats cascade!")
        
        if real_results['current'] < real_results['cascade']:
            print(f"🚨 CRITICAL: Current threshold is suboptimal!")
    
    # 3. Analyze cascade logic
    analyze_cascade_logic()
    
    # Final assessment
    print(f"\n💡 FUNDAMENTAL QUESTIONS RAISED:")
    print("1. Are we testing on simulated future data? (VALIDITY)")
    print("2. Does cascade justify its complexity? (ARCHITECTURE)")
    print("3. Should we predict draws at all? (STRATEGY)")
    print("4. What is the true performance on real data? (PERFORMANCE)")
    
    print(f"\n🎯 NEXT STEPS:")
    print("- Retrain and test on real data only")
    print("- Compare simple vs cascade approaches honestly")  
    print("- Decide on draw prediction strategy")
    print("- Establish true baseline performance")

if __name__ == "__main__":
    main()