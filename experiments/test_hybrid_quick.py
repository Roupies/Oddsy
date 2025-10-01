#!/usr/bin/env python3
"""
🎯 Quick Hybrid Test - Enhanced vs Baseline vs Hybrid
=====================================================
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Import from our main script
exec(open('generate_j6_natural_enhanced.py').read().replace('if __name__ == "__main__":', 'if False:'))

def quick_hybrid_test():
    """Quick test of Enhanced vs Baseline vs Simple Hybrid"""
    print("🚀 QUICK HYBRID TEST - Enhanced vs Baseline vs Hybrid")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('data/processed/v_auto_update_20250922_093416.csv')
    epl_2025 = data[data['Season'] == '2025-2026'].copy()
    epl_with_results = epl_2025[epl_2025['FullTimeResult'].notna()]
    
    print(f"✅ Test data: {len(epl_with_results)} matches")
    
    # Load models
    real_odds_data = load_real_odds_data()
    enhanced_cascade, enhanced_features = create_natural_enhanced_cascade()
    
    try:
        baseline_model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Baseline Champion loaded")
    except:
        print("❌ Baseline Champion not found")
        return
    
    test_matches = epl_with_results.tail(20)  # Smaller test set
    
    enhanced_correct = 0
    baseline_correct = 0
    hybrid_correct = 0
    total = 0
    
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
    ]
    
    print(f"\n📊 Testing on {len(test_matches)} matches...")
    
    for idx, match in test_matches.iterrows():
        home_team = match['HomeTeam']  
        away_team = match['AwayTeam']
        actual_result = match['FullTimeResult']
        match_date = match['Date']
        
        try:
            # Calculate features
            features_dict = calculate_enhanced_features(
                data, home_team, away_team, match_date, 
                j6_odds=None, real_odds_data=real_odds_data
            )
            
            # Enhanced model prediction
            X_enhanced = pd.DataFrame([features_dict])[enhanced_features]
            enhanced_pred = enhanced_cascade.predict(X_enhanced)[0]
            enhanced_class = ['H', 'D', 'A'][enhanced_pred]
            enhanced_proba = enhanced_cascade.predict_proba(X_enhanced)[0]
            
            # Baseline model prediction
            baseline_dict = {}
            for feat in baseline_features:
                if feat in features_dict:
                    baseline_dict[feat] = features_dict[feat]
                elif feat == 'market_entropy_norm' and 'market_entropy_historical' in features_dict:
                    baseline_dict[feat] = features_dict['market_entropy_historical']
                else:
                    baseline_dict[feat] = 0.5
            
            X_baseline = pd.DataFrame([baseline_dict])[baseline_features]
            baseline_pred = baseline_model.predict(X_baseline)[0]
            baseline_class = ['H', 'D', 'A'][baseline_pred]
            
            # Simple hybrid: Enhanced if draw confidence > 0.4, else Baseline
            draw_confidence = enhanced_proba[1]
            if draw_confidence > 0.4:
                hybrid_class = enhanced_class
            else:
                hybrid_class = baseline_class
            
            # Evaluate
            enhanced_correct += (enhanced_class == actual_result)
            baseline_correct += (baseline_class == actual_result)
            hybrid_correct += (hybrid_class == actual_result)
            total += 1
            
            print(f"{home_team} vs {away_team}: Actual={actual_result}")
            print(f"  Enhanced: {enhanced_class} (D={draw_confidence:.2f})")
            print(f"  Baseline: {baseline_class}")
            print(f"  Hybrid:   {hybrid_class}")
            print()
            
        except Exception as e:
            print(f'⚠️ Error: {str(e)}')
    
    # Results
    if total > 0:
        enhanced_acc = enhanced_correct / total
        baseline_acc = baseline_correct / total
        hybrid_acc = hybrid_correct / total
        
        print(f"🎯 QUICK TEST RESULTS ({total} matches):")
        print(f"Enhanced accuracy: {enhanced_acc:.1%} ({enhanced_correct}/{total})")
        print(f"Baseline accuracy: {baseline_acc:.1%} ({baseline_correct}/{total})")
        print(f"Hybrid accuracy:   {hybrid_acc:.1%} ({hybrid_correct}/{total})")
        
        print(f"\n📈 Improvements:")
        print(f"Hybrid vs Enhanced: {hybrid_acc - enhanced_acc:.1%}")
        print(f"Hybrid vs Baseline: {hybrid_acc - baseline_acc:.1%}")
        
        if hybrid_acc > max(enhanced_acc, baseline_acc):
            print("✅ Hybrid WINS!")
        elif baseline_acc > enhanced_acc:
            print("✅ Baseline WINS!")
        else:
            print("✅ Enhanced WINS!")

if __name__ == "__main__":
    quick_hybrid_test()