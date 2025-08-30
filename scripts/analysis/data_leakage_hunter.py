import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def hunt_data_leakage():
    """
    CRITICAL INVESTIGATION: Find and fix data leakage causing 99.7% training accuracy
    
    Investigation Plan:
    1. Test each feature individually for leakage
    2. Check temporal integrity of rolling features
    3. Validate feature calculation methodology
    4. Compare with match results to identify future information
    5. Create leakage-free feature set
    """
    
    logger = setup_logging()
    logger.info("=== 🔍 DATA LEAKAGE HUNTER - Critical Investigation ===")
    
    # Load dataset
    df_file = "data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv"
    logger.info(f"Loading dataset: {df_file}")
    
    df = pd.read_csv(df_file)
    logger.info(f"Dataset: {df.shape}")
    
    # Features to investigate
    v12_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized'
    ]
    
    # Prepare target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Investigating features: {v12_features}")
    
    # =========================
    # 1. INDIVIDUAL FEATURE LEAKAGE TEST
    # =========================
    logger.info("\n🔍 1. INDIVIDUAL FEATURE LEAKAGE TEST")
    
    rf_leak_test = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
    
    feature_leakage_scores = {}
    
    for feature in v12_features:
        X_single = df[[feature]]
        
        # Training accuracy (potential leakage indicator)
        rf_leak_test.fit(X_single, y)
        train_acc = rf_leak_test.score(X_single, y)
        
        # Cross-validation accuracy (realistic performance)
        cv_scores = cross_val_score(rf_leak_test, X_single, y, 
                                  cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
        cv_acc = cv_scores.mean()
        
        # Leakage indicator: huge gap between train and CV
        leakage_gap = train_acc - cv_acc
        
        feature_leakage_scores[feature] = {
            'train_acc': train_acc,
            'cv_acc': cv_acc,
            'leakage_gap': leakage_gap
        }
        
        status = ""
        if train_acc > 0.95:
            status += "🚨 PERFECT_TRAIN "
        if leakage_gap > 0.3:
            status += "🚨 HUGE_GAP "
        if cv_acc < 0.4:
            status += "📉 POOR_CV "
        if not status:
            status = "✅ OK"
        
        logger.info(f"  {feature}:")
        logger.info(f"    Train: {train_acc:.3f}, CV: {cv_acc:.3f}, Gap: {leakage_gap:.3f} {status}")
    
    # Identify worst leakage offenders
    leakage_suspects = []
    for feature, scores in feature_leakage_scores.items():
        if scores['train_acc'] > 0.9 or scores['leakage_gap'] > 0.3:
            leakage_suspects.append(feature)
    
    logger.info(f"\n  🚨 LEAKAGE SUSPECTS: {leakage_suspects}")
    
    # =========================
    # 2. ROLLING FEATURES TEMPORAL CHECK
    # =========================
    logger.info("\n🔍 2. ROLLING FEATURES TEMPORAL INTEGRITY CHECK")
    
    # Features that use rolling calculations
    rolling_features = ['shots_diff_normalized', 'corners_diff_normalized', 'form_diff_normalized']
    
    logger.info("  Checking if rolling features use future information...")
    
    # Convert Date for temporal analysis
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values('Date').copy()
    
    # Sample a few matches to manually verify
    sample_matches = df_sorted.head(10).copy()
    
    logger.info("  Sample matches for temporal verification:")
    for idx, row in sample_matches.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        result = row['FullTimeResult']
        
        logger.info(f"    {date}: {home_team} vs {away_team} = {result}")
        
        # Check rolling features
        for feature in rolling_features:
            value = row[feature]
            logger.info(f"      {feature}: {value:.3f}")
    
    # =========================
    # 3. FEATURE CALCULATION VALIDATION
    # =========================
    logger.info("\n🔍 3. FEATURE CALCULATION VALIDATION")
    
    # Check for impossible values that indicate leakage
    logger.info("  Checking for impossible/suspicious values:")
    
    for feature in v12_features:
        values = df[feature]
        
        # Check basic statistics
        min_val, max_val = values.min(), values.max()
        mean_val, std_val = values.mean(), values.std()
        
        suspicious = []
        
        # Check for values outside expected range [0,1]
        if min_val < 0 or max_val > 1:
            suspicious.append(f"OUT_OF_RANGE[{min_val:.3f},{max_val:.3f}]")
        
        # Check for zero variance (constant values)
        if std_val < 0.001:
            suspicious.append(f"ZERO_VARIANCE({mean_val:.3f})")
        
        # Check for extreme skewness (may indicate leakage)
        from scipy.stats import skew
        skewness = skew(values)
        if abs(skewness) > 3:
            suspicious.append(f"EXTREME_SKEW({skewness:.2f})")
        
        # Check for too many extreme values
        extreme_count = ((values < 0.1) | (values > 0.9)).sum()
        extreme_pct = extreme_count / len(values)
        if extreme_pct > 0.5:
            suspicious.append(f"TOO_MANY_EXTREMES({extreme_pct:.1%})")
        
        if suspicious:
            logger.warning(f"    🚨 {feature}: {', '.join(suspicious)}")
        else:
            logger.info(f"    ✅ {feature}: looks normal")
    
    # =========================
    # 4. MATCH RESULT CORRELATION CHECK
    # =========================
    logger.info("\n🔍 4. MATCH RESULT CORRELATION CHECK")
    
    # Calculate correlation with match results
    logger.info("  Feature correlation with match results:")
    
    # Convert categorical target to numeric for correlation
    result_numeric = y.copy()  # H=0, D=1, A=2
    
    correlations = {}
    for feature in v12_features:
        # Calculate correlation
        corr = np.corrcoef(df[feature], result_numeric)[0, 1]
        correlations[feature] = abs(corr)  # Use absolute correlation
        
        # High correlation might indicate leakage
        if abs(corr) > 0.3:
            logger.warning(f"    🚨 {feature}: {corr:.3f} (HIGH CORRELATION - potential leakage)")
        else:
            logger.info(f"    ✅ {feature}: {corr:.3f}")
    
    # =========================
    # 5. FEATURE ENGINEERING DEEP DIVE
    # =========================
    logger.info("\n🔍 5. FEATURE ENGINEERING DEEP DIVE")
    
    # Check if we can trace back to original data
    original_file = "data/processed/premier_league_ml_ready.csv"
    if os.path.exists(original_file):
        logger.info(f"  Loading original baseline for comparison: {original_file}")
        df_original = pd.read_csv(original_file)
        
        # Compare feature values
        common_features = [f for f in v12_features if f in df_original.columns]
        logger.info(f"  Common features: {common_features}")
        
        for feature in common_features[:3]:  # Check first 3 for brevity
            orig_values = df_original[feature].head(10)
            curr_values = df[feature].head(10)
            
            logger.info(f"  {feature} comparison (first 10):")
            for i in range(min(5, len(orig_values))):
                logger.info(f"    Original: {orig_values.iloc[i]:.3f}, Current: {curr_values.iloc[i]:.3f}")
    
    # =========================
    # 6. CREATE LEAKAGE-FREE DATASET
    # =========================
    logger.info("\n🔍 6. LEAKAGE-FREE DATASET CREATION")
    
    # Identify clean features (low leakage indicators)
    clean_features = []
    suspicious_features = []
    
    for feature in v12_features:
        scores = feature_leakage_scores[feature]
        
        # Criteria for clean feature:
        # - Training accuracy < 0.8
        # - Leakage gap < 0.2
        # - Not zero variance
        # - Reasonable correlation with target
        
        is_clean = (
            scores['train_acc'] < 0.8 and
            scores['leakage_gap'] < 0.2 and
            df[feature].std() > 0.001 and
            correlations[feature] < 0.3
        )
        
        if is_clean:
            clean_features.append(feature)
        else:
            suspicious_features.append(feature)
    
    logger.info(f"  Clean features: {clean_features}")
    logger.info(f"  Suspicious features: {suspicious_features}")
    
    # Test model with clean features only
    if clean_features:
        logger.info("\n  Testing model with clean features only:")
        
        X_clean = df[clean_features]
        rf_clean = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        
        # Training accuracy
        rf_clean.fit(X_clean, y)
        train_acc_clean = rf_clean.score(X_clean, y)
        
        # CV accuracy
        cv_scores_clean = cross_val_score(rf_clean, X_clean, y,
                                        cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
        cv_acc_clean = cv_scores_clean.mean()
        
        leakage_gap_clean = train_acc_clean - cv_acc_clean
        
        logger.info(f"    Clean model train accuracy: {train_acc_clean:.3f}")
        logger.info(f"    Clean model CV accuracy: {cv_acc_clean:.3f}")
        logger.info(f"    Clean model leakage gap: {leakage_gap_clean:.3f}")
        
        if train_acc_clean < 0.8 and leakage_gap_clean < 0.1:
            logger.info("    ✅ LEAKAGE SIGNIFICANTLY REDUCED!")
            leakage_fixed = True
        else:
            logger.warning("    ⚠️ Still showing signs of leakage")
            leakage_fixed = False
    else:
        logger.error("    ❌ No clean features found!")
        leakage_fixed = False
    
    # =========================
    # 7. RECOMMENDATIONS
    # =========================
    logger.info("\n🎯 LEAKAGE INVESTIGATION RESULTS")
    
    logger.info("  Summary of findings:")
    logger.info(f"    Features with leakage: {len(leakage_suspects)}/{len(v12_features)}")
    logger.info(f"    Worst offenders: {leakage_suspects}")
    logger.info(f"    Clean features available: {len(clean_features)}")
    logger.info(f"    Leakage fixed: {'✅' if leakage_fixed else '❌'}")
    
    # Make recommendations
    logger.info("\n🚀 RECOMMENDATIONS:")
    
    if leakage_fixed and clean_features:
        recommendation = "USE_CLEAN_FEATURES"
        logger.info("  ✅ SOLUTION FOUND: Use clean feature subset")
        logger.info(f"     - Use only: {clean_features}")
        logger.info(f"     - Remove: {suspicious_features}")
        logger.info(f"     - Expected CV accuracy: {cv_acc_clean:.3f}")
        
    elif len(suspicious_features) <= 2:
        recommendation = "REMOVE_WORST_FEATURES"
        logger.info("  ⚠️ SELECTIVE REMOVAL: Remove worst 1-2 features")
        logger.info(f"     - Remove: {suspicious_features}")
        logger.info(f"     - Keep: {clean_features}")
        
    else:
        recommendation = "REBUILD_FEATURES"
        logger.info("  🔧 REBUILD REQUIRED: Too many leaky features")
        logger.info("     - Most features show leakage signs")
        logger.info("     - Need to rebuild feature engineering pipeline")
        logger.info("     - Check temporal integrity in source code")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "total_features_tested": len(v12_features),
        "leakage_suspects": leakage_suspects,
        "clean_features": clean_features,
        "suspicious_features": suspicious_features,
        "leakage_fixed": leakage_fixed,
        "recommendation": recommendation,
        "feature_leakage_scores": feature_leakage_scores,
        "correlations": correlations,
        "clean_model_performance": {
            "train_accuracy": float(train_acc_clean) if clean_features else None,
            "cv_accuracy": float(cv_acc_clean) if clean_features else None,
            "leakage_gap": float(leakage_gap_clean) if clean_features else None
        }
    }
    
    results_file = f"models/leakage_investigation_{timestamp}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📊 Investigation results saved: {results_file}")
    logger.info("=== 🔍 DATA LEAKAGE HUNT COMPLETED ===")
    
    return {
        'recommendation': recommendation,
        'leakage_suspects': leakage_suspects,
        'clean_features': clean_features,
        'leakage_fixed': leakage_fixed,
        'clean_cv_accuracy': cv_acc_clean if clean_features else None,
        'results_file': results_file
    }

if __name__ == "__main__":
    result = hunt_data_leakage()
    print(f"\n🎯 LEAKAGE HUNT RESULTS:")
    print(f"Leakage Suspects: {result['leakage_suspects']}")
    print(f"Clean Features: {result['clean_features']}")
    print(f"Leakage Fixed: {'✅' if result['leakage_fixed'] else '❌'}")
    print(f"Recommendation: {result['recommendation']}")
    if result['clean_cv_accuracy']:
        print(f"Clean Model CV: {result['clean_cv_accuracy']:.4f}")