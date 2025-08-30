import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def create_intelligent_interactions(df):
    """
    Create selective, football-intelligent feature interactions.
    Focus on meaningful combinations that make sense in football context.
    """
    
    # Base features
    interactions_df = df.copy()
    
    # Interaction 1: Team Strength × Recent Form
    # Strong teams in good form = very dangerous
    interactions_df['strength_form'] = (
        interactions_df['elo_diff_normalized'] * 
        interactions_df['form_diff_normalized']
    )
    
    # Interaction 2: Offensive Dominance Combined
    # Teams that both shoot more AND get more corners = total dominance
    interactions_df['offensive_dominance'] = (
        interactions_df['shots_diff_normalized'] * 
        interactions_df['corners_diff_normalized']
    )
    
    # Interaction 3: Season Experience Effect
    # Elo difference matters more later in season (teams know their level)
    interactions_df['elo_season_adjusted'] = (
        interactions_df['elo_diff_normalized'] * 
        (0.5 + 0.5 * interactions_df['matchday_normalized'])  # Weight: 0.5 to 1.0
    )
    
    # Interaction 4: Form Momentum in Season Context
    # Form matters more in mid/late season than early season
    interactions_df['form_season_weighted'] = (
        interactions_df['form_diff_normalized'] * 
        interactions_df['season_period_numeric']
    )
    
    # Interaction 5: Historical vs Current Performance
    # H2H history combined with current form
    interactions_df['h2h_form_blend'] = (
        0.6 * interactions_df['h2h_score'] + 
        0.4 * interactions_df['form_diff_normalized']
    )
    
    # Normalize all interactions to [0,1]
    interaction_features = [
        'strength_form', 'offensive_dominance', 'elo_season_adjusted',
        'form_season_weighted', 'h2h_form_blend'
    ]
    
    for feature in interaction_features:
        min_val = interactions_df[feature].min()
        max_val = interactions_df[feature].max()
        if max_val > min_val:  # Avoid division by zero
            interactions_df[feature] = (interactions_df[feature] - min_val) / (max_val - min_val)
        else:
            interactions_df[feature] = 0.5  # Neutral if no variation
    
    return interactions_df, interaction_features

def test_feature_interactions_v13():
    """
    Test impact of intelligent feature interactions on model performance.
    Compare various combinations to find optimal feature set.
    """
    
    logger = setup_logging()
    logger.info("=== v1.3 Feature Interactions Testing ===")
    
    # Load base dataset
    input_file = "data/processed/premier_league_ml_ready.csv"
    df = pd.read_csv(input_file)
    
    # Original v1.2 features
    original_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized'
    ]
    
    # Create interactions
    logger.info("Creating intelligent feature interactions...")
    df_interactions, interaction_features = create_intelligent_interactions(df)
    
    logger.info(f"Original features: {len(original_features)}")
    logger.info(f"New interactions: {len(interaction_features)}")
    for i, feature in enumerate(interaction_features, 1):
        logger.info(f"  {i}. {feature}")
    
    # Prepare target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    # Setup cross-validation
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Define test configurations
    test_configs = {
        'Baseline (8 original)': original_features,
        'All Features (8+5)': original_features + interaction_features,
        'Best Original + Top 2 Interactions': original_features + ['strength_form', 'offensive_dominance'],
        'Best Original + Top 3 Interactions': original_features + ['strength_form', 'offensive_dominance', 'elo_season_adjusted'],
        'Selective Mix': ['elo_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized', 
                         'form_diff_normalized', 'matchday_normalized', 'strength_form', 
                         'offensive_dominance', 'elo_season_adjusted']
    }
    
    # Test models
    models = {
        'Random Forest Optimized': RandomForestClassifier(
            n_estimators=300, max_depth=12, max_features='log2',
            min_samples_leaf=2, min_samples_split=15, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'Voting Hard (RF+XGB)': VotingClassifier([
            ('rf', RandomForestClassifier(
                n_estimators=300, max_depth=12, max_features='log2',
                min_samples_leaf=2, min_samples_split=15, class_weight='balanced',
                random_state=42, n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                eval_metric='mlogloss', n_jobs=-1
            ))
        ], voting='hard')
    }
    
    results = {}
    
    # Test all combinations
    logger.info("\\n=== Testing Feature Combinations ===")
    
    for config_name, features in test_configs.items():
        logger.info(f"\\nTesting configuration: {config_name}")
        logger.info(f"Features ({len(features)}): {features}")
        
        X = df_interactions[features]
        
        # Verify no NaN values
        if X.isnull().sum().sum() > 0:
            logger.warning(f"NaN values detected in {config_name}, skipping...")
            continue
        
        config_results = {}
        
        for model_name, model in models.items():
            logger.info(f"  Testing with {model_name}...")
            
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            logger.info(f"    CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
            logger.info(f"    Mean: {mean_score:.4f} (±{std_score:.4f})")
            
            config_results[model_name] = {
                'cv_scores': cv_scores.tolist(),
                'mean': float(mean_score),
                'std': float(std_score)
            }
        
        results[config_name] = {
            'features': features,
            'feature_count': len(features),
            'results': config_results
        }
    
    # Find best configuration
    best_config = None
    best_score = 0
    best_model = None
    
    logger.info("\\n=== Feature Interaction Results Summary ===")
    
    for config_name, config_data in results.items():
        logger.info(f"\\n{config_name} ({config_data['feature_count']} features):")
        
        for model_name, model_results in config_data['results'].items():
            score = model_results['mean']
            std = model_results['std']
            logger.info(f"  {model_name}: {score:.4f} (±{std:.4f})")
            
            if score > best_score:
                best_score = score
                best_config = config_name
                best_model = model_name
    
    # Compare against baselines
    baseline_v12 = 0.522
    baseline_v13_ensemble = 0.524
    improvement_v12 = (best_score - baseline_v12) * 100
    improvement_v13 = (best_score - baseline_v13_ensemble) * 100
    
    logger.info(f"\\n=== Best Configuration Found ===")
    logger.info(f"Config: {best_config}")
    logger.info(f"Model: {best_model}")
    logger.info(f"Score: {best_score:.4f}")
    logger.info(f"Features: {results[best_config]['features']}")
    
    logger.info(f"\\nImprovement Analysis:")
    logger.info(f"  vs v1.2 Baseline (52.2%): {improvement_v12:+.1f}pp")
    logger.info(f"  vs v1.3 Ensemble (52.4%): {improvement_v13:+.1f}pp")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results_file = f"models/feature_interactions_v13_results_{timestamp}.json"
    
    final_results = {
        "version": "v1.3 Feature Interactions",
        "timestamp": timestamp,
        "best_config": best_config,
        "best_model": best_model,
        "best_score": float(best_score),
        "baseline_v12": baseline_v12,
        "baseline_v13_ensemble": baseline_v13_ensemble,
        "improvement_vs_v12": float(improvement_v12),
        "improvement_vs_v13": float(improvement_v13),
        "interaction_features_created": interaction_features,
        "all_configurations": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"\\nResults saved: {results_file}")
    
    # Final recommendation
    if improvement_v13 > 0.5:
        logger.info("✅ SIGNIFICANT IMPROVEMENT: Use feature interactions")
        recommendation = "USE_INTERACTIONS"
    elif improvement_v13 > 0.0:
        logger.info("⚠️ MARGINAL IMPROVEMENT: Consider feature interactions")
        recommendation = "CONSIDER_INTERACTIONS"
    else:
        logger.info("❌ NO IMPROVEMENT: Stick with ensemble without interactions")
        recommendation = "STICK_WITH_ENSEMBLE"
    
    return {
        'best_config': best_config,
        'best_model': best_model,
        'best_score': best_score,
        'improvement_v12': improvement_v12,
        'improvement_v13': improvement_v13,
        'recommendation': recommendation,
        'results_file': results_file
    }

if __name__ == "__main__":
    result = test_feature_interactions_v13()
    print(f"\\nv1.3 Feature Interactions completed!")
    print(f"Best: {result['best_config']} + {result['best_model']}")
    print(f"Score: {result['best_score']:.4f}")
    print(f"Improvement vs v1.2: {result['improvement_v12']:+.1f}pp")
    print(f"Improvement vs v1.3 ensemble: {result['improvement_v13']:+.1f}pp")
    print(f"Recommendation: {result['recommendation']}")