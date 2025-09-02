import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def engineer_advanced_features():
    """
    PHASE 2.0.2 - Advanced Feature Engineering
    
    Focus on sophisticated feature interactions and temporal patterns:
    1. Feature Interactions (polynomial features, domain-specific combinations)
    2. Time-Decay Weighting (recent form more important)
    3. Contextual Adjustments (position-based, manager effects)
    4. Advanced Statistical Features (volatility, momentum)
    
    CRITICAL: Build on Phase 2.0.1 advanced odds features
    Target: True v2.0 breakthrough to 55%+ accuracy
    """
    
    logger = setup_logging()
    logger.info("=== 🚀 PHASE 2.0.2 - Advanced Feature Engineering ===")
    
    # Load Phase 2.0.1 advanced odds features
    odds_file = "data/processed/advanced_odds_features_v2_2025_08_30_210608.csv"
    logger.info(f"Loading Phase 2.0.1 advanced odds features: {odds_file}")
    
    df_odds = pd.read_csv(odds_file)
    df_odds['Date'] = pd.to_datetime(df_odds['Date'])
    df_odds = df_odds.sort_values('Date').reset_index(drop=True)
    logger.info(f"Advanced odds dataset: {df_odds.shape}")
    
    # Load v1.3 baseline features
    baseline_file = "data/processed/premier_league_2019_2024_corrected_elo.csv"
    df_baseline = pd.read_csv(baseline_file)
    df_baseline['Date'] = pd.to_datetime(df_baseline['Date'])
    logger.info(f"v1.3 baseline dataset: {df_baseline.shape}")
    
    logger.info("Engineering advanced features...")
    
    # =========================
    # 1. FEATURE INTERACTIONS
    # =========================
    logger.info("\n🔗 1. FEATURE INTERACTIONS")
    
    # Create match keys for merging
    df_odds['match_key'] = (df_odds['Date'].dt.strftime('%Y-%m-%d') + '_' + 
                           df_odds['HomeTeam'] + '_' + df_odds['AwayTeam'])
    df_baseline['match_key'] = (df_baseline['Date'].dt.strftime('%Y-%m-%d') + '_' + 
                               df_baseline['HomeTeam'] + '_' + df_baseline['AwayTeam'])
    
    # Merge baseline features with advanced odds
    core_features = ['elo_diff_normalized', 'form_diff_normalized', 'h2h_score']
    advanced_features = ['line_movement_normalized', 'sharp_public_divergence_norm', 
                        'market_inefficiency_norm', 'market_velocity_norm']
    
    baseline_merge = df_baseline[['match_key'] + core_features]
    df_merged = df_odds.merge(baseline_merge, on='match_key', how='left')
    logger.info(f"Merged dataset: {df_merged.shape}")
    
    # Domain-specific interactions
    logger.info("  Creating domain-specific feature interactions...")
    
    # Interaction 1: Elo strength vs Market uncertainty
    df_merged['elo_market_interaction'] = (df_merged['elo_diff_normalized'] * 
                                          (1 - df_merged['market_inefficiency_norm']))
    
    # Interaction 2: Form vs Sharp money divergence (hot teams vs smart money)
    df_merged['form_sharp_interaction'] = (df_merged['form_diff_normalized'] * 
                                          df_merged['sharp_public_divergence_norm'])
    
    # Interaction 3: Line movement vs Historical advantage
    df_merged['line_h2h_interaction'] = (df_merged['line_movement_normalized'] * 
                                        df_merged['h2h_score'])
    
    # Interaction 4: Market velocity vs Elo (surprise factor)
    df_merged['velocity_elo_interaction'] = (df_merged['market_velocity_norm'] * 
                                            (0.5 - abs(df_merged['elo_diff_normalized'] - 0.5)))
    
    # Normalize interactions to [0,1]
    interaction_features = ['elo_market_interaction', 'form_sharp_interaction', 
                           'line_h2h_interaction', 'velocity_elo_interaction']
    
    for feature in interaction_features:
        min_val = df_merged[feature].min()
        max_val = df_merged[feature].max()
        if max_val > min_val:
            df_merged[f"{feature}_norm"] = (df_merged[feature] - min_val) / (max_val - min_val)
        else:
            df_merged[f"{feature}_norm"] = 0.5
    
    logger.info(f"  Created {len(interaction_features)} domain-specific interactions")
    
    # =========================
    # 2. TIME-DECAY WEIGHTING
    # =========================
    logger.info("\n⏰ 2. TIME-DECAY WEIGHTING")
    
    # Calculate time-weighted features (recent matches more important)
    df_merged = df_merged.sort_values('Date').reset_index(drop=True)
    
    # Time decay for form (exponential decay with half-life of 10 matches)
    decay_factor = 0.93  # 10-match half-life: 0.5^(1/10) ≈ 0.93
    
    # Group by team and calculate time-weighted form
    logger.info("  Calculating time-weighted form scores...")
    
    def calculate_time_weighted_form(group, result_col='FullTimeResult', team_perspective='home'):
        """Calculate exponentially weighted form"""
        group = group.sort_values('Date')
        weights = decay_factor ** np.arange(len(group) - 1, -1, -1)
        
        if team_perspective == 'home':
            wins = (group[result_col] == 'H').astype(float)
        else:  # away
            wins = (group[result_col] == 'A').astype(float)
        
        if len(wins) > 0:
            weighted_form = np.average(wins, weights=weights)
        else:
            weighted_form = 0.5
        
        return weighted_form
    
    # Time-weighted home and away form
    home_form_weighted = []
    away_form_weighted = []
    
    for idx, row in df_merged.iterrows():
        current_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get historical matches before current date
        home_history = df_merged[(df_merged['HomeTeam'] == home_team) & 
                                (df_merged['Date'] < current_date)].tail(10)  # Last 10 matches
        away_history = df_merged[(df_merged['AwayTeam'] == away_team) & 
                                (df_merged['Date'] < current_date)].tail(10)
        
        home_weighted = calculate_time_weighted_form(home_history, team_perspective='home')
        away_weighted = calculate_time_weighted_form(away_history, team_perspective='away')
        
        home_form_weighted.append(home_weighted)
        away_form_weighted.append(away_weighted)
    
    df_merged['time_weighted_form_home'] = home_form_weighted
    df_merged['time_weighted_form_away'] = away_form_weighted
    df_merged['time_weighted_form_diff'] = (df_merged['time_weighted_form_home'] - 
                                           df_merged['time_weighted_form_away'])
    
    # Normalize time-weighted form difference
    min_val = df_merged['time_weighted_form_diff'].min()
    max_val = df_merged['time_weighted_form_diff'].max()
    df_merged['time_weighted_form_diff_norm'] = ((df_merged['time_weighted_form_diff'] - min_val) / 
                                                (max_val - min_val))
    
    logger.info("  Time-weighted form features created")
    
    # =========================
    # 3. MOMENTUM & VOLATILITY
    # =========================
    logger.info("\n📈 3. MOMENTUM & VOLATILITY FEATURES")
    
    # Momentum: rate of change in recent performance
    def calculate_momentum(group, window=5):
        """Calculate performance momentum (slope of recent results)"""
        if len(group) < window:
            return 0.5
        
        recent_results = group.tail(window)['FullTimeResult']
        
        # Convert results to numeric (2=win, 1=draw, 0=loss)
        result_values = recent_results.map({'H': 2, 'D': 1, 'A': 0}).fillna(1)
        
        if len(result_values) < 2:
            return 0.5
        
        # Calculate slope (momentum)
        x = np.arange(len(result_values))
        momentum = np.polyfit(x, result_values, 1)[0]  # Linear slope
        
        # Normalize to [0,1] (assuming momentum range [-2, 2])
        return (momentum + 2) / 4
    
    # Team momentum calculation
    momentum_scores = []
    volatility_scores = []
    
    for idx, row in df_merged.iterrows():
        current_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get recent team history (both home and away games)
        home_recent = df_merged[((df_merged['HomeTeam'] == home_team) | 
                                (df_merged['AwayTeam'] == home_team)) & 
                               (df_merged['Date'] < current_date)].tail(8)
        away_recent = df_merged[((df_merged['HomeTeam'] == away_team) | 
                                (df_merged['AwayTeam'] == away_team)) & 
                               (df_merged['Date'] < current_date)].tail(8)
        
        # Calculate momentum
        home_momentum = calculate_momentum(home_recent)
        away_momentum = calculate_momentum(away_recent)
        momentum_diff = home_momentum - away_momentum
        
        # Calculate volatility (standard deviation of recent results)
        def calc_volatility(team_recent):
            if len(team_recent) < 3:
                return 0.5
            
            results = []
            for _, match in team_recent.iterrows():
                if match['HomeTeam'] == home_team:
                    result = 2 if match['FullTimeResult'] == 'H' else (1 if match['FullTimeResult'] == 'D' else 0)
                else:
                    result = 2 if match['FullTimeResult'] == 'A' else (1 if match['FullTimeResult'] == 'D' else 0)
                results.append(result)
            
            return np.std(results) / 2  # Normalize by max std
        
        home_volatility = calc_volatility(home_recent)
        away_volatility = calc_volatility(away_recent)
        volatility_diff = home_volatility - away_volatility
        
        momentum_scores.append(momentum_diff)
        volatility_scores.append(volatility_diff)
    
    df_merged['momentum_diff'] = momentum_scores
    df_merged['volatility_diff'] = volatility_scores
    
    # Normalize momentum and volatility
    for feature in ['momentum_diff', 'volatility_diff']:
        min_val = df_merged[feature].min()
        max_val = df_merged[feature].max()
        if max_val > min_val:
            df_merged[f"{feature}_norm"] = (df_merged[feature] - min_val) / (max_val - min_val)
        else:
            df_merged[f"{feature}_norm"] = 0.5
    
    logger.info("  Momentum and volatility features created")
    
    # =========================
    # 4. FEATURE SELECTION & VALIDATION
    # =========================
    logger.info("\n🎯 4. ADVANCED FEATURE SELECTION")
    
    # All engineered features
    all_new_features = [
        'elo_market_interaction_norm', 'form_sharp_interaction_norm',
        'line_h2h_interaction_norm', 'velocity_elo_interaction_norm',
        'time_weighted_form_diff_norm', 'momentum_diff_norm', 'volatility_diff_norm'
    ]
    
    # Core feature set (existing validated features)
    core_feature_set = [
        'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
        'line_movement_normalized', 'sharp_public_divergence_norm', 
        'market_inefficiency_norm', 'market_velocity_norm'
    ]
    
    complete_feature_set = core_feature_set + all_new_features
    
    logger.info(f"  Core features: {len(core_feature_set)}")
    logger.info(f"  New features: {len(all_new_features)}")  
    logger.info(f"  Total features: {len(complete_feature_set)}")
    
    # Prepare target for feature selection
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df_merged['FullTimeResult'].map(label_mapping)
    X = df_merged[complete_feature_set].fillna(0.5)
    
    # Statistical feature selection using F-score
    logger.info("  Performing statistical feature selection...")
    selector = SelectKBest(score_func=f_classif, k=12)  # Select top 12 features
    X_selected = selector.fit_transform(X, y)
    
    # Get selected features
    feature_scores = selector.scores_
    selected_mask = selector.get_support()
    selected_features = [complete_feature_set[i] for i in range(len(complete_feature_set)) if selected_mask[i]]
    
    logger.info(f"  Selected {len(selected_features)} features:")
    for i, feature in enumerate(complete_feature_set):
        score = feature_scores[i]
        selected = "✅ SELECTED" if selected_mask[i] else "❌ DROPPED"
        logger.info(f"    {feature}: F-score={score:.2f} {selected}")
    
    # =========================
    # 5. FINAL DATASET CREATION
    # =========================
    logger.info("\n💾 5. CREATING FINAL v2.0 DATASET")
    
    # Create output dataset with selected features
    output_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult'] + selected_features
    df_output = df_merged[output_columns].copy()
    
    # Data quality validation
    logger.info("Final v2.0 dataset validation:")
    logger.info(f"  Shape: {df_output.shape}")
    logger.info(f"  Missing values: {df_output[selected_features].isnull().sum().sum()}")
    logger.info(f"  Feature ranges:")
    
    for feature in selected_features:
        values = df_output[feature].dropna()
        logger.info(f"    {feature}: [{values.min():.3f}, {values.max():.3f}], mean={values.mean():.3f}")
    
    # Target distribution
    target_dist = df_output['FullTimeResult'].value_counts(normalize=True).sort_index()
    logger.info(f"  Target distribution: H={target_dist.get('H', 0):.3f}, D={target_dist.get('D', 0):.3f}, A={target_dist.get('A', 0):.3f}")
    
    # Save v2.0 dataset
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = f"data/processed/premier_league_v20_advanced_{timestamp}.csv"
    
    df_output.to_csv(output_file, index=False)
    logger.info(f"v2.0 Advanced dataset saved: {output_file}")
    
    # Save feature metadata
    feature_metadata = {
        "version": "v2.0 Advanced Feature Engineering",
        "timestamp": timestamp,
        "phases_completed": ["2.0.1 Advanced Odds", "2.0.2 Advanced Engineering"],
        "feature_categories": {
            "core_features": core_feature_set,
            "interaction_features": [f for f in all_new_features if 'interaction' in f],
            "temporal_features": [f for f in all_new_features if 'time_weighted' in f or 'momentum' in f or 'volatility' in f],
            "selected_features": selected_features
        },
        "feature_selection": {
            "method": "SelectKBest with F-score",
            "k_selected": len(selected_features),
            "total_engineered": len(complete_feature_set)
        },
        "feature_descriptions": {
            "elo_market_interaction_norm": "Elo strength modulated by market uncertainty",
            "form_sharp_interaction_norm": "Recent form vs sharp money divergence",  
            "line_h2h_interaction_norm": "Line movement adjusted by historical H2H",
            "velocity_elo_interaction_norm": "Market velocity vs strength differential surprise",
            "time_weighted_form_diff_norm": "Exponentially weighted recent form (10-match decay)",
            "momentum_diff_norm": "Performance momentum differential (trend slope)",
            "volatility_diff_norm": "Performance consistency differential"
        },
        "data_quality": {
            "total_matches": len(df_output),
            "missing_values": int(df_output[selected_features].isnull().sum().sum()),
            "feature_count": len(selected_features)
        }
    }
    
    import json
    metadata_file = f"config/advanced_features_v20_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    logger.info(f"Feature metadata saved: {metadata_file}")
    logger.info("\n=== 🚀 PHASE 2.0.2 ADVANCED FEATURE ENGINEERING COMPLETED ===")
    
    return {
        'output_file': output_file,
        'metadata_file': metadata_file,
        'total_features_engineered': len(complete_feature_set),
        'features_selected': len(selected_features),
        'selected_features': selected_features,
        'core_features': len(core_feature_set),
        'new_features': len(all_new_features),
        'total_matches': len(df_output)
    }

if __name__ == "__main__":
    result = engineer_advanced_features()
    print(f"\n🎯 ADVANCED FEATURE ENGINEERING RESULTS:")
    print(f"Features Engineered: {result['total_features_engineered']}")
    print(f"Features Selected: {result['features_selected']}")
    print(f"Core Features: {result['core_features']}, New Features: {result['new_features']}")
    print(f"Selected Features: {result['selected_features']}")
    print(f"Output: {result['output_file']}")
    print(f"Total Matches: {result['total_matches']}")