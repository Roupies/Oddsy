#!/usr/bin/env python3
"""
Test Player Features Impact - Synthetic Data PoC
Create synthetic but realistic player absence data to validate approach

Strategy: Generate realistic patterns (10-20% absence rates) and test impact
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticPlayerFeaturesTester:
    """Test player features impact using synthetic but realistic data."""
    
    def __init__(self):
        # Realistic absence rates based on football knowledge
        self.backup_gk_rate = 0.12      # 12% of matches use backup GK
        self.top_scorer_missing_rate = 0.18  # 18% of matches missing top scorer
        
        # Team strength tiers (affects absence impact)
        self.team_tiers = {
            'tier1': ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham'],
            'tier2': ['Brighton', 'Newcastle', 'West Ham', 'Aston Villa', 'Crystal Palace'],
            'tier3': ['Brentford', 'Fulham', 'Wolves', 'Everton', 'Bournemouth', 'Nottm Forest'],
            'tier4': ['Sheffield United', 'Burnley', 'Luton', 'Norwich', 'Watford', 'Leeds']
        }
    
    def generate_realistic_player_features(self, df):
        """Generate synthetic but realistic player absence features."""
        
        logger.info("Generating realistic synthetic player features...")
        
        df = df.copy()
        np.random.seed(42)  # Reproducible results
        
        n_matches = len(df)
        
        # Generate backup goalkeeper features
        # Higher-tier teams have better backup GKs (less impact)
        home_backup_probs = []
        away_backup_probs = []
        
        for _, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Base rates adjusted by team tier
            home_tier = self.get_team_tier(home_team)
            away_tier = self.get_team_tier(away_team)
            
            # Tier 1 teams have slightly lower backup rates (better squad depth)
            home_rate = self.backup_gk_rate * (0.8 if home_tier == 'tier1' else 1.0 if home_tier == 'tier2' else 1.2)
            away_rate = self.backup_gk_rate * (0.8 if away_tier == 'tier1' else 1.0 if away_tier == 'tier2' else 1.2)
            
            home_backup_probs.append(home_rate)
            away_backup_probs.append(away_rate)
        
        # Generate binary features
        df['home_backup_gk_playing'] = np.random.binomial(1, home_backup_probs)
        df['away_backup_gk_playing'] = np.random.binomial(1, away_backup_probs)
        
        # Generate top scorer missing features (similar logic)
        home_missing_probs = []
        away_missing_probs = []
        
        for _, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            home_tier = self.get_team_tier(home_team)
            away_tier = self.get_team_tier(away_team)
            
            # Top teams have better squad depth
            home_rate = self.top_scorer_missing_rate * (0.7 if home_tier == 'tier1' else 1.0 if home_tier == 'tier2' else 1.3)
            away_rate = self.top_scorer_missing_rate * (0.7 if away_tier == 'tier1' else 1.0 if away_tier == 'tier2' else 1.3)
            
            home_missing_probs.append(home_rate)
            away_missing_probs.append(away_rate)
        
        df['home_top_scorer_missing'] = np.random.binomial(1, home_missing_probs)
        df['away_top_scorer_missing'] = np.random.binomial(1, away_missing_probs)
        
        # Create derived advantage features
        df['gk_advantage'] = df['away_backup_gk_playing'] - df['home_backup_gk_playing']  # Positive = home advantage
        df['scorer_advantage'] = df['away_top_scorer_missing'] - df['home_top_scorer_missing']  # Positive = home advantage
        
        # Add realistic correlations with existing features
        # Better teams (higher Elo) should have less impact from absences
        elo_normalized = df.get('elo_diff_normalized', 0.5)
        
        # Adjust player impact based on team strength difference
        strength_factor = np.where(elo_normalized > 0.6, 0.8,  # Strong home team - less absence impact
                                 np.where(elo_normalized < 0.4, 1.2, 1.0))  # Weak home team - more impact
        
        df['gk_advantage_weighted'] = df['gk_advantage'] * strength_factor
        df['scorer_advantage_weighted'] = df['scorer_advantage'] * strength_factor
        
        player_features = [
            'home_backup_gk_playing', 'away_backup_gk_playing',
            'home_top_scorer_missing', 'away_top_scorer_missing', 
            'gk_advantage', 'scorer_advantage',
            'gk_advantage_weighted', 'scorer_advantage_weighted'
        ]
        
        logger.info("Synthetic player features generated:")
        for feature in player_features:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            logger.info(f"  • {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        return df, player_features
    
    def get_team_tier(self, team_name):
        """Get team tier for strength-based adjustments."""
        for tier, teams in self.team_tiers.items():
            if team_name in teams:
                return tier
        return 'tier3'  # Default
    
    def test_synthetic_player_impact(self):
        """Test player features impact using synthetic data."""
        
        logger.info("🚀 Testing synthetic player features impact...")
        
        # Load v3.1 baseline dataset
        df = pd.read_csv('data/processed/v31_efficiency_features_2025_09_06.csv')
        logger.info(f"Loaded v3.1 dataset: {df.shape}")
        
        # Generate synthetic player features
        enhanced_df, player_features = self.generate_realistic_player_features(df)
        
        # Baseline features (from previous v3.1 test)
        baseline_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
            'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
            'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        # Select best efficiency features from previous test
        efficiency_features = [
            'goalkeeping_advantage_10', 'net_performance_advantage_10_normalized',
            'away_goalkeeping_efficiency_10_normalized', 'net_performance_advantage_10',
            'goalkeeping_advantage_10_normalized'
        ]
        
        # Combined feature sets to test
        test_scenarios = {
            'baseline': baseline_features,
            'baseline_plus_efficiency': baseline_features + efficiency_features,
            'baseline_plus_players': baseline_features + player_features,
            'all_features': baseline_features + efficiency_features + player_features
        }
        
        # Clean data and prepare for testing
        all_test_features = list(set(baseline_features + efficiency_features + player_features))
        available_features = [f for f in all_test_features if f in enhanced_df.columns]
        
        df_clean = enhanced_df.dropna(subset=available_features + ['FullTimeResult'])
        logger.info(f"Clean dataset for testing: {df_clean.shape}")
        
        # Train/test split
        split_idx = int(len(df_clean) * 0.8)
        df_train = df_clean[:split_idx]
        df_test = df_clean[split_idx:]
        
        # Target encoding
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y_train = df_train['FullTimeResult'].map(target_mapping)
        y_test = df_test['FullTimeResult'].map(target_mapping)
        
        # Model configuration
        model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Test each scenario
        results = {}
        
        for scenario_name, features in test_scenarios.items():
            logger.info(f"Testing {scenario_name} scenario...")
            
            # Filter available features
            scenario_features = [f for f in features if f in enhanced_df.columns]
            
            if len(scenario_features) < 5:
                logger.warning(f"Insufficient features for {scenario_name}: {len(scenario_features)}")
                continue
            
            # Train model
            model = RandomForestClassifier(**model_params)
            X_train = df_train[scenario_features]
            X_test = df_test[scenario_features]
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Feature importance
            feature_importance = list(zip(scenario_features, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            results[scenario_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'num_features': len(scenario_features),
                'top_features': feature_importance[:10]
            }
        
        # Generate comparative report
        print("\\n" + "="*80)
        print("🎯 SYNTHETIC PLAYER FEATURES IMPACT TEST")
        print("="*80)
        
        print(f"\\n📊 PERFORMANCE COMPARISON:")
        baseline_acc = results.get('baseline', {}).get('accuracy', 0)
        
        for scenario, result in results.items():
            accuracy = result['accuracy']
            improvement = (accuracy - baseline_acc) * 100 if baseline_acc > 0 else 0
            
            print(f"   • {scenario.title()}: {accuracy:.4f} ({accuracy*100:.2f}%) [{improvement:+.2f}pp]")
        
        # Detailed analysis of best scenario
        best_scenario = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_name, best_result = best_scenario
        
        print(f"\\n🏆 BEST SCENARIO: {best_name.upper()}")
        print(f"   • Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
        print(f"   • Features: {best_result['num_features']}")
        print(f"   • Top Features:")
        
        for i, (feature, importance) in enumerate(best_result['top_features'][:8]):
            feature_type = "🎯 PLAYER" if feature in player_features else "📊 OTHER"
            print(f"     {i+1}. {feature_type} {feature}: {importance:.3f}")
        
        # Player features analysis
        player_feature_importance = []
        if best_name in ['baseline_plus_players', 'all_features']:
            for feature, importance in best_result['top_features']:
                if feature in player_features:
                    player_feature_importance.append((feature, importance))
        
        print(f"\\n🎯 PLAYER FEATURES ANALYSIS:")
        if player_feature_importance:
            print(f"   • Player features in top 10: {len(player_feature_importance)}")
            print(f"   • Top player feature: {player_feature_importance[0][0]} ({player_feature_importance[0][1]:.3f})")
            total_player_importance = sum(imp for _, imp in player_feature_importance)
            print(f"   • Combined player importance: {total_player_importance:.3f}")
        else:
            print(f"   • No player features in top 10 importance")
        
        # Success assessment
        print(f"\\n🚀 SUCCESS ASSESSMENT:")
        
        if 'all_features' in results and 'baseline_plus_efficiency' in results:
            all_acc = results['all_features']['accuracy']
            efficiency_acc = results['baseline_plus_efficiency']['accuracy']
            player_contribution = (all_acc - efficiency_acc) * 100
            
            print(f"   • v3.1 Efficiency baseline: {efficiency_acc:.4f} ({efficiency_acc*100:.2f}%)")
            print(f"   • With player features: {all_acc:.4f} ({all_acc*100:.2f}%)")
            print(f"   • Player contribution: {player_contribution:+.2f}pp")
            
            if player_contribution >= 0.5:
                status = "✅ SUCCESS - Player features add value!"
                recommendation = "Proceed with real FBref data collection"
            elif player_contribution >= 0.2:
                status = "⚡ MARGINAL - Small but positive impact"
                recommendation = "Consider proceeding with caution"
            else:
                status = "❌ FAILED - No meaningful improvement"
                recommendation = "Focus on other approaches (ensemble, external data)"
            
            print(f"   • Status: {status}")
            print(f"   • Recommendation: {recommendation}")
        
        print(f"\\n📋 NEXT STEPS:")
        if player_feature_importance and len(player_feature_importance) > 0:
            print(f"   1. Real data validation: Collect actual FBref lineup data")
            print(f"   2. Feature engineering: Refine player absence detection")
            print(f"   3. Scale up: Extend to full 5-season dataset")
        else:
            print(f"   1. Alternative approach: Focus on ensemble methods")
            print(f"   2. External data: Weather, referee, injury reports")
            print(f"   3. Architecture: Advanced model architectures")
        
        return results, player_feature_importance

def main():
    """Execute synthetic player features impact test."""
    
    logger.info("🚀 Starting Synthetic Player Features Impact Test...")
    
    tester = SyntheticPlayerFeaturesTester()
    results, player_importance = tester.test_synthetic_player_impact()
    
    logger.info("✅ Synthetic Player Features Test Complete!")
    return results, player_importance

if __name__ == "__main__":
    main()