#!/usr/bin/env python3
"""
Test All Models on Real Premier League 2025-26 Matches
Complete comparison on 39 real matches from first 2 matchweeks

Strategy: Fair comparison using identical methodology and real match outcomes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealSeasonModelTester:
    """Test all models on real 2025-26 season matches."""
    
    def __init__(self):
        self.models_config = {
            'v23_corrected': {
                'name': 'v2.3 Corrected xG Integration (Champion)',
                'dataset': 'data/processed/v13_xg_corrected_features_fixed_complete.csv',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10',
                    'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
                    'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
                ]
            },
            'v31_efficiency': {
                'name': 'v3.1 Efficiency Breakthrough (Money Focus)', 
                'dataset': 'data/processed/v31_efficiency_features_fixed_complete.csv',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'away_xg_eff_10',
                    'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
                    'form_diff_normalized', 'h2h_score', 'away_goals_sum_5', 'goalkeeping_advantage_10',
                    'away_goalkeeping_efficiency_10_normalized', 'goalkeeping_advantage_10_normalized', 
                    'net_performance_advantage_10_normalized', 'net_performance_advantage_10', 'goalkeeping_advantage_5_normalized'
                ]
            },
            'v40_fatigue': {
                'name': 'v4.0 Fatigue Features (Fixture Congestion)',
                'dataset': 'data/processed/v40_fatigue_features_fixed_complete.csv', 
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'away_xg_eff_10',
                    'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
                    'form_diff_normalized', 'h2h_score', 'away_goals_sum_5', 'goalkeeping_advantage_10',
                    'away_goalkeeping_efficiency_10_normalized', 'goalkeeping_advantage_10_normalized', 
                    'net_performance_advantage_10_normalized', 'net_performance_advantage_10', 'goalkeeping_advantage_5_normalized'
                ]
            },
            'v41_referee': {
                'name': 'v4.1 Referee Intelligence (Official Bias)',
                'dataset': 'data/processed/v41_referee_features_fixed_complete.csv',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'away_xg_eff_10',
                    'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
                    'form_diff_normalized', 'h2h_score', 'away_goals_sum_5', 'goalkeeping_advantage_10',
                    'away_goalkeeping_efficiency_10_normalized', 'goalkeeping_advantage_10_normalized', 
                    'net_performance_advantage_10_normalized', 'net_performance_advantage_10', 'goalkeeping_advantage_5_normalized'
                ]
            }
        }
        
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def load_real_season_data(self):
        """Load real 2025-26 season matches."""
        
        logger.info("Loading real Premier League 2025-26 matches...")
        
        df_real = pd.read_csv('data/processed/premier_league_2025_26_all_matches_played.csv')
        df_real['Date'] = pd.to_datetime(df_real['Date'])
        
        logger.info(f"Loaded {len(df_real)} real matches from 2025-26 season")
        
        return df_real
    
    def prepare_features(self, df_real, model_key):
        """Prepare features for specific model."""
        
        model_config = self.models_config[model_key]
        required_features = model_config['features']
        
        logger.info(f"Preparing features for {model_key}...")
        
        # Check available features
        available_features = [f for f in required_features if f in df_real.columns]
        missing_features = [f for f in required_features if f not in df_real.columns]
        
        if missing_features:
            logger.warning(f"Missing features for {model_key}: {missing_features}")
            if len(available_features) < len(required_features) * 0.7:  # Less than 70% features available
                logger.error(f"Too many missing features for {model_key} - skipping")
                return None, None
        
        return available_features, missing_features
        available_features = [f for f in required_features if f in df_real.columns]
        missing_features = [f for f in required_features if f not in df_real.columns]
        
        if missing_features:
            logger.warning(f"Missing features for {model_key}: {missing_features}")
            # Add missing features with neutral values
            for feature in missing_features:
                if 'efficiency' in feature or 'fatigue' in feature or 'referee' in feature:
                    df_real[feature] = 0.5  # Neutral efficiency/fatigue/referee values
                elif 'days_rest' in feature:
                    df_real[feature] = 3.0  # Standard rest days
                else:
                    df_real[feature] = 0.5  # Neutral default
            
            available_features = required_features
        
        logger.info(f"Prepared {len(available_features)}/{len(required_features)} features for {model_key}")
        
        return available_features
    
    def train_and_test_model(self, model_key, df_real, available_features):
        """Train model on historical data and test on real 2025-26 matches."""
        
        model_config = self.models_config[model_key]
        
        logger.info(f"Testing {model_key}...")
        
        # Load historical training data
        df_historical = pd.read_csv(model_config['dataset'])
        df_historical['Date'] = pd.to_datetime(df_historical['Date'])
        
        # Prepare training data
        X_train = df_historical[available_features]
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y_train = df_historical['FullTimeResult'].map(target_mapping)
        
        # Remove missing targets
        valid_mask = y_train.notna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        logger.info(f"{model_key} training data: {len(X_train)} matches")
        
        # Train model
        model = RandomForestClassifier(**self.model_params)
        model.fit(X_train, y_train)
        
        # Test on real 2025-26 data
        X_test = df_real[available_features]
        y_test = df_real['FullTimeResult'].map(target_mapping)
        
        # Generate predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Convert back to labels
        label_mapping = {0: 'H', 1: 'D', 2: 'A'}
        predicted_labels = [label_mapping[pred] for pred in predictions]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Create detailed results
        results = df_real[['MatchWeek', 'Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'FTHG', 'FTAG']].copy()
        results['Predicted'] = predicted_labels
        results['Prob_Home'] = probabilities[:, 0]
        results['Prob_Draw'] = probabilities[:, 1]
        results['Prob_Away'] = probabilities[:, 2]
        results['Confidence'] = np.max(probabilities, axis=1)
        results['Correct'] = (results['FullTimeResult'] == results['Predicted'])
        
        # Feature importance
        feature_importance = list(zip(available_features, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"✅ {model_key}: {accuracy:.1%} accuracy on real matches")
        
        return {
            'model_key': model_key,
            'model_name': model_config['name'],
            'accuracy': accuracy,
            'num_features': len(available_features),
            'results': results,
            'feature_importance': feature_importance,
            'mean_confidence': results['Confidence'].mean()
        }
    
    def run_comprehensive_comparison(self):
        """Run complete comparison of all models on real 2025-26 data."""
        
        logger.info("🚀 Starting comprehensive all-models comparison on real 2025-26 data...")
        
        # Load real season data
        df_real = self.load_real_season_data()
        
        # Test all models
        model_results = {}
        
        for model_key in self.models_config.keys():
            available_features = self.prepare_features(df_real, model_key)
            result = self.train_and_test_model(model_key, df_real, available_features)
            model_results[model_key] = result
        
        return model_results, df_real
    
    def generate_comprehensive_report(self, model_results, df_real):
        """Generate comprehensive comparison report."""
        
        # Sort models by accuracy
        sorted_models = sorted(model_results.values(), key=lambda x: x['accuracy'], reverse=True)
        
        print("\\n" + "="*120)
        print("🏆 COMPREHENSIVE ALL MODELS - REAL PREMIER LEAGUE 2025-26 SEASON TEST")
        print("="*120)
        print(f"Real matches tested: {len(df_real)} matches from first 2 matchweeks")
        print(f"Season period: {df_real['Date'].min().strftime('%B %d, %Y')} to {df_real['Date'].max().strftime('%B %d, %Y')}")
        
        print(f"\\n📊 OVERALL PERFORMANCE RANKING:")
        print(f"{'Rank':<5} {'Model':<15} {'Description':<45} {'Accuracy':<10} {'Features':<10} {'Confidence':<12} {'Status'}")
        print("-" * 120)
        
        for i, result in enumerate(sorted_models, 1):
            model_key = result['model_key']
            description = result['model_name'][:43] + "..." if len(result['model_name']) > 45 else result['model_name']
            accuracy = f"{result['accuracy']:.1%}"
            features = str(result['num_features'])
            confidence = f"{result['mean_confidence']:.1%}"
            
            if result['accuracy'] >= 0.55:
                status = "🚀 EXCEPTIONAL"
            elif result['accuracy'] >= 0.50:
                status = "✅ EXCELLENT" 
            elif result['accuracy'] >= 0.40:
                status = "📊 GOOD"
            else:
                status = "⚠️ NEEDS WORK"
            
            print(f"{i:<5} {model_key:<15} {description:<45} {accuracy:<10} {features:<10} {confidence:<12} {status}")
        
        champion = sorted_models[0]
        print(f"\\n🏆 CHAMPION ON REAL DATA: {champion['model_key'].upper()}")
        print(f"   • Description: {champion['model_name']}")
        print(f"   • Real season accuracy: {champion['accuracy']:.1%}")
        print(f"   • Features used: {champion['num_features']}")
        print(f"   • Mean confidence: {champion['mean_confidence']:.1%}")
        
        print(f"\\n⭐ TOP 5 FEATURES (Champion {champion['model_key']}):")
        for i, (feature, importance) in enumerate(champion['feature_importance'][:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")
        
        # Accuracy by result type comparison
        print(f"\\n🎯 ACCURACY BY RESULT TYPE COMPARISON:")
        print(f"{'Model':<15} {'Home Wins':<12} {'Draws':<12} {'Away Wins':<12}")
        print("-" * 55)
        
        for result in sorted_models:
            model_results_df = result['results']
            
            # Calculate accuracy by result type
            home_accuracy = 0
            draw_accuracy = 0  
            away_accuracy = 0
            
            home_matches = model_results_df[model_results_df['FullTimeResult'] == 'H']
            if len(home_matches) > 0:
                home_accuracy = home_matches['Correct'].mean()
                
            draw_matches = model_results_df[model_results_df['FullTimeResult'] == 'D']
            if len(draw_matches) > 0:
                draw_accuracy = draw_matches['Correct'].mean()
                
            away_matches = model_results_df[model_results_df['FullTimeResult'] == 'A']
            if len(away_matches) > 0:
                away_accuracy = away_matches['Correct'].mean()
            
            model_key = result['model_key']
            print(f"{model_key:<15} {home_accuracy:.1%}        {draw_accuracy:.1%}        {away_accuracy:.1%}")
        
        # Prediction distribution comparison
        actual_dist = df_real['FullTimeResult'].value_counts()
        
        print(f"\\n📊 PREDICTION DISTRIBUTION COMPARISON:")
        print(f"Actual distribution: H:{actual_dist.get('H',0)} ({actual_dist.get('H',0)/len(df_real)*100:.1f}%), " +
              f"D:{actual_dist.get('D',0)} ({actual_dist.get('D',0)/len(df_real)*100:.1f}%), " +
              f"A:{actual_dist.get('A',0)} ({actual_dist.get('A',0)/len(df_real)*100:.1f}%)")
        
        print(f"{'Model':<15} {'Pred H':<8} {'Pred D':<8} {'Pred A':<8} {'Bias Analysis'}")
        print("-" * 70)
        
        for result in sorted_models:
            model_results_df = result['results']
            predicted_dist = model_results_df['Predicted'].value_counts()
            
            pred_h = predicted_dist.get('H', 0)
            pred_d = predicted_dist.get('D', 0)
            pred_a = predicted_dist.get('A', 0)
            
            # Calculate bias
            actual_h = actual_dist.get('H', 0)
            bias = pred_a - actual_dist.get('A', 0)
            bias_str = f"Away{bias:+d}" if bias != 0 else "Balanced"
            
            model_key = result['model_key']
            print(f"{model_key:<15} {pred_h:<8} {pred_d:<8} {pred_a:<8} {bias_str}")
        
        # Key match analysis
        print(f"\\n🔥 KEY MATCH ANALYSIS:")
        
        key_matches = [
            ('Liverpool', 'Bournemouth'),  # Season opener
            ('Arsenal', 'Leeds United'),   # Big win
            ('Manchester City', 'Leeds United')  # City performance
        ]
        
        for home_team, away_team in key_matches:
            match_data = df_real[(df_real['HomeTeam'] == home_team) & (df_real['AwayTeam'] == away_team)]
            if len(match_data) > 0:
                match = match_data.iloc[0]
                actual_result = match['FullTimeResult']
                score = f"{match['FTHG']}-{match['FTAG']}"
                
                print(f"\\n   {home_team} vs {away_team} (Actual: {actual_result}, Score: {score}):")
                
                for result in sorted_models:
                    model_results_df = result['results']
                    model_match = model_results_df[(model_results_df['HomeTeam'] == home_team) & 
                                                  (model_results_df['AwayTeam'] == away_team)]
                    if len(model_match) > 0:
                        mm = model_match.iloc[0]
                        predicted = mm['Predicted']
                        confidence = mm['Confidence']
                        status = "✅" if mm['Correct'] else "❌"
                        
                        model_key = result['model_key']
                        print(f"   • {model_key}: {predicted} ({confidence:.1%}) {status}")
        
        # Benchmark comparison
        print(f"\\n🎯 BENCHMARK COMPARISON:")
        print(f"{'Model':<15} {'Accuracy':<10} {'vs Random':<12} {'vs Home':<12} {'vs Majority'}")
        print("-" * 65)
        
        random_baseline = 0.333
        home_bias_accuracy = actual_dist.get('H', 0) / len(df_real)
        majority_class_accuracy = max(actual_dist.values()) / len(df_real)
        
        for result in sorted_models:
            accuracy = result['accuracy']
            vs_random = f"{(accuracy - random_baseline)*100:+.1f}pp"
            vs_home = f"{(accuracy - home_bias_accuracy)*100:+.1f}pp" 
            vs_majority = f"{(accuracy - majority_class_accuracy)*100:+.1f}pp"
            
            model_key = result['model_key']
            print(f"{model_key:<15} {accuracy:.1%}      {vs_random:<12} {vs_home:<12} {vs_majority}")
        
        # Final recommendations
        print(f"\\n📋 FINAL RECOMMENDATIONS:")
        champion_accuracy = champion['accuracy']
        
        if champion_accuracy >= 0.60:
            print(f"   🚀 EXCEPTIONAL: Model significantly outperforms on real season data")
            print(f"   📈 Validated for production use with high confidence")
        elif champion_accuracy >= 0.50:
            print(f"   ✅ EXCELLENT: Strong real-world performance validates model")
            print(f"   📊 Ready for deployment with realistic expectations")
        elif champion_accuracy >= 0.40:
            print(f"   📊 GOOD: Positive predictive value on real matches")
            print(f"   🔧 Consider ensemble or feature refinement")
        else:
            print(f"   ⚠️ NEEDS IMPROVEMENT: Performance below expectations on real data")
            print(f"   🔍 Investigate model-reality gap")
        
        # Save detailed results
        for model_key, result in model_results.items():
            result['results'].to_csv(f'data/predictions/{model_key}_real_2025_26_detailed.csv', index=False)
            logger.info(f"Saved {model_key} detailed results to data/predictions/{model_key}_real_2025_26_detailed.csv")
        
        return sorted_models

def main():
    """Main testing workflow."""
    
    logger.info("🚀 Starting comprehensive real season model testing...")
    
    # Initialize tester
    tester = RealSeasonModelTester()
    
    # Run comparison
    model_results, df_real = tester.run_comprehensive_comparison()
    
    # Generate report
    sorted_results = tester.generate_comprehensive_report(model_results, df_real)
    
    logger.info("✅ Real season testing complete!")
    
    return model_results, sorted_results

if __name__ == "__main__":
    results, sorted_results = main()