#!/usr/bin/env python3
"""
All Models Complete Comparison - 30 Matches Across 3 Matchdays
Test ALL major models (v2.3, v3.1, v4.0, v4.1) on identical 30 matches for comprehensive comparison

Strategy: Fair head-to-head comparison using identical data and methodology
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AllModelsComparator:
    """Compare all major model versions on identical 30-match dataset."""
    
    def __init__(self):
        # Consistent model parameters for fair comparison
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Define all models with their specific features
        self.models_config = {
            'v23_corrected': {
                'dataset': 'data/processed/v13_xg_corrected_features_fixed_complete.csv',
                'description': 'v2.3 Corrected xG Integration (Champion)',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10',
                    'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
                    'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
                ]
            },
            'v31_efficiency': {
                'dataset': 'data/processed/v31_efficiency_features_fixed_complete.csv',
                'description': 'v3.1 Efficiency Breakthrough (Moneyball)',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                    'corners_diff_normalized', 'goalkeeping_advantage_10',
                    'away_goalkeeping_efficiency_10_normalized', 'goalkeeping_advantage_10_normalized',
                    'net_performance_advantage_10_normalized', 'net_performance_advantage_10',
                    'goalkeeping_advantage_5_normalized', 'away_xg_eff_10', 'matchday_normalized',
                    'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
                ]
            },
            'v40_fatigue': {
                'dataset': 'data/processed/v40_fatigue_features_fixed_complete.csv',
                'description': 'v4.0 Fatigue Features (Fixture Congestion)',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                    'corners_diff_normalized', 'form_diff_normalized', 'home_xg_eff_10',
                    'away_xg_eff_10', 'h2h_score', 'matchday_normalized', 'away_goals_sum_5',
                    'goalkeeping_advantage_10_normalized', 'away_goalkeeping_efficiency_10_normalized',
                    'goalkeeping_advantage_10', 'net_performance_advantage_10',
                    'net_performance_advantage_10_normalized', 'fatigue_advantage', 
                    'home_days_since_last_match', 'away_days_since_last_match', 
                    'fixture_density_differential'
                ]
            },
            'v41_referee': {
                'dataset': 'data/processed/v41_referee_features_fixed_2025_09_07.csv',
                'description': 'v4.1 Referee Intelligence (Official Patterns)',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                    'corners_diff_normalized', 'form_diff_normalized', 'home_xg_eff_10',
                    'away_xg_eff_10', 'h2h_score', 'matchday_normalized', 'away_goals_sum_5',
                    'goalkeeping_advantage_10_normalized', 'away_goalkeeping_efficiency_10_normalized',
                    'goalkeeping_advantage_10', 'net_performance_advantage_10',
                    'net_performance_advantage_10_normalized', 'fatigue_advantage', 
                    'home_days_since_last_match', 'away_days_since_last_match', 
                    'fixture_density_differential', 'referee_bias_index_weighted',
                    'referee_home_bias_index', 'referee_disciplinary_index', 
                    'referee_home_impact_score', 'referee_experience_factor'
                ]
            }
        }
        
        self.prediction_data = None
        self.results_comparison = {}
    
    def load_prediction_data(self):
        """Load the 30-match prediction dataset."""
        
        logger.info("Loading 30-match prediction dataset...")
        
        df = pd.read_csv('data/processed/premier_league_2025_26_first_3_matchdays_complete.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info(f"Loaded prediction data: {len(df)} matches across {df['MatchWeek'].nunique()} matchdays")
        
        self.prediction_data = df
        return df
    
    def prepare_features_for_model(self, model_name, model_config):
        """Prepare features from prediction data to match model's expected features."""
        
        logger.info(f"Preparing features for {model_name}...")
        
        # Get expected features for this model
        expected_features = model_config['features']
        
        # Map features from our prediction data to model's expected features
        feature_mapping = {}
        available_features = []
        
        for feature in expected_features:
            if feature in self.prediction_data.columns:
                feature_mapping[feature] = feature
                available_features.append(feature)
            else:
                # Try to create missing features with reasonable defaults
                if 'referee' in feature:
                    # Referee features - use neutral values for early season
                    if 'bias' in feature:
                        self.prediction_data[feature] = 1.0  # Neutral bias
                    elif 'disciplinary' in feature:
                        self.prediction_data[feature] = 1.0  # Average disciplinary
                    elif 'experience' in feature:
                        self.prediction_data[feature] = 0.5  # Moderate experience
                    else:
                        self.prediction_data[feature] = 0.5  # Neutral default
                
                elif 'fatigue' in feature:
                    # Fatigue features - minimal fatigue at start of season
                    if 'days_since' in feature:
                        self.prediction_data[feature] = 14.0  # 2 weeks rest
                    elif 'density' in feature:
                        self.prediction_data[feature] = 0.1  # Low congestion
                    else:
                        self.prediction_data[feature] = 0.0  # No fatigue advantage
                
                elif 'goalkeeping' in feature or 'net_performance' in feature:
                    # Efficiency features - use neutral values
                    self.prediction_data[feature] = 0.0 if 'normalized' in feature else 1.0
                
                else:
                    # Other missing features - use neutral/average values
                    self.prediction_data[feature] = 0.5
                
                available_features.append(feature)
                logger.debug(f"Created missing feature {feature} with default values")
        
        logger.info(f"Prepared {len(available_features)}/{len(expected_features)} features for {model_name}")
        
        return available_features
    
    def test_single_model(self, model_name, model_config):
        """Test a single model on the 30-match dataset."""
        
        logger.info(f"Testing {model_name}...")
        
        try:
            # Load historical training data
            df_historical = pd.read_csv(model_config['dataset'])
            df_historical['Date'] = pd.to_datetime(df_historical['Date'])
            
            # Prepare features for this model
            available_features = self.prepare_features_for_model(model_name, model_config)
            
            if len(available_features) < 5:
                logger.error(f"Insufficient features for {model_name}: {len(available_features)}")
                return None
            
            # Prepare training data
            X_train = df_historical[available_features]
            target_mapping = {'H': 0, 'D': 1, 'A': 2}
            y_train = df_historical['FullTimeResult'].map(target_mapping)
            
            # Remove missing values
            valid_mask = y_train.notna() & X_train.notna().all(axis=1)
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            
            logger.info(f"{model_name} training data: {len(X_train)} matches")
            
            # Train model
            model = RandomForestClassifier(**self.model_params)
            model.fit(X_train, y_train)
            
            # Prepare prediction data
            X_predict = self.prediction_data[available_features]
            
            # Generate predictions
            predictions = model.predict(X_predict)
            probabilities = model.predict_proba(X_predict)
            
            # Convert predictions to labels
            label_mapping = {0: 'H', 1: 'D', 2: 'A'}
            predicted_labels = [label_mapping[pred] for pred in predictions]
            
            # Create results
            results = self.prediction_data[['MatchWeek', 'Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult']].copy()
            results['Predicted'] = predicted_labels
            results['Prob_Home'] = probabilities[:, 0]
            results['Prob_Draw'] = probabilities[:, 1]
            results['Prob_Away'] = probabilities[:, 2]
            results['Confidence'] = np.max(probabilities, axis=1)
            results['Correct'] = (results['FullTimeResult'] == results['Predicted'])
            
            # Calculate metrics
            accuracy = results['Correct'].mean()
            
            # Feature importance
            feature_importance = list(zip(available_features, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Accuracy by result type
            result_accuracy = {}
            for result_type in ['H', 'D', 'A']:
                type_results = results[results['FullTimeResult'] == result_type]
                if len(type_results) > 0:
                    type_accuracy = type_results['Correct'].sum() / len(type_results)
                    result_accuracy[result_type] = {
                        'count': len(type_results),
                        'correct': type_results['Correct'].sum(),
                        'accuracy': type_accuracy
                    }
            
            # Prediction distribution
            pred_dist = results['Predicted'].value_counts()
            
            model_results = {
                'model_name': model_name,
                'description': model_config['description'],
                'accuracy': accuracy,
                'total_features': len(available_features),
                'feature_importance': feature_importance,
                'result_accuracy': result_accuracy,
                'prediction_distribution': pred_dist,
                'detailed_results': results,
                'mean_confidence': results['Confidence'].mean(),
                'status': 'SUCCESS'
            }
            
            logger.info(f"✅ {model_name}: {accuracy:.1%} accuracy")
            
            return model_results
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {str(e)}")
            return {
                'model_name': model_name,
                'description': model_config['description'],
                'accuracy': 0.0,
                'status': 'FAILED',
                'error': str(e)
            }
    
    def test_all_models(self):
        """Test all models on the 30-match dataset."""
        
        logger.info("🚀 Testing all models on 30-match dataset...")
        
        # Load prediction data
        self.load_prediction_data()
        
        # Test each model
        for model_name, model_config in self.models_config.items():
            result = self.test_single_model(model_name, model_config)
            if result:
                self.results_comparison[model_name] = result
        
        return self.results_comparison
    
    def generate_comprehensive_comparison_report(self):
        """Generate comprehensive comparison report for all models."""
        
        # Filter successful models and sort by accuracy
        successful_models = {k: v for k, v in self.results_comparison.items() if v['status'] == 'SUCCESS'}
        sorted_models = sorted(successful_models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\\n" + "="*120)
        print("🏆 COMPREHENSIVE ALL MODELS COMPARISON - 30 MATCHES ACROSS 3 MATCHDAYS")
        print("="*120)
        
        print(f"\\n📊 OVERALL PERFORMANCE RANKING:")
        print(f"{'Rank':<4} {'Model':<15} {'Description':<35} {'Accuracy':<10} {'Features':<9} {'Confidence':<11} {'Status'}")
        print("-" * 120)
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            accuracy = results['accuracy']
            features = results['total_features']
            confidence = results['mean_confidence']
            
            # Performance tier
            if accuracy >= 0.60:
                tier = "🚀 EXCEPTIONAL"
            elif accuracy >= 0.50:
                tier = "✅ EXCELLENT"
            elif accuracy >= 0.40:
                tier = "📊 GOOD"
            else:
                tier = "⚠️ NEEDS WORK"
            
            print(f"{rank:<4} {model_name:<15} {results['description'][:34]:<35} {accuracy:.1%}  {features:<9} {confidence:.1%}     {tier}")
        
        # Best performer analysis
        if sorted_models:
            best_model_name, best_results = sorted_models[0]
            
            print(f"\\n🏆 CHAMPION: {best_model_name.upper()}")
            print(f"   • Description: {best_results['description']}")
            print(f"   • Accuracy: {best_results['accuracy']:.1%}")
            print(f"   • Features: {best_results['total_features']}")
            print(f"   • Mean confidence: {best_results['mean_confidence']:.1%}")
            
            print(f"\\n⭐ TOP 5 FEATURES ({best_model_name}):")
            for i, (feature, importance) in enumerate(best_results['feature_importance'][:5], 1):
                print(f"   {i}. {feature}: {importance:.3f}")
        
        # Accuracy by result type comparison
        print(f"\\n🎯 ACCURACY BY RESULT TYPE COMPARISON:")
        print(f"{'Model':<15} {'Home Wins':<12} {'Draws':<12} {'Away Wins':<12}")
        print("-" * 55)
        
        for model_name, results in sorted_models:
            home_acc = results['result_accuracy'].get('H', {}).get('accuracy', 0)
            draw_acc = results['result_accuracy'].get('D', {}).get('accuracy', 0)
            away_acc = results['result_accuracy'].get('A', {}).get('accuracy', 0)
            
            print(f"{model_name:<15} {home_acc:.1%}      {draw_acc:.1%}      {away_acc:.1%}")
        
        # Prediction distribution comparison
        print(f"\\n📊 PREDICTION DISTRIBUTION COMPARISON:")
        print(f"Actual distribution: H:17 (56.7%), D:4 (13.3%), A:9 (30.0%)")
        print(f"{'Model':<15} {'Pred H':<8} {'Pred D':<8} {'Pred A':<8} {'Bias Analysis'}")
        print("-" * 70)
        
        for model_name, results in sorted_models:
            pred_dist = results['prediction_distribution']
            pred_h = pred_dist.get('H', 0)
            pred_d = pred_dist.get('D', 0)
            pred_a = pred_dist.get('A', 0)
            
            # Bias analysis
            home_bias = pred_h - 17
            away_bias = pred_a - 9
            
            if abs(home_bias) > abs(away_bias):
                bias = f"{'Home+' if home_bias > 0 else 'Home-'}{abs(home_bias)}"
            else:
                bias = f"{'Away+' if away_bias > 0 else 'Away-'}{abs(away_bias)}"
            
            print(f"{model_name:<15} {pred_h:<8} {pred_d:<8} {pred_a:<8} {bias}")
        
        # Head-to-head match comparison for key matches
        print(f"\\n🔥 HEAD-TO-HEAD COMPARISON (Key Matches):")
        key_matches = [
            "Arsenal vs Tottenham",  # North London Derby
            "Liverpool vs Manchester City",  # Title clash
            "Chelsea vs Manchester United"   # Big 6 clash
        ]
        
        for match in key_matches:
            match_found = False
            for model_name, results in sorted_models:
                match_results = results['detailed_results']
                for _, row in match_results.iterrows():
                    match_name = f"{row['HomeTeam']} vs {row['AwayTeam']}"
                    if match in match_name:
                        if not match_found:
                            print(f"\\n   {match_name} (Actual: {row['FullTimeResult']}):")
                            match_found = True
                        
                        status = "✅" if row['Correct'] else "❌"
                        print(f"   • {model_name}: {row['Predicted']} ({row['Confidence']:.1%}) {status}")
                        break
        
        # Benchmark comparison
        print(f"\\n🎯 BENCHMARK COMPARISON:")
        actual_dist = {'H': 17, 'D': 4, 'A': 9}
        benchmarks = {
            'Random': 1/3,
            'Home Bias': 17/30,  # Always predict most common result
            'Majority Class': max(actual_dist.values())/30
        }
        
        print(f"{'Model':<15} {'Accuracy':<10} {'vs Random':<10} {'vs Home':<10} {'vs Majority':<12}")
        print("-" * 60)
        
        for model_name, results in sorted_models:
            accuracy = results['accuracy']
            vs_random = (accuracy - benchmarks['Random']) * 100
            vs_home = (accuracy - benchmarks['Home Bias']) * 100
            vs_majority = (accuracy - benchmarks['Majority Class']) * 100
            
            print(f"{model_name:<15} {accuracy:.1%}    {vs_random:+.1f}pp    {vs_home:+.1f}pp    {vs_majority:+.1f}pp")
        
        # Final recommendations
        print(f"\\n📋 FINAL RECOMMENDATIONS:")
        
        if sorted_models:
            best_accuracy = sorted_models[0][1]['accuracy']
            
            if best_accuracy >= 0.60:
                print(f"   🚀 DEPLOY BEST MODEL: Exceptional performance justifies immediate deployment")
            elif best_accuracy >= 0.50:
                print(f"   ✅ PRODUCTION READY: Strong performance validates model architecture")
            elif best_accuracy >= 0.40:
                print(f"   📊 FURTHER DEVELOPMENT: Positive signals but needs improvement")
            else:
                print(f"   ⚠️ BACK TO DRAWING BOARD: Performance below acceptable thresholds")
            
            # Architecture insights
            feature_counts = [results['total_features'] for _, results in sorted_models]
            accuracies = [results['accuracy'] for _, results in sorted_models]
            
            if sorted_models[0][1]['total_features'] == min(feature_counts):
                print(f"   🎯 SIMPLICITY WINS: Fewest features = Best performance")
            elif sorted_models[0][1]['total_features'] == max(feature_counts):
                print(f"   📈 COMPLEXITY PAYS: Most features = Best performance")
            else:
                print(f"   ⚖️ BALANCED APPROACH: Moderate feature count optimal")
        
        # Failed models
        failed_models = {k: v for k, v in self.results_comparison.items() if v['status'] == 'FAILED'}
        if failed_models:
            print(f"\\n❌ FAILED MODELS:")
            for model_name, results in failed_models.items():
                print(f"   • {model_name}: {results.get('error', 'Unknown error')}")

def main():
    """Main comparison workflow."""
    
    logger.info("🚀 Starting comprehensive all-models comparison...")
    
    # Initialize comparator
    comparator = AllModelsComparator()
    
    # Test all models
    results = comparator.test_all_models()
    
    # Generate comprehensive report
    comparator.generate_comprehensive_comparison_report()
    
    # Save detailed results
    for model_name, model_results in results.items():
        if model_results['status'] == 'SUCCESS':
            filename = f"data/predictions/{model_name}_30_matches_detailed.csv"
            model_results['detailed_results'].to_csv(filename, index=False)
            logger.info(f"Saved {model_name} detailed results to {filename}")
    
    logger.info("✅ Comprehensive comparison complete!")
    
    return results

if __name__ == "__main__":
    comparison_results = main()