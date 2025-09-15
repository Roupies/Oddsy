#!/usr/bin/env python3
"""
Comprehensive Model Comparison - All Best Models Tested
Test all major model versions with corrected methodology on complete 2280 matches

Strategy: Fair comparison using identical temporal splits and validation methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveModelComparison:
    """Compare all major model versions with consistent methodology."""
    
    def __init__(self):
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Define all models to test (FIXED COMPLETE VERSIONS)
        self.models_to_test = {
            'v13_corrected': {
                'path': 'data/processed/v13_xg_corrected_features_fixed_complete.csv',
                'description': 'v2.3 Corrected xG Integration (Former Production)',
                'features': [
                    'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10',
                    'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
                    'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
                ]
            },
            'v31_efficiency': {
                'path': 'data/processed/v31_efficiency_features_fixed_complete.csv',
                'description': 'v3.1 Efficiency Breakthrough (Moneyball xG)',
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
                'path': 'data/processed/v40_fatigue_features_fixed_complete.csv',
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
            'v41_referee_fixed': {
                'path': 'data/processed/v41_referee_features_fixed_2025_09_07.csv',
                'description': 'v4.1 Referee Intelligence (Official Decision Patterns)',
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
    
    def load_and_prepare_data(self, dataset_path, features):
        """Load and prepare dataset for testing."""
        
        if not Path(dataset_path).exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            return None, None
        
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {dataset_path}: {df.shape}")
        
        # Filter to available features
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Clean data
        df_clean = df.dropna(subset=available_features + ['FullTimeResult'])
        
        if len(df_clean) < len(df):
            logger.error(f"CRITICAL: Data loss detected: {len(df) - len(df_clean)} matches dropped!")
            # This should not happen with fixed datasets
        
        # Sort by date for temporal integrity
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        
        return df_clean, available_features
    
    def test_model(self, dataset_path, features, description):
        """Test a single model version."""
        
        logger.info(f"Testing: {description}")
        
        # Load data
        df, available_features = self.load_and_prepare_data(dataset_path, features)
        
        if df is None or len(available_features) < 5:
            return {
                'description': description,
                'status': 'FAILED',
                'reason': 'Data loading failed or insufficient features',
                'accuracy': 0.0,
                'total_matches': 0,
                'features_count': 0
            }
        
        # Temporal split: Train on 5 seasons (1900), Test on last season (380) - IDENTICAL FOR ALL
        test_season_size = 380  # Exactly 380 for all datasets (complete seasons)
        
        df_train = df[:-test_season_size]
        df_test = df[-test_season_size:]
        
        X_train = df_train[available_features]
        X_test = df_test[available_features]
        
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y_train = df_train['FullTimeResult'].map(target_mapping)
        y_test = df_test['FullTimeResult'].map(target_mapping)
        
        # Train model
        model = RandomForestClassifier(**self.model_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Cross-validation for stability
        tscv = TimeSeriesSplit(n_splits=5)
        X_full = df[available_features]
        y_full = df['FullTimeResult'].map(target_mapping)
        cv_scores = cross_val_score(model, X_full, y_full, cv=tscv, scoring='accuracy')
        
        # Feature importance
        feature_importance = list(zip(available_features, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Test date range for verification
        test_start = df_test['Date'].min()
        test_end = df_test['Date'].max()
        
        return {
            'description': description,
            'status': 'SUCCESS',
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'total_matches': len(df),
            'train_matches': len(df_train),
            'test_matches': len(df_test),
            'features_count': len(available_features),
            'top_features': feature_importance[:10],
            'test_period': f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
        }
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison of all models."""
        
        logger.info("🚀 Starting comprehensive model comparison...")
        
        results = {}
        
        for model_name, model_config in self.models_to_test.items():
            try:
                result = self.test_model(
                    model_config['path'], 
                    model_config['features'], 
                    model_config['description']
                )
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Error testing {model_name}: {str(e)}")
                results[model_name] = {
                    'description': model_config['description'],
                    'status': 'ERROR',
                    'reason': str(e),
                    'accuracy': 0.0
                }
        
        # Generate comparison report
        self.generate_comparison_report(results)
        
        return results
    
    def generate_comparison_report(self, results):
        """Generate comprehensive comparison report."""
        
        print("\\n" + "="*100)
        print("🏆 COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*100)
        
        # Sort by accuracy
        successful_results = {k: v for k, v in results.items() if v['status'] == 'SUCCESS'}
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\\n📊 PERFORMANCE RANKING:")
        print(f"{'Rank':<4} {'Model':<20} {'Accuracy':<10} {'CV Score':<12} {'Matches':<8} {'Features':<9} {'Status'}")
        print("-" * 100)
        
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            accuracy = result['accuracy']
            cv_score = f"{result['cv_mean']:.3f}±{result['cv_std']:.3f}"
            matches = result['test_matches']
            features = result['features_count']
            
            # Performance tier
            if accuracy >= 0.55:
                tier = "🚀 EXCELLENT"
            elif accuracy >= 0.52:
                tier = "✅ GOOD"
            else:
                tier = "📊 BASELINE"
            
            print(f"{rank:<4} {model_name:<20} {accuracy:.4f} {cv_score:<12} {matches:<8} {features:<9} {tier}")
        
        # Failed models
        failed_results = {k: v for k, v in results.items() if v['status'] != 'SUCCESS'}
        if failed_results:
            print(f"\\n❌ FAILED MODELS:")
            for model_name, result in failed_results.items():
                print(f"   • {model_name}: {result['reason']}")
        
        # Best model analysis
        if sorted_results:
            best_model_name, best_result = sorted_results[0]
            
            print(f"\\n🏆 BEST PERFORMER: {best_model_name.upper()}")
            print(f"   • Description: {best_result['description']}")
            print(f"   • Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
            print(f"   • F1-Macro: {best_result['f1_macro']:.3f}")
            print(f"   • CV Stability: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
            print(f"   • Test Period: {best_result['test_period']}")
            print(f"   • Features: {best_result['features_count']}")
            
            print(f"\\n⭐ TOP 10 FEATURES ({best_model_name}):")
            for i, (feature, importance) in enumerate(best_result['top_features'], 1):
                print(f"   {i:2d}. {feature}: {importance:.3f}")
        
        # Benchmark analysis
        print(f"\\n🎯 BENCHMARK ANALYSIS:")
        baselines = {
            'Random': 0.333,
            'Majority Class': 0.436,
            'Good Target': 0.520,
            'Excellent Target': 0.550,
            'Elite Target': 0.600
        }
        
        if sorted_results:
            best_accuracy = sorted_results[0][1]['accuracy']
            print(f"   Best model performance: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            
            for baseline_name, baseline_score in baselines.items():
                diff = (best_accuracy - baseline_score) * 100
                status = "✅" if best_accuracy > baseline_score else "❌"
                print(f"   {status} vs {baseline_name} ({baseline_score:.1%}): {diff:+.2f}pp")
        
        print(f"\\n📋 METHODOLOGY VALIDATION:")
        print(f"   • Temporal splits: Train on 5 seasons, Test on most recent season")
        print(f"   • Model consistency: Same RandomForest parameters across all tests")
        print(f"   • Data integrity: Complete 2280 matches where available")
        print(f"   • Fair comparison: Identical validation methodology")

if __name__ == "__main__":
    comparator = ComprehensiveModelComparison()
    results = comparator.run_comprehensive_comparison()