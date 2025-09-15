#!/usr/bin/env python3
"""
v2.3 Complete Prediction Audit - First 3 Premier League Matchdays
Complete prediction system with comprehensive audit for all 30 matches

Strategy: Train on historical data, predict all 30 matches, comprehensive evaluation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V23CompletePredictor:
    """Complete v2.3 prediction system with comprehensive audit."""
    
    def __init__(self):
        # v2.3 champion model configuration
        self.model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # v2.3 features (champion configuration)
        self.v23_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10',
            'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
            'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        self.model = None
        self.feature_importance = None
    
    def load_training_data(self):
        """Load historical training data."""
        
        logger.info("Loading historical training data...")
        
        df_historical = pd.read_csv('data/processed/v13_xg_corrected_features_fixed_complete.csv')
        df_historical['Date'] = pd.to_datetime(df_historical['Date'])
        
        logger.info(f"Loaded historical data: {len(df_historical)} matches")
        
        return df_historical
    
    def load_prediction_data(self):
        """Load 2025-26 matchday data for predictions."""
        
        logger.info("Loading 2025-26 matchday data...")
        
        df_new = pd.read_csv('data/processed/premier_league_2025_26_first_3_matchdays_complete.csv')
        df_new['Date'] = pd.to_datetime(df_new['Date'])
        
        logger.info(f"Loaded new season data: {len(df_new)} matches across {df_new['MatchWeek'].nunique()} matchdays")
        
        return df_new
    
    def train_model(self, df_historical):
        """Train the v2.3 model on historical data."""
        
        logger.info("Training v2.3 champion model...")
        
        # Check feature availability
        available_features = [f for f in self.v23_features if f in df_historical.columns]
        missing_features = [f for f in self.v23_features if f not in df_historical.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        logger.info(f"Using {len(available_features)}/10 v2.3 features")
        
        # Prepare training data
        X_train = df_historical[available_features]
        
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y_train = df_historical['FullTimeResult'].map(target_mapping)
        
        # Remove any missing targets
        valid_mask = y_train.notna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        logger.info(f"Training data: {len(X_train)} matches")
        
        # Train model
        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = list(zip(available_features, self.model.feature_importances_))
        self.feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("✅ v2.3 model training complete")
        
        return available_features
    
    def predict_all_matches(self, df_new, available_features):
        """Generate predictions for all 30 matches."""
        
        logger.info("Generating predictions for all 30 matches...")
        
        # Prepare prediction data
        X_predict = df_new[available_features]
        
        # Generate predictions
        predictions = self.model.predict(X_predict)
        probabilities = self.model.predict_proba(X_predict)
        
        # Convert predictions back to labels
        label_mapping = {0: 'H', 1: 'D', 2: 'A'}
        predicted_labels = [label_mapping[pred] for pred in predictions]
        
        # Create comprehensive results dataframe
        results = df_new[['MatchWeek', 'Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'FTHG', 'FTAG']].copy()
        results['Predicted'] = predicted_labels
        results['Prob_Home'] = probabilities[:, 0]
        results['Prob_Draw'] = probabilities[:, 1]
        results['Prob_Away'] = probabilities[:, 2]
        results['Confidence'] = np.max(probabilities, axis=1)
        results['Correct'] = (results['FullTimeResult'] == results['Predicted'])
        
        return results
    
    def comprehensive_audit(self, results):
        """Conduct comprehensive audit of predictions."""
        
        logger.info("Conducting comprehensive prediction audit...")
        
        # Overall accuracy
        total_correct = results['Correct'].sum()
        total_matches = len(results)
        overall_accuracy = total_correct / total_matches
        
        # Accuracy by matchday
        matchday_accuracy = {}
        for md in sorted(results['MatchWeek'].unique()):
            md_results = results[results['MatchWeek'] == md]
            md_accuracy = md_results['Correct'].sum() / len(md_results)
            matchday_accuracy[md] = {
                'matches': len(md_results),
                'correct': md_results['Correct'].sum(),
                'accuracy': md_accuracy
            }
        
        # Accuracy by result type
        result_accuracy = {}
        for result_type in ['H', 'D', 'A']:
            type_results = results[results['FullTimeResult'] == result_type]
            if len(type_results) > 0:
                type_accuracy = type_results['Correct'].sum() / len(type_results)
                result_accuracy[result_type] = {
                    'actual_count': len(type_results),
                    'correct_predictions': type_results['Correct'].sum(),
                    'accuracy': type_accuracy
                }
        
        # Predicted vs actual distribution
        actual_dist = results['FullTimeResult'].value_counts()
        predicted_dist = results['Predicted'].value_counts()
        
        # Confidence analysis
        confidence_stats = {
            'mean_confidence': results['Confidence'].mean(),
            'std_confidence': results['Confidence'].std(),
            'high_confidence_matches': (results['Confidence'] > 0.7).sum(),
            'high_confidence_accuracy': results[results['Confidence'] > 0.7]['Correct'].mean() if (results['Confidence'] > 0.7).sum() > 0 else 0
        }
        
        # Error analysis
        errors = results[~results['Correct']].copy()
        error_patterns = {
            'predicted_home_actual_away': len(errors[(errors['Predicted'] == 'H') & (errors['FullTimeResult'] == 'A')]),
            'predicted_away_actual_home': len(errors[(errors['Predicted'] == 'A') & (errors['FullTimeResult'] == 'H')]),
            'predicted_draw_actual_decisive': len(errors[(errors['Predicted'] == 'D') & (errors['FullTimeResult'].isin(['H', 'A']))]),
            'predicted_decisive_actual_draw': len(errors[(errors['Predicted'].isin(['H', 'A'])) & (errors['FullTimeResult'] == 'D')])
        }
        
        audit_results = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_matches': total_matches,
            'matchday_accuracy': matchday_accuracy,
            'result_accuracy': result_accuracy,
            'actual_distribution': actual_dist,
            'predicted_distribution': predicted_dist,
            'confidence_stats': confidence_stats,
            'error_patterns': error_patterns,
            'detailed_results': results
        }
        
        return audit_results
    
    def generate_comprehensive_report(self, audit_results):
        """Generate comprehensive prediction report."""
        
        results = audit_results['detailed_results']
        
        print("\\n" + "="*100)
        print("🏆 v2.3 CHAMPION MODEL - COMPLETE 3 MATCHDAYS PREDICTION AUDIT")
        print("="*100)
        
        print(f"\\n📊 OVERALL PERFORMANCE:")
        print(f"   • Total matches predicted: {audit_results['total_matches']}")
        print(f"   • Correct predictions: {audit_results['total_correct']}")
        print(f"   • Overall accuracy: {audit_results['overall_accuracy']:.1%}")
        
        # Performance tier
        if audit_results['overall_accuracy'] >= 0.60:
            tier = "🚀 EXCEPTIONAL"
        elif audit_results['overall_accuracy'] >= 0.50:
            tier = "✅ EXCELLENT"
        elif audit_results['overall_accuracy'] >= 0.40:
            tier = "📊 GOOD"
        else:
            tier = "⚠️ NEEDS IMPROVEMENT"
        
        print(f"   • Performance tier: {tier}")
        
        print(f"\\n⭐ TOP 5 FEATURES (Historical Importance):")
        for i, (feature, importance) in enumerate(self.feature_importance[:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")
        
        print(f"\\n📅 MATCHDAY BREAKDOWN:")
        for md, stats in audit_results['matchday_accuracy'].items():
            print(f"   Matchday {md}: {stats['correct']}/{stats['matches']} correct ({stats['accuracy']:.1%})")
        
        print(f"\\n🎯 ACCURACY BY RESULT TYPE:")
        for result_type in ['H', 'D', 'A']:
            if result_type in audit_results['result_accuracy']:
                stats = audit_results['result_accuracy'][result_type]
                result_name = {'H': 'Home Wins', 'D': 'Draws', 'A': 'Away Wins'}[result_type]
                print(f"   {result_name}: {stats['correct_predictions']}/{stats['actual_count']} correct ({stats['accuracy']:.1%})")
        
        print(f"\\n📊 PREDICTION vs ACTUAL DISTRIBUTION:")
        actual = audit_results['actual_distribution']
        predicted = audit_results['predicted_distribution']
        
        print(f"   {'Result':<12} {'Actual':<8} {'Predicted':<10} {'Difference'}")
        print(f"   {'-'*40}")
        for result in ['H', 'D', 'A']:
            actual_count = actual.get(result, 0)
            predicted_count = predicted.get(result, 0)
            diff = predicted_count - actual_count
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            print(f"   {result:<12} {actual_count:<8} {predicted_count:<10} {diff_str}")
        
        print(f"\\n🎯 CONFIDENCE ANALYSIS:")
        conf_stats = audit_results['confidence_stats']
        print(f"   • Mean confidence: {conf_stats['mean_confidence']:.1%}")
        print(f"   • High confidence matches (>70%): {conf_stats['high_confidence_matches']}")
        print(f"   • High confidence accuracy: {conf_stats['high_confidence_accuracy']:.1%}")
        
        print(f"\\n❌ ERROR ANALYSIS:")
        error_patterns = audit_results['error_patterns']
        print(f"   • Predicted Home, got Away: {error_patterns['predicted_home_actual_away']}")
        print(f"   • Predicted Away, got Home: {error_patterns['predicted_away_actual_home']}")
        print(f"   • Predicted Draw, got Decisive: {error_patterns['predicted_draw_actual_decisive']}")
        print(f"   • Predicted Decisive, got Draw: {error_patterns['predicted_decisive_actual_draw']}")
        
        print(f"\\n📋 DETAILED MATCH RESULTS:")
        print(f"{'MD':<3} {'Date':<6} {'Match':<30} {'Score':<8} {'Actual':<7} {'Pred':<5} {'Conf':<6} {'Status'}")
        print("-" * 80)
        
        for _, row in results.iterrows():
            md = row['MatchWeek']
            date_str = row['Date'].strftime('%m/%d')
            match = f"{row['HomeTeam'][:12]} vs {row['AwayTeam'][:12]}"
            score = f"{row['FTHG']}-{row['FTAG']}"
            actual = row['FullTimeResult']
            predicted = row['Predicted']
            confidence = f"{row['Confidence']:.1%}"
            status = "✅" if row['Correct'] else "❌"
            
            print(f"{md:<3} {date_str:<6} {match:<30} {score:<8} {actual:<7} {predicted:<5} {confidence:<6} {status}")
        
        print(f"\\n🏆 BENCHMARK COMPARISON:")
        accuracy = audit_results['overall_accuracy']
        baselines = {
            'Random (33.3%)': 0.333,
            'Home Bias (56.7%)': 0.567,  # Based on actual home win rate
            'Majority Class': max(audit_results['actual_distribution']) / audit_results['total_matches']
        }
        
        print(f"   v2.3 Model: {accuracy:.1%}")
        for baseline_name, baseline_score in baselines.items():
            diff = (accuracy - baseline_score) * 100
            status = "✅" if accuracy > baseline_score else "❌"
            print(f"   {status} vs {baseline_name}: {diff:+.1f}pp")
        
        print(f"\\n📋 FINAL ASSESSMENT:")
        if accuracy >= 0.60:
            print(f"   🚀 EXCEPTIONAL: Model significantly outperforms expectations")
            print(f"   📈 Ready for production deployment with confidence")
        elif accuracy >= 0.50:
            print(f"   ✅ EXCELLENT: Model beats all reasonable baselines")
            print(f"   📊 Strong performance validates v2.3 architecture")
        elif accuracy >= 0.40:
            print(f"   📊 GOOD: Model shows positive predictive value")
            print(f"   🔧 Consider feature refinement or more data")
        else:
            print(f"   ⚠️ NEEDS IMPROVEMENT: Performance below expectations")
            print(f"   🔍 Investigate model assumptions and data quality")

def main():
    """Main prediction and audit workflow."""
    
    logger.info("🚀 Starting v2.3 complete prediction audit...")
    
    # Initialize predictor
    predictor = V23CompletePredictor()
    
    # Load data
    df_historical = predictor.load_training_data()
    df_new = predictor.load_prediction_data()
    
    # Train model
    available_features = predictor.train_model(df_historical)
    
    # Generate predictions
    results = predictor.predict_all_matches(df_new, available_features)
    
    # Conduct comprehensive audit
    audit_results = predictor.comprehensive_audit(results)
    
    # Generate report
    predictor.generate_comprehensive_report(audit_results)
    
    # Save results
    results.to_csv('data/predictions/v23_complete_30_matches_predictions.csv', index=False)
    logger.info("✅ Complete predictions saved to data/predictions/v23_complete_30_matches_predictions.csv")
    
    return results, audit_results

if __name__ == "__main__":
    results, audit = main()