#!/usr/bin/env python3
"""
Error Profile Quantitative Analysis for EPL 2025-26 Rolling Validation

Deep statistical analysis of established teams prediction errors to identify
precise failure patterns and guide anti-bias feature engineering.

Based on Gemini's expert suggestion for quantitative profiling vs manual analysis.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp
import warnings
warnings.filterwarnings('ignore')

class ErrorProfileAnalyzer:
    """
    Quantitative analysis of prediction errors for established teams
    """
    
    def __init__(self, report_file, dataset_file):
        self.report_file = report_file
        self.dataset_file = dataset_file
        self.report_data = None
        self.dataset = None
        self.all_predictions = []
        self.established_predictions = []
        self.promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
        
        # Features to analyze for error patterns
        self.key_features = [
            'elo_diff_normalized',
            'form_diff_normalized', 
            'market_entropy_norm',
            'home_xg_eff_10',
            'away_xg_eff_10',
            'shots_diff_normalized',
            'corners_diff_normalized',
            'h2h_score',
            'away_goals_sum_5'
        ]
        
    def load_data(self):
        """Load rolling validation report and dataset"""
        print("📊 Loading data for error profile analysis...")
        
        # Load validation report
        with open(self.report_file, 'r') as f:
            self.report_data = json.load(f)
        print(f"✅ Loaded validation report: {len(self.report_data['all_predictions'])} predictions")
        
        # Load dataset for features
        self.dataset = pd.read_csv(self.dataset_file)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        print(f"✅ Loaded dataset: {len(self.dataset)} matches")
        
        # Process predictions
        self.all_predictions = self.report_data['all_predictions']
        
        # Filter established teams predictions only
        self.established_predictions = [
            p for p in self.all_predictions 
            if not p.get('involves_promoted', False)
        ]
        
        print(f"📈 Analysis scope:")
        print(f"   Total predictions: {len(self.all_predictions)}")
        print(f"   Established teams: {len(self.established_predictions)}")
        print(f"   Promoted teams: {len(self.all_predictions) - len(self.established_predictions)}")
        
    def create_features_dataframe(self):
        """Create DataFrame with predictions and corresponding features"""
        
        # Extract prediction data
        pred_data = []
        
        for pred in self.established_predictions:
            # Find corresponding match in dataset
            match_date = pd.to_datetime(pred['date']).date()
            home_team = pred['home_team']
            away_team = pred['away_team']
            
            # Search in EPL 2025-26 data
            dataset_match = self.dataset[
                (self.dataset['Date'].dt.date == match_date) &
                (self.dataset['HomeTeam'] == home_team) &
                (self.dataset['AwayTeam'] == away_team)
            ]
            
            if len(dataset_match) > 0:
                match_features = dataset_match.iloc[0]
                
                pred_record = {
                    'match_id': pred['match_id'],
                    'gameweek': pred['gameweek'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual_result': pred['actual_result'],
                    'predicted_result': pred['predicted_result'],
                    'correct': pred['correct'],
                    'confidence': pred['confidence'],
                    'prob_home': pred['predicted_probas']['H'],
                    'prob_draw': pred['predicted_probas']['D'],
                    'prob_away': pred['predicted_probas']['A']
                }
                
                # Add features
                for feature in self.key_features:
                    if feature in match_features:
                        pred_record[feature] = match_features[feature]
                    else:
                        pred_record[feature] = np.nan
                        
                pred_data.append(pred_record)
                
        self.predictions_df = pd.DataFrame(pred_data)
        print(f"✅ Created features DataFrame: {len(self.predictions_df)} matches with features")
        
        return self.predictions_df
        
    def analyze_error_distributions(self):
        """Compare feature distributions between correct and incorrect predictions"""
        
        if self.predictions_df is None:
            self.create_features_dataframe()
            
        print("\n🔍 QUANTITATIVE ERROR PROFILE ANALYSIS")
        print("="*60)
        
        # Split correct vs incorrect predictions
        correct_preds = self.predictions_df[self.predictions_df['correct'] == True]
        error_preds = self.predictions_df[self.predictions_df['correct'] == False]
        
        print(f"📊 Analysis scope:")
        print(f"   Correct predictions: {len(correct_preds)}")
        print(f"   Error predictions: {len(error_preds)}")
        
        # Analyze each feature
        analysis_results = {}
        
        print(f"\n📈 Feature Distribution Analysis:")
        print("-" * 80)
        print(f"{'Feature':<25} | {'Correct Mean':<12} | {'Error Mean':<11} | {'P-Value':<8} | {'Significant'}")
        print("-" * 80)
        
        for feature in self.key_features:
            if feature not in self.predictions_df.columns:
                continue
                
            # Remove NaN values
            correct_values = correct_preds[feature].dropna()
            error_values = error_preds[feature].dropna()
            
            if len(correct_values) == 0 or len(error_values) == 0:
                continue
                
            # Statistical test (Mann-Whitney U for non-parametric)
            try:
                statistic, p_value = mannwhitneyu(correct_values, error_values, 
                                                  alternative='two-sided')
                
                # Calculate descriptive stats
                correct_mean = correct_values.mean()
                error_mean = error_values.mean()
                
                # Check significance
                significant = "🟢 YES" if p_value < 0.05 else "🔴 NO"
                
                print(f"{feature:<25} | {correct_mean:<12.3f} | {error_mean:<11.3f} | {p_value:<8.3f} | {significant}")
                
                analysis_results[feature] = {
                    'correct_mean': correct_mean,
                    'correct_std': correct_values.std(),
                    'error_mean': error_mean,
                    'error_std': error_values.std(),
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(correct_mean - error_mean) / np.sqrt((correct_values.var() + error_values.var()) / 2)
                }
                
            except Exception as e:
                print(f"{feature:<25} | ERROR: {str(e)[:50]}")
                
        return analysis_results
        
    def identify_vulnerability_zones(self):
        """Identify specific ranges where model fails most often"""
        
        print(f"\n🎯 VULNERABILITY ZONES IDENTIFICATION")
        print("="*50)
        
        # Analyze errors by feature ranges
        error_preds = self.predictions_df[self.predictions_df['correct'] == False]
        
        vulnerability_zones = {}
        
        for feature in ['elo_diff_normalized', 'market_entropy_norm', 'form_diff_normalized']:
            if feature not in self.predictions_df.columns:
                continue
                
            print(f"\n📊 {feature} Vulnerability Analysis:")
            
            # Create bins for analysis
            all_values = self.predictions_df[feature].dropna()
            error_values = error_preds[feature].dropna()
            
            # Quartile analysis
            q1, q2, q3 = all_values.quantile([0.25, 0.5, 0.75])
            
            # Define ranges
            ranges = {
                'Low': (all_values.min(), q1),
                'Med-Low': (q1, q2), 
                'Med-High': (q2, q3),
                'High': (q3, all_values.max())
            }
            
            range_analysis = {}
            
            for range_name, (min_val, max_val) in ranges.items():
                # Count total and errors in this range
                in_range = self.predictions_df[
                    (self.predictions_df[feature] >= min_val) &
                    (self.predictions_df[feature] <= max_val)
                ]
                
                errors_in_range = in_range[in_range['correct'] == False]
                
                total_in_range = len(in_range)
                errors_count = len(errors_in_range)
                error_rate = errors_count / total_in_range if total_in_range > 0 else 0
                
                range_analysis[range_name] = {
                    'range': (min_val, max_val),
                    'total_matches': total_in_range,
                    'errors': errors_count,
                    'error_rate': error_rate
                }
                
                print(f"   {range_name:<10} [{min_val:.3f}, {max_val:.3f}]: {errors_count}/{total_in_range} ({error_rate:.1%} error rate)")
                
            vulnerability_zones[feature] = range_analysis
            
        return vulnerability_zones
        
    def analyze_multi_feature_patterns(self):
        """Identify combinations of features that lead to errors"""
        
        print(f"\n🔬 MULTI-FEATURE ERROR PATTERNS")
        print("="*40)
        
        error_preds = self.predictions_df[self.predictions_df['correct'] == False]
        
        # High-risk combinations
        risk_combinations = []
        
        # Pattern 1: Close matches with high uncertainty
        close_uncertain = error_preds[
            (error_preds['elo_diff_normalized'] >= 0.45) &
            (error_preds['elo_diff_normalized'] <= 0.55) &
            (error_preds['market_entropy_norm'] >= 0.8)
        ]
        
        if len(close_uncertain) > 0:
            risk_combinations.append({
                'pattern': 'Close Elo + High Market Uncertainty',
                'condition': 'elo_diff ∈ [0.45, 0.55] AND entropy >= 0.8',
                'matches': len(close_uncertain),
                'examples': close_uncertain[['home_team', 'away_team', 'actual_result', 'predicted_result']].head(3).to_dict('records')
            })
            
        # Pattern 2: Poor away form with overconfident home prediction
        poor_away = error_preds[
            (error_preds['away_xg_eff_10'] <= 0.3) &
            (error_preds['prob_home'] >= 0.6) &
            (error_preds['actual_result'] != 'H')
        ]
        
        if len(poor_away) > 0:
            risk_combinations.append({
                'pattern': 'Poor Away xG + Overconfident Home Prediction',
                'condition': 'away_xg_eff <= 0.3 AND prob_home >= 0.6 AND actual != H',
                'matches': len(poor_away),
                'examples': poor_away[['home_team', 'away_team', 'actual_result', 'predicted_result']].head(3).to_dict('records')
            })
            
        # Pattern 3: High confidence but wrong
        high_conf_wrong = error_preds[error_preds['confidence'] >= 0.65]
        
        if len(high_conf_wrong) > 0:
            risk_combinations.append({
                'pattern': 'High Confidence but Wrong',
                'condition': 'confidence >= 0.65 AND incorrect',
                'matches': len(high_conf_wrong),
                'examples': high_conf_wrong[['home_team', 'away_team', 'actual_result', 'predicted_result', 'confidence']].head(3).to_dict('records')
            })
            
        # Display patterns
        for i, pattern in enumerate(risk_combinations, 1):
            print(f"\n🚨 Pattern {i}: {pattern['pattern']}")
            print(f"   Condition: {pattern['condition']}")
            print(f"   Matches: {pattern['matches']}")
            
            if pattern['examples']:
                print("   Examples:")
                for j, example in enumerate(pattern['examples'], 1):
                    print(f"     {j}. {example['home_team']} vs {example['away_team']}: " +
                          f"Pred={example['predicted_result']}, Actual={example['actual_result']}")
                          
        return risk_combinations
        
    def generate_visualizations(self):
        """Create visualizations for error analysis"""
        
        print(f"\n📊 Generating error profile visualizations...")
        
        # Create output directory
        viz_dir = Path("results/error_analysis_viz")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data
        correct_preds = self.predictions_df[self.predictions_df['correct'] == True]
        error_preds = self.predictions_df[self.predictions_df['correct'] == False]
        
        # 1. Feature distributions comparison
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Feature Distributions: Correct vs Error Predictions', fontsize=16)
        
        for i, feature in enumerate(self.key_features[:9]):
            if feature not in self.predictions_df.columns:
                continue
                
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Plot distributions
            correct_vals = correct_preds[feature].dropna()
            error_vals = error_preds[feature].dropna()
            
            ax.hist(correct_vals, alpha=0.7, label=f'Correct ({len(correct_vals)})', 
                   color='green', bins=15)
            ax.hist(error_vals, alpha=0.7, label=f'Errors ({len(error_vals)})', 
                   color='red', bins=15)
            
            ax.set_title(feature)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_distributions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence vs Accuracy scatter
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by correctness
        colors = ['red' if not correct else 'green' for correct in self.predictions_df['correct']]
        
        scatter = ax.scatter(self.predictions_df['confidence'], 
                           self.predictions_df['correct'].astype(int),
                           c=colors, alpha=0.6, s=60)
        
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Correct (1) vs Incorrect (0)')
        ax.set_title('Confidence vs Accuracy for Established Teams')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.predictions_df['confidence'], 
                      self.predictions_df['correct'].astype(int), 1)
        p = np.poly1d(z)
        ax.plot(self.predictions_df['confidence'], p(self.predictions_df['confidence']), 
               "r--", alpha=0.8, linewidth=2)
        
        plt.savefig(viz_dir / 'confidence_accuracy_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved in: {viz_dir}")
        
    def generate_comprehensive_report(self):
        """Generate comprehensive error profile report"""
        
        print(f"\n📋 Generating comprehensive error profile report...")
        
        # Run all analyses
        distribution_analysis = self.analyze_error_distributions()
        vulnerability_zones = self.identify_vulnerability_zones()
        pattern_analysis = self.analyze_multi_feature_patterns()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Compile report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'analysis_scope': {
                'total_established_matches': len(self.established_predictions),
                'correct_predictions': len(self.predictions_df[self.predictions_df['correct'] == True]),
                'error_predictions': len(self.predictions_df[self.predictions_df['correct'] == False]),
                'error_rate': (len(self.predictions_df[self.predictions_df['correct'] == False]) / 
                              len(self.predictions_df)) if len(self.predictions_df) > 0 else 0
            },
            'feature_analysis': distribution_analysis,
            'vulnerability_zones': vulnerability_zones,
            'error_patterns': pattern_analysis,
            'key_insights': self.extract_key_insights(distribution_analysis, vulnerability_zones, pattern_analysis),
            'recommendations': self.generate_recommendations(distribution_analysis, vulnerability_zones, pattern_analysis)
        }
        
        # Save report
        output_dir = Path("results/error_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"error_profile_quantitative_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"✅ Comprehensive report saved: {report_file}")
        
        return report
        
    def extract_key_insights(self, dist_analysis, vuln_zones, patterns):
        """Extract actionable insights from analyses"""
        
        insights = []
        
        # Feature significance insights
        significant_features = [f for f, data in dist_analysis.items() if data.get('significant', False)]
        
        if significant_features:
            insights.append({
                'type': 'significant_features',
                'description': f"Features with statistically different distributions in errors: {', '.join(significant_features)}",
                'impact': 'high',
                'action': 'Focus anti-bias engineering on these features'
            })
            
        # Vulnerability zones
        high_risk_zones = []
        for feature, zones in vuln_zones.items():
            for zone, data in zones.items():
                if data['error_rate'] > 0.6:  # 60%+ error rate
                    high_risk_zones.append(f"{feature}[{zone}]: {data['error_rate']:.1%}")
                    
        if high_risk_zones:
            insights.append({
                'type': 'vulnerability_zones',
                'description': f"High-risk zones identified: {', '.join(high_risk_zones)}",
                'impact': 'high',
                'action': 'Create specific features to handle these ranges'
            })
            
        # Pattern insights
        if patterns:
            most_common = max(patterns, key=lambda x: x['matches'])
            insights.append({
                'type': 'dominant_error_pattern',
                'description': f"Most common error pattern: {most_common['pattern']} ({most_common['matches']} matches)",
                'impact': 'medium',
                'action': 'Design features specifically to detect this pattern'
            })
            
        return insights
        
    def generate_recommendations(self, dist_analysis, vuln_zones, patterns):
        """Generate specific recommendations for model improvement"""
        
        recommendations = []
        
        # Feature engineering recommendations
        if 'market_entropy_norm' in [f for f, d in dist_analysis.items() if d.get('significant', False)]:
            recommendations.append({
                'category': 'feature_engineering',
                'priority': 'high',
                'action': 'Create uncertainty_amplified feature: market_entropy_norm * elo_uncertainty_factor',
                'rationale': 'Market entropy shows significant difference between correct/error predictions'
            })
            
        if 'elo_diff_normalized' in [f for f, d in dist_analysis.items() if d.get('significant', False)]:
            recommendations.append({
                'category': 'feature_engineering', 
                'priority': 'high',
                'action': 'Add elo_confidence_zone feature to identify close matches (0.45-0.55 range)',
                'rationale': 'Close Elo matches show higher error rates'
            })
            
        # Model architecture recommendations
        if any(p['matches'] >= 3 for p in patterns):
            recommendations.append({
                'category': 'model_architecture',
                'priority': 'medium',
                'action': 'Implement binary Home/Not-Home classifier as first stage',
                'rationale': 'Multiple error patterns suggest need for specialized approach to home bias'
            })
            
        # Validation recommendations  
        recommendations.append({
            'category': 'validation',
            'priority': 'high',
            'action': 'Use Group K-Fold validation by team to prevent overfitting',
            'rationale': 'Error analysis suggests model may be memorizing team-specific patterns'
        })
        
        return recommendations
        
def main():
    """Main execution function"""
    
    # Configuration
    report_file = "results/rolling_validation_2025_26/rolling_validation_report_20250914_182914.json"
    dataset_file = "data/processed/v15_final_enhanced.csv"
    
    # Initialize analyzer
    analyzer = ErrorProfileAnalyzer(report_file, dataset_file)
    
    print("🔬 EPL 2025-26 ERROR PROFILE QUANTITATIVE ANALYSIS")
    print("="*60)
    
    # Load data
    analyzer.load_data()
    
    # Create features dataframe
    analyzer.create_features_dataframe()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Display key insights
    print(f"\n💡 KEY INSIGHTS:")
    for insight in report['key_insights']:
        print(f"   🎯 {insight['description']}")
        print(f"      → Action: {insight['action']}")
        
    print(f"\n🚀 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   📋 {rec['category'].title()}: {rec['action']}")
        print(f"      Rationale: {rec['rationale']}")
        
    print(f"\n🎉 Quantitative error profile analysis completed!")
    
    return analyzer, report

if __name__ == "__main__":
    analyzer, report = main()