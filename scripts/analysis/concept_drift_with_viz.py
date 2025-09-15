#!/usr/bin/env python3
"""
Concept Drift Detection with Visualizations for EPL 2024-25 vs 2025-26

Statistical analysis to detect if the fundamental dynamics of EPL football
have changed between seasons, which could explain model performance issues.

Implements Kolmogorov-Smirnov tests + advanced visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp, mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class ConceptDriftDetector:
    """
    Detect concept drift between EPL seasons using statistical tests
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.season_2024_25 = None
        self.season_2025_26 = None
        
        # Key features to analyze for drift
        self.drift_features = [
            'elo_diff_normalized',
            'form_diff_normalized',
            'market_entropy_norm',
            'home_xg_eff_10',
            'away_xg_eff_10', 
            'shots_diff_normalized',
            'corners_diff_normalized',
            'away_goals_sum_5'
        ]
        
        # Result distribution features
        self.result_features = ['FullTimeResult']
        
    def load_and_prepare_data(self):
        """Load dataset and split by seasons"""
        
        print("📊 Loading and preparing data for concept drift analysis...")
        
        # Load dataset
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        
        print(f"✅ Loaded dataset: {len(self.dataset)} matches")
        
        # Split by seasons
        self.season_2024_25 = self.dataset[self.dataset['Season'] == '2024-2025']
        self.season_2025_26 = self.dataset[self.dataset['Season'] == '2025-2026']
        
        print(f"📈 Season breakdown:")
        print(f"   2024-25: {len(self.season_2024_25)} matches")
        print(f"   2025-26: {len(self.season_2025_26)} matches (J1-4 EPL)")
        
        # Remove promoted teams from 2025-26 for fair comparison
        promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
        
        self.season_2025_26_established = self.season_2025_26[
            (~self.season_2025_26['HomeTeam'].isin(promoted_teams)) &
            (~self.season_2025_26['AwayTeam'].isin(promoted_teams))
        ]
        
        print(f"   2025-26 (established only): {len(self.season_2025_26_established)} matches")
        
        if len(self.season_2024_25) == 0:
            print("⚠️  Warning: No 2024-25 data found. Using 2023-24 as baseline.")
            self.season_2024_25 = self.dataset[self.dataset['Season'] == '2023-2024']
            print(f"   Using 2023-24: {len(self.season_2024_25)} matches")
            
    def test_feature_distributions(self):
        """Test for distributional changes in key features"""
        
        print(f"\n🔍 CONCEPT DRIFT DETECTION - FEATURE DISTRIBUTIONS")
        print("="*70)
        
        drift_results = {}
        
        print(f"{'Feature':<25} | {'KS Statistic':<12} | {'P-Value':<10} | {'Drift Detected'}")
        print("-" * 70)
        
        for feature in self.drift_features:
            if feature not in self.season_2024_25.columns or feature not in self.season_2025_26_established.columns:
                print(f"{feature:<25} | Feature not found in data")
                continue
                
            # Get clean data (remove NaN)
            data_2024 = self.season_2024_25[feature].dropna()
            data_2025 = self.season_2025_26_established[feature].dropna()
            
            if len(data_2024) == 0 or len(data_2025) == 0:
                print(f"{feature:<25} | Insufficient data")
                continue
                
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = ks_2samp(data_2024, data_2025)
            
            # Mann-Whitney U test (non-parametric)
            mw_statistic, mw_p_value = mannwhitneyu(data_2024, data_2025, alternative='two-sided')
            
            # Determine if drift detected
            drift_detected = ks_p_value < 0.05 or mw_p_value < 0.05
            drift_status = "🚨 YES" if drift_detected else "✅ NO"
            
            print(f"{feature:<25} | {ks_statistic:<12.4f} | {ks_p_value:<10.4f} | {drift_status}")
            
            # Store results
            drift_results[feature] = {
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'mw_statistic': mw_statistic,
                'mw_p_value': mw_p_value,
                'drift_detected': drift_detected,
                'data_2024_mean': data_2024.mean(),
                'data_2024_std': data_2024.std(),
                'data_2025_mean': data_2025.mean(),
                'data_2025_std': data_2025.std(),
                'effect_size': abs(data_2024.mean() - data_2025.mean()) / np.sqrt((data_2024.var() + data_2025.var()) / 2)
            }
            
        return drift_results
        
    def test_result_distributions(self):
        """Test for changes in match outcome distributions"""
        
        print(f"\n🎯 CONCEPT DRIFT - MATCH OUTCOME DISTRIBUTIONS")
        print("="*50)
        
        # Get result distributions
        results_2024 = self.season_2024_25['FullTimeResult'].value_counts(normalize=True).sort_index()
        results_2025 = self.season_2025_26_established['FullTimeResult'].value_counts(normalize=True).sort_index()
        
        print(f"📊 Result Distribution Comparison:")
        print(f"{'Outcome':<10} | {'2024-25':<10} | {'2025-26':<10} | {'Change'}")
        print("-" * 45)
        
        outcome_changes = {}
        for outcome in ['H', 'D', 'A']:
            perc_2024 = results_2024.get(outcome, 0) * 100
            perc_2025 = results_2025.get(outcome, 0) * 100
            change = perc_2025 - perc_2024
            
            change_symbol = "📈" if change > 5 else "📉" if change < -5 else "➡️"
            
            print(f"{outcome:<10} | {perc_2024:<10.1f}% | {perc_2025:<10.1f}% | {change:+.1f}pp {change_symbol}")
            
            outcome_changes[outcome] = {
                'perc_2024': perc_2024,
                'perc_2025': perc_2025,
                'change_pp': change
            }
            
        # Chi-square test for independence
        if len(results_2024) > 0 and len(results_2025) > 0:
            # Create contingency table
            contingency_data = []
            for outcome in ['H', 'D', 'A']:
                count_2024 = (results_2024.get(outcome, 0) * len(self.season_2024_25))
                count_2025 = (results_2025.get(outcome, 0) * len(self.season_2025_26_established))
                contingency_data.append([count_2024, count_2025])
                
            contingency_table = np.array(contingency_data)
            
            try:
                chi2, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
                
                print(f"\n📈 Statistical Test Results:")
                print(f"   Chi-square statistic: {chi2:.4f}")
                print(f"   P-value: {chi2_p_value:.4f}")
                print(f"   Result distribution drift: {'🚨 SIGNIFICANT' if chi2_p_value < 0.05 else '✅ NOT SIGNIFICANT'}")
                
                outcome_changes['statistical_test'] = {
                    'chi2_statistic': chi2,
                    'p_value': chi2_p_value,
                    'drift_detected': chi2_p_value < 0.05
                }
                
            except Exception as e:
                print(f"   Statistical test failed: {e}")
                
        return outcome_changes
        
    def analyze_correlation_changes(self):
        """Analyze changes in feature correlations between seasons"""
        
        print(f"\n🔗 CONCEPT DRIFT - FEATURE CORRELATION CHANGES")
        print("="*50)
        
        # Calculate correlation matrices
        corr_2024 = self.season_2024_25[self.drift_features].corr()
        corr_2025 = self.season_2025_26_established[self.drift_features].corr()
        
        # Calculate correlation differences
        corr_diff = corr_2025 - corr_2024
        
        # Find significant correlation changes
        significant_changes = []
        
        for i, feature1 in enumerate(self.drift_features):
            for j, feature2 in enumerate(self.drift_features):
                if i < j:  # Only upper triangle
                    if not pd.isna(corr_diff.loc[feature1, feature2]):
                        change = corr_diff.loc[feature1, feature2]
                        if abs(change) > 0.2:  # Significant change threshold
                            significant_changes.append({
                                'feature_pair': f"{feature1} <-> {feature2}",
                                'corr_2024': corr_2024.loc[feature1, feature2],
                                'corr_2025': corr_2025.loc[feature1, feature2],
                                'change': change
                            })
                            
        if significant_changes:
            print(f"📊 Significant Correlation Changes (|Δ| > 0.2):")
            for change in significant_changes:
                direction = "📈" if change['change'] > 0 else "📉"
                print(f"   {direction} {change['feature_pair']}: {change['corr_2024']:.3f} → {change['corr_2025']:.3f} (Δ{change['change']:+.3f})")
        else:
            print("✅ No significant correlation changes detected.")
            
        return {
            'corr_2024': corr_2024,
            'corr_2025': corr_2025,
            'corr_diff': corr_diff,
            'significant_changes': significant_changes
        }
        
    def generate_visualizations(self, drift_results, outcome_changes, correlation_analysis):
        """Generate comprehensive visualizations"""
        
        print(f"\n📊 Generating concept drift visualizations...")
        
        # Create output directory
        viz_dir = Path("results/concept_drift_viz")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature distributions comparison
        n_features = len(self.drift_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle('Feature Distributions: 2024-25 vs 2025-26 (Established Teams Only)', fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, feature in enumerate(self.drift_features):
            if feature not in drift_results:
                continue
                
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Get data
            data_2024 = self.season_2024_25[feature].dropna()
            data_2025 = self.season_2025_26_established[feature].dropna()
            
            # Plot distributions
            ax.hist(data_2024, alpha=0.7, label=f'2024-25 (n={len(data_2024)})', 
                   color='blue', bins=20, density=True)
            ax.hist(data_2025, alpha=0.7, label=f'2025-26 (n={len(data_2025)})', 
                   color='red', bins=20, density=True)
            
            # Add drift indicator
            if drift_results[feature]['drift_detected']:
                ax.text(0.05, 0.95, '🚨 DRIFT', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                       fontsize=10, verticalalignment='top')
                       
            ax.set_title(f'{feature}\np-value: {drift_results[feature]["ks_p_value"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_distributions_drift.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Result distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Match Outcome Distribution Changes', fontsize=16)
        
        # Bar plot
        outcomes = ['H', 'D', 'A']
        perc_2024 = [outcome_changes.get(outcome, {}).get('perc_2024', 0) for outcome in outcomes]
        perc_2025 = [outcome_changes.get(outcome, {}).get('perc_2025', 0) for outcome in outcomes]
        
        x = np.arange(len(outcomes))
        width = 0.35
        
        ax1.bar(x - width/2, perc_2024, width, label='2024-25', color='blue', alpha=0.7)
        ax1.bar(x + width/2, perc_2025, width, label='2025-26', color='red', alpha=0.7)
        
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Outcome Distribution Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(outcomes)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Change plot
        changes = [outcome_changes.get(outcome, {}).get('change_pp', 0) for outcome in outcomes]
        colors = ['green' if c > 0 else 'red' for c in changes]
        
        ax2.bar(outcomes, changes, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Percentage Point Change')
        ax2.set_title('Change in Outcome Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'outcome_distribution_drift.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Feature Correlation Changes', fontsize=16)
        
        # 2024-25 correlations
        sns.heatmap(correlation_analysis['corr_2024'], annot=True, cmap='coolwarm', 
                   center=0, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('2024-25 Correlations')
        
        # 2025-26 correlations  
        sns.heatmap(correlation_analysis['corr_2025'], annot=True, cmap='coolwarm',
                   center=0, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('2025-26 Correlations')
        
        # Differences
        sns.heatmap(correlation_analysis['corr_diff'], annot=True, cmap='RdBu_r',
                   center=0, ax=axes[2], cbar_kws={'shrink': 0.8})
        axes[2].set_title('Correlation Changes')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved in: {viz_dir}")
        
    def generate_comprehensive_report(self):
        """Generate comprehensive concept drift report"""
        
        print(f"\n📋 Generating comprehensive concept drift report...")
        
        # Run all analyses
        drift_results = self.test_feature_distributions()
        outcome_changes = self.test_result_distributions()
        correlation_analysis = self.analyze_correlation_changes()
        
        # Generate visualizations
        self.generate_visualizations(drift_results, outcome_changes, correlation_analysis)
        
        # Count significant drifts
        significant_drifts = sum(1 for result in drift_results.values() if result['drift_detected'])
        
        # Compile report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'analysis_scope': {
                'season_2024_25_matches': len(self.season_2024_25),
                'season_2025_26_matches': len(self.season_2025_26),
                'season_2025_26_established_only': len(self.season_2025_26_established),
                'features_analyzed': len(self.drift_features)
            },
            'feature_drift_analysis': drift_results,
            'outcome_distribution_changes': outcome_changes,
            'correlation_changes': {
                'significant_changes': correlation_analysis['significant_changes'],
                'n_significant_changes': len(correlation_analysis['significant_changes'])
            },
            'summary': {
                'features_with_drift': significant_drifts,
                'total_features': len(drift_results),
                'drift_percentage': (significant_drifts / len(drift_results)) * 100 if drift_results else 0,
                'outcome_distribution_changed': outcome_changes.get('statistical_test', {}).get('drift_detected', False),
                'major_concept_drift_detected': significant_drifts > len(drift_results) * 0.3  # >30% features
            },
            'key_insights': self.extract_drift_insights(drift_results, outcome_changes, correlation_analysis),
            'recommendations': self.generate_drift_recommendations(drift_results, outcome_changes, correlation_analysis)
        }
        
        # Save report
        output_dir = Path("results/concept_drift_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"concept_drift_analysis_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
            
        print(f"✅ Comprehensive drift report saved: {report_file}")
        
        return report
        
    def extract_drift_insights(self, drift_results, outcome_changes, correlation_analysis):
        """Extract key insights from drift analysis"""
        
        insights = []
        
        # Feature drift insights
        drifted_features = [f for f, result in drift_results.items() if result['drift_detected']]
        
        if drifted_features:
            insights.append({
                'type': 'feature_drift',
                'description': f"Significant distributional changes detected in: {', '.join(drifted_features)}",
                'impact': 'high',
                'explanation': 'Model trained on different feature distributions may not generalize well'
            })
        else:
            insights.append({
                'type': 'feature_stability',
                'description': "No significant feature drift detected - underlying data patterns remain stable",
                'impact': 'positive',
                'explanation': 'Model performance issues unlikely due to changing data patterns'
            })
            
        # Outcome distribution insights
        if outcome_changes.get('statistical_test', {}).get('drift_detected', False):
            insights.append({
                'type': 'outcome_drift',
                'description': "Match outcome distribution has changed significantly between seasons",
                'impact': 'high',
                'explanation': 'Fundamental league dynamics may have shifted'
            })
            
        # Large individual changes
        large_changes = [outcome for outcome, data in outcome_changes.items() 
                        if isinstance(data, dict) and abs(data.get('change_pp', 0)) > 5]
        
        if large_changes:
            insights.append({
                'type': 'outcome_pattern_change',
                'description': f"Large changes in outcome frequency: {large_changes}",
                'impact': 'medium',
                'explanation': 'Model may need recalibration for new outcome patterns'
            })
            
        return insights
        
    def generate_drift_recommendations(self, drift_results, outcome_changes, correlation_analysis):
        """Generate recommendations based on drift analysis"""
        
        recommendations = []
        
        # Feature drift recommendations
        drifted_features = [f for f, result in drift_results.items() if result['drift_detected']]
        
        if drifted_features:
            recommendations.append({
                'category': 'data_preprocessing',
                'priority': 'high',
                'action': f'Recalibrate features with significant drift: {", ".join(drifted_features)}',
                'method': 'Apply feature scaling/normalization based on recent data distribution'
            })
            
            recommendations.append({
                'category': 'model_training',
                'priority': 'high', 
                'action': 'Consider domain adaptation techniques for drifted features',
                'method': 'Weight recent data more heavily or use transfer learning approaches'
            })
        else:
            recommendations.append({
                'category': 'model_diagnosis',
                'priority': 'medium',
                'action': 'Focus on model architecture issues rather than data drift',
                'method': 'Investigate overfitting, feature importance, and bias issues'
            })
            
        # Outcome distribution recommendations
        if outcome_changes.get('statistical_test', {}).get('drift_detected', False):
            recommendations.append({
                'category': 'model_calibration',
                'priority': 'high',
                'action': 'Recalibrate model probabilities for new outcome distributions',
                'method': 'Use Platt scaling or isotonic regression on recent data'
            })
            
        # Correlation change recommendations
        if len(correlation_analysis.get('significant_changes', [])) > 0:
            recommendations.append({
                'category': 'feature_engineering',
                'priority': 'medium',
                'action': 'Review feature interactions given correlation changes',
                'method': 'Consider new interaction features or remove highly correlated pairs'
            })
            
        return recommendations

def main():
    """Main execution function"""
    
    # Configuration
    dataset_file = "data/processed/v15_final_enhanced.csv"
    
    # Initialize detector
    detector = ConceptDriftDetector(dataset_file)
    
    print("🔍 EPL CONCEPT DRIFT DETECTION ANALYSIS")
    print("="*50)
    print("Comparing 2024-25 vs 2025-26 (Established Teams Only)")
    
    # Load and prepare data
    detector.load_and_prepare_data()
    
    # Generate comprehensive report
    report = detector.generate_comprehensive_report()
    
    # Display summary
    summary = report['summary']
    print(f"\n🎯 CONCEPT DRIFT SUMMARY:")
    print(f"   Features with drift: {summary['features_with_drift']}/{summary['total_features']} ({summary['drift_percentage']:.1f}%)")
    print(f"   Outcome distribution changed: {'🚨 YES' if summary['outcome_distribution_changed'] else '✅ NO'}")
    print(f"   Major concept drift: {'🚨 DETECTED' if summary['major_concept_drift_detected'] else '✅ NOT DETECTED'}")
    
    # Display insights
    print(f"\n💡 KEY INSIGHTS:")
    for insight in report['key_insights']:
        impact_emoji = "🚨" if insight['impact'] == 'high' else "🟡" if insight['impact'] == 'medium' else "✅"
        print(f"   {impact_emoji} {insight['description']}")
        
    # Display recommendations
    print(f"\n🚀 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        priority_emoji = "🚨" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "ℹ️"
        print(f"   {priority_emoji} {rec['category'].title()}: {rec['action']}")
        
    print(f"\n🎉 Concept drift analysis completed!")
    
    return detector, report

if __name__ == "__main__":
    detector, report = main()